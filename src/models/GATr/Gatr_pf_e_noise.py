from os import path
import sys

# sys.path.append(
#     path.abspath("/afs/cern.ch/work/m/mgarciam/private/geometric-algebra-transformer/")
# )
# sys.path.append(path.abspath("/mnt/proj3/dd-23-91/cern/geometric-algebra-transformer/"))
from time import time
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar, extract_point, embed_scalar
from torch_scatter import scatter_add, scatter_mean
import torch
import torch.nn as nn
from src.logger.plotting_tools import PlotCoordinates
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.layers.obj_cond_inf import calc_energy_loss
from src.layers.object_cond import object_condensation_loss2
from src.utils.pid_conversion import pid_conversion_dict
from src.layers.utils_training import obtain_batch_numbers, obtain_clustering_for_matched_showers


from src.models.energy_correction_NN import (
    EnergyCorrection
)
from src.layers.inference_oc import create_and_store_graph_output
import lightning as L
from src.utils.nn.tools import log_losses_wandb_tracking
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from src.layers.inference_oc_tracks import (
    evaluate_efficiency_tracks,
    store_at_batch_end,
)
from xformers.ops.fmha import BlockDiagonalMask
import os
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

# from src.layers.obtain_statistics import (
#     obtain_statistics_graph_tracking,
#     create_stats_dict,
#     save_stat_dict,
#     plot_distributions,
# )
from src.utils.nn.tools import log_losses_wandb
import torch.nn.functional as F




class ExampleWrapper(L.LightningModule):
    """Example wrapper around a GATr model.

    Expects input data that consists of a point cloud: one 3D point for each item in the data.
    Returns outputs that consists of one scalar number for the whole dataset.

    Parameters
    ----------
    blocks : int
        Number of transformer blocks
    hidden_mv_channels : int
        Number of hidden multivector channels
    hidden_s_channels : int
        Number of hidden scalar channels
    """

    def __init__(
        self,
        args,
        dev,
        input_dim: int = 5,
        output_dim: int = 4,
        n_postgn_dense_blocks: int = 3,
        n_gravnet_blocks: int = 4,
        clust_space_norm: str = "twonorm",
        k_gravnet: int = 7,
        activation: str = "elu",
        weird_batchnom=False,
        blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=64,
        config=None
    ):
        super().__init__()
        self.strict_loading = False
        self.input_dim = 3
        self.output_dim = 4
        self.loss_final = 0
        self.number_b = 0
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []
        self.args = args
        self.dev = dev
        self.config = config
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=2,
            out_s_channels=1,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        )
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        self.clustering = nn.Linear(3, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(2, 1)
        # Initialize the energy correction module
        if self.args.correction:
            self.energy_correction = EnergyCorrection(self)
            # Not a pytorch module! Otherwise it causes a recursion error when loading model weights
            self.ec_model_wrapper_charged = self.energy_correction.model_charged
            self.ec_model_wrapper_neutral = self.energy_correction.model_neutral
            self.pids_neutral = self.energy_correction.pids_neutral
            self.pids_charged = self.energy_correction.pids_charged
        else:
            self.pids_neutral = []
            self.pids_charged = []
        # freeze these models completely
        # for param in self.ec_model_wrapper_charged.model.parameters():
        #    param.requires_grad = False
        # for param in self.ec_model_wrapper_neutral.model.parameters():
        #    param.requires_grad = False

    def forward(self, g, y, step_count, eval="", return_train=False):
        inputs = g.ndata["pos_hits_xyz"]
        if self.trainer.is_global_zero and step_count % 500 == 0:
            g.ndata["original_coords"] = g.ndata["pos_hits_xyz"]
            PlotCoordinates(
                g,
                path="input_coords",
                outdir=self.args.model_prefix,
                # features_type="ones",
                predict=self.args.predict,
                epoch=str(self.current_epoch) + eval,
                step_count=step_count,
            )
        inputs_scalar = g.ndata["hit_type"].view(-1, 1)
        inputs = self.ScaledGooeyBatchNorm2_1(inputs)
        # inputs = inputs.unsqueeze(0)
        embedded_inputs = embed_point(inputs) + embed_scalar(inputs_scalar)
        embedded_inputs = embedded_inputs.unsqueeze(
            -2
        )  # (batch_size*num_points, 1, 16)
        mask = self.build_attention_mask(g)
        scalars = torch.zeros((inputs.shape[0], 1))
        scalars = g.ndata["h"][:, -2:]  # this corresponds to e,p
        # Pass data through GATr
        forward_time_start = time()
        embedded_outputs, scalar_outputs = self.gatr(
            embedded_inputs, scalars=scalars, attention_mask=mask
        )  # (..., num_points, 1, 16)
        forward_time_end = time()
        # wandb.log({"time_gatr_pass": forward_time_end - forward_time_start})
        points = extract_point(embedded_outputs[:, 0, :])

        # Extract scalar and aggregate outputs from point cloud
        nodewise_outputs = extract_scalar(embedded_outputs)  # (..., num_points, 1, 1)
        x_point = points
        x_scalar = torch.cat(
            (nodewise_outputs.view(-1, 1), scalar_outputs.view(-1, 1)), dim=1
        )
        x_cluster_coord = self.clustering(x_point)
        beta = self.beta(x_scalar)
        # if self.args.tracks:
        #     mask = g.ndata["hit_type"] == 1
        #     beta[mask] = 9
        g.ndata["final_cluster"] = x_cluster_coord
        g.ndata["beta"] = beta.view(-1)
        if self.trainer.is_global_zero and step_count % 500 == 0:
            PlotCoordinates(
                g,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
                epoch=str(self.current_epoch) + eval,
                step_count=step_count,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        pred_energy_corr = torch.ones_like(beta.view(-1, 1)).flatten()
        if self.args.correction:
            result = self.energy_correction.forward_correction(g, x, y, return_train)
            # loop through params and print the ones without grad
            #for name, param in self.named_parameters():
            #    if not param.requires_grad:
            #        print("doesn't have grad", name)
            return result
        else:
            pred_energy_corr = torch.ones_like(beta.view(-1, 1))
            return x, pred_energy_corr, 0, 0



    def build_attention_mask(self, g):
        """Construct attention mask from pytorch geometric batch.

        Parameters
        ----------
        inputs : torch_geometric.data.Batch
            Data batch.

        Returns
        -------
        attention_mask : xformers.ops.fmha.BlockDiagonalMask
            Block-diagonal attention mask: within each sample, each token can attend to each other
            token.
        """
        batch_numbers = obtain_batch_numbers(g)
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]
        initial_time = time()
        if self.trainer.is_global_zero:
            result = self(batch_g, y, batch_idx)
        else:
            result = self(batch_g, y, 1)
        model_output = result[0]
        e_cor = result[1]
        loss_time_start = time()
        (loss, losses,) = object_condensation_loss2(
            batch_g,
            model_output,
            e_cor,
            y,
            clust_loss_only=True,
            add_energy_loss=False,
            calc_e_frac_loss=False,
            q_min=self.args.qmin,
            frac_clustering_loss=self.args.frac_cluster_loss,
            attr_weight=self.args.L_attractive_weight,
            repul_weight=self.args.L_repulsive_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            loss_type=self.args.losstype,
        )
        loss_time_end = time()
        misc_time_start = time()
        if self.args.correction:
            loss_EC, loss_pos, loss_neutral_pid, loss_charged_pid = self.energy_correction.get_loss(batch_g, y, result)
            loss = loss + loss_EC + loss_pos + loss_neutral_pid + loss_charged_pid
        if self.trainer.is_global_zero:
            log_losses_wandb(True, batch_idx, 0, losses, loss, 0)
        self.loss_final = loss.item() + self.loss_final
        self.number_b = self.number_b + 1
        del model_output
        del e_cor
        del losses
        # final_time = time()
        # wandb.log({"misc_time_inside_training": final_time - misc_time_start})
        # wandb.log({"training_step_time": final_time - initial_time, "loss_time_inside_training": loss_time_end - loss_time_start})
        return loss

    def validation_step(self, batch, batch_idx):
        cluster_features_path = os.path.join(self.args.model_prefix, "cluster_features")
        show_df_eval_path = os.path.join(
            self.args.model_prefix, "showers_df_evaluation"
        )
        if not os.path.exists(show_df_eval_path):
            os.makedirs(show_df_eval_path)
        if not os.path.exists(cluster_features_path):
            os.makedirs(cluster_features_path)
        self.validation_step_outputs = []
        y = batch[1]
        batch_g = batch[0]
        shap_vals, ec_x = None, None
        if self.args.correction:
            result = self(batch_g, y, 1)
            model_output = result[0]
            outputs = self.energy_correction.get_validation_step_outputs(batch_g, y, result)
            loss_ll = 0
            e_cor1, pred_pos, pred_ref_pt, pred_pid, num_fakes, extra_features, fakes_labels = outputs
            e_cor = e_cor1
        #################################################################
        else:
            model_output, e_cor1, loss_ll, _ = self(batch_g, y, 1)
            loss_ll = 0
            e_cor1 = torch.ones_like(model_output[:, 0].view(-1, 1))
            e_cor = e_cor1
            pred_pos = None
            pred_pid = None
            pred_ref_pt = None
            num_fakes = None
            extra_features = None
        # if self.global_step < 200:
        #     self.args.losstype = "hgcalimplementation"
        # else:
        #     self.args.losstype = "vrepweighted"
        (loss, losses,) = object_condensation_loss2(
            batch_g,
            model_output,
            e_cor1,
            y,
            clust_loss_only=True,
            add_energy_loss=False,
            calc_e_frac_loss=False,
            q_min=self.args.qmin,
            frac_clustering_loss=self.args.frac_cluster_loss,
            attr_weight=self.args.L_attractive_weight,
            repul_weight=self.args.L_repulsive_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            loss_type=self.args.losstype,
        )
        loss_ec = 0
        print("Starting validation step", batch_idx, loss)
        if self.trainer.is_global_zero:
            log_losses_wandb(
                True, batch_idx, 0, losses, loss, loss_ll, loss_ec, val=True
            )
        if self.args.explain_ec:
            self.validation_step_outputs.append(
                [model_output, e_cor, batch_g, y, shap_vals, ec_x, num_fakes]
            )
        else:
            if self.args.correction:
                self.validation_step_outputs.append([model_output, e_cor, batch_g, y, num_fakes])
        if self.args.predict:
            if self.args.correction:
                model_output1 = model_output
                e_corr = e_cor
            else:
                model_output1 = torch.cat((model_output, e_cor.view(-1, 1)), dim=1)
                e_corr = None
            (
                df_batch_pandora,
                df_batch1,
                self.total_number_events,
            ) = create_and_store_graph_output(
                batch_g,
                model_output1,
                y,
                0,
                batch_idx,
                0,
                path_save=show_df_eval_path,
                store=True,
                predict=True,
                e_corr=e_corr,
                tracks=self.args.tracks,
                shap_vals=shap_vals,
                ec_x=ec_x,
                total_number_events=self.total_number_events,
                pred_pos=pred_pos,
                pred_ref_pt=pred_ref_pt,
                pred_pid=pred_pid,
                use_gt_clusters=self.args.use_gt_clusters,
                pids_neutral=self.pids_neutral,
                pids_charged=self.pids_charged,
                number_of_fakes=num_fakes,
                extra_features=extra_features,
                fakes_labels=fakes_labels
            )
            self.df_showers_pandora.append(df_batch_pandora)
            print("Appending another batch", len(df_batch1))
            self.df_showes_db.append(df_batch1)
        del losses
        del loss
        del model_output

    def on_train_epoch_end(self):
        self.log("train_loss_epoch", self.loss_final / self.number_b)

    def on_train_epoch_start(self):
        # if self.trainer.is_global_zero and self.current_epoch == 0:
        #     self.stat_dict = {}
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.total_number_events = 0
        self.make_mom_zero()
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            print("making momentum 0")
            self.ScaledGooeyBatchNorm2_1.momentum = 0

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            if self.args.predict:
                from src.layers.inference_oc import store_at_batch_end
                import pandas as pd
                if self.args.explain_ec:
                    shap_vals = self.validation_step_outputs[0][4]
                    path_shap_vals = os.path.join(
                        self.args.model_prefix, "shap_vals.pkl"
                    )
                    torch.save(shap_vals, path_shap_vals)
                    print("SHAP values saved!")
                # self.df_showers = pd.concat(self.df_showers)
                self.df_showers_pandora = pd.concat(self.df_showers_pandora)
                self.df_showes_db = pd.concat(self.df_showes_db)
                print(self.df_showes_db.keys())
                store_at_batch_end(
                    path_save=os.path.join(
                        self.args.model_prefix, "showers_df_evaluation"
                    ),
                    # df_batch=self.df_showers,
                    df_batch_pandora=self.df_showers_pandora,
                    df_batch1=self.df_showes_db,
                    step=0,
                    predict=True,
                    store=True,
                )
            # else:
            #     model_output = self.validation_step_outputs[0][0]
            #     e_corr = self.validation_step_outputs[0][1]
            #     batch_g = self.validation_step_outputs[0][2]
            #     y = self.validation_step_outputs[0][3]
            #     shap_vals = None
            #     ec_x = None
            #     if self.args.explain_ec:
            #         shap_vals = self.validation_step_outputs[0][4]
            #         ec_x = self.validation_step_outputs[0][5]
            #     if self.args.correction:
            #         model_output1 = model_output
            #         e_corr = e_corr
            #     else:
            #         model_output1 = torch.cat((model_output, e_corr.view(-1, 1)), dim=1)
            #         e_corr = None
            #     create_and_store_graph_output(
            #         batch_g,
            #         model_output1,
            #         y,
            #         0,
            #         0,
            #         0,
            #         path_save=os.path.join(
            #             self.args.model_prefix, "showers_df_evaluation"
            #         ),
            #         store=True,
            #         predict=False,
            #         e_corr=e_corr,
            #         tracks=self.args.tracks,
            #         shap_vals=shap_vals,
            #         ec_x=ec_x,
            #         use_gt_clusters=self.args.use_gt_clusters,
            #     )
            #     del model_output1
            #     del batch_g
        self.validation_step_outputs = []
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=int(7900*3), # for now for testing
        #     eta_min=1e-6,
        # )
        scheduler = CosineAnnealingThenFixedScheduler(optimizer,T_max=int(7900*3), fixed_lr=1e-6 ) #10000
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,  # ReduceLROnPlateau(optimizer, patience=3),
                "interval": "step",
                "monitor": "train_loss_epoch",
                "frequency": 1
            }}
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        # Manually step the scheduler
        scheduler.step()
   



def obtain_batch_numbers(g):
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes))
        num_nodes = gj.number_of_nodes()
    batch = torch.cat(batch_numbers, dim=0)
    return batch




class CosineAnnealingThenFixedScheduler:
    def __init__(self, optimizer, T_max, fixed_lr):
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=fixed_lr)
        self.fixed_lr = 1e-6
        self.T_max = T_max
        self.step_count = 0
        self.optimizer = optimizer

    def step(self):
        if self.step_count < self.T_max:
            self.cosine_scheduler.step()
            # for param_group in self.optimizer.param_groups:
            #     print("before scheduler change", param_group['lr'])
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.fixed_lr
                # print("after scheduler change",param_group['lr'])
        self.step_count += 1

    def get_last_lr(self):
        if self.step_count < self.T_max:
            return self.cosine_scheduler.get_last_lr()
        else:
            return [self.fixed_lr for _ in self.optimizer.param_groups]
    def state_dict(self):
        # Save the state including current step count and cosine scheduler state
        return {
            "step_count": self.step_count,
            "cosine_scheduler_state": self.cosine_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        # Restore step count and cosine scheduler state
        self.step_count = state_dict["step_count"]
        self.cosine_scheduler.load_state_dict(state_dict["cosine_scheduler_state"])