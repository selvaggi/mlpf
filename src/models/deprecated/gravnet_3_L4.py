import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_add
from torch.nn import Parameter
from src.layers.GravNetConv3 import GravNetConv, WeirdBatchNorm
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.layers.object_cond import (
    calc_LV_Lbeta,
    get_clustering,
    calc_LV_Lbeta_inference,
)
from src.layers.obj_cond_inf import calc_energy_loss
from src.layers.mlp_readout_layer import MLPReadout
from src.layers.inference_oc import create_and_store_graph_output
from src.models.gravnet_calibration import (
    object_condensation_loss2,
    obtain_batch_numbers,
)
from src.utils.save_features import save_features
import lightning as L
from src.utils.nn.tools import log_losses_wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.post_clustering_features import get_post_clustering_features
from src.models.GattedGCN_correction import GraphTransformerNet, GCNNet, LinearGNNLayer
from src.layers.inference_oc import hfdb_obtain_labels
import torch_cmspepr
from src.layers.inference_oc import match_showers
from lightning.pytorch.callbacks import BaseFinetuning
import os
import wandb
from src.layers.obtain_statistics import (
    obtain_statistics_graph,
    create_stats_dict,
    save_stat_dict,
    plot_distributions,
)
from src.models.point_transformer import PointTransformer


class GravnetModel(L.LightningModule):
    def __init__(
        self,
        args,
        dev,
        input_dim: int = 8,
        output_dim: int = 4,
        n_postgn_dense_blocks: int = 3,
        n_gravnet_blocks: int = 4,
        clust_space_norm: str = "twonorm",
        k_gravnet: int = 7,
        activation: str = "elu",
        weird_batchnom=False,
        use_correction=False,
    ):

        super(GravnetModel, self).__init__()
        self.dev = dev
        self.loss_final = 0
        self.number_b = 0
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []
        self.args = args
        self.validation_step_outputs = []
        assert activation in ["relu", "tanh", "sigmoid", "elu"]
        acts = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
        }
        self.act = acts[activation]

        N_NEIGHBOURS = [16, 128, 16, 128]
        TOTAL_ITERATIONS = len(N_NEIGHBOURS)
        self.return_graphs = False
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = TOTAL_ITERATIONS
        self.n_postgn_dense_blocks = n_postgn_dense_blocks
        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_1 = WeirdBatchNorm(self.input_dim)
        else:
            self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim)
        self.backbone = PointTransformer(
            300,
            1,
            3,
            2,
            4,
            32,
            None,
            16,
        )
        self.clustering = nn.Linear(64, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(64, 1)

    def forward(self, g, y, step_count):
        h = g.ndata["h"]
        original_coords = x[:, 0:3]
        g.ndata["original_coords"] = original_coords
        x = original_coords

        x = self.ScaledGooeyBatchNorm2_1(x)
        x = x.unsqueeze(0)
        h, hidden_state = self.backbone(x)
        x_cluster_coord = self.clustering(h)
        beta = self.beta(h)
        if self.args.tracks:
            mask = g.ndata["hit_type"] == 1
            beta[mask] = 9
        g.ndata["final_cluster"] = x_cluster_coord
        g.ndata["beta"] = beta.view(-1)
        if self.trainer.is_global_zero and (step_count % 100 == 0):
            PlotCoordinates(
                g,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
                epoch=str(self.current_epoch),
                step_count=step_count,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)

        pred_energy_corr = torch.ones_like(beta.view(-1, 1))
        return x, pred_energy_corr, 0

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]
        # if self.trainer.is_global_zero and self.current_epoch == 0:
        #     self.stat_dict = obtain_statistics_graph(self.stat_dict, y, batch_g)
        if self.trainer.is_global_zero:
            if self.args.correction:
                (
                    model_output,
                    e_cor1,
                    true_e,
                    sum_e,
                    new_graphs,
                    batch_idx,
                    graph_level_features,
                ) = self(batch_g, y, batch_idx)
                e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
                loss_ll = 0
            else:
                model_output, e_cor, loss_ll = self(batch_g, y, batch_idx)
        else:
            if self.args.correction:
                (
                    model_output,
                    e_cor1,
                    true_e,
                    sum_e,
                    new_graphs,
                    batch_idx,
                    graph_level_features,
                ) = self(batch_g, y, 1)
                e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
                loss_ll = 0
            else:
                model_output, e_cor, loss_ll = self(batch_g, y, 1)
                e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        if self.args.correction:
            print(model_output.shape, e_cor.shape, e_cor1.shape)
            e_cor1 += 1.0  # We regress the number around zero!!! # TODO: uncomment this if needed
        """energies_sums_features = new_graphs.ndata["h"][:, 15]
        energies_sums = [sum_e[i] for i in batch_idx]
        energies_sums = torch.tensor(energies_sums).to(energies_sums_features.device).flatten()
        print(energies_sums[energies_sums != energies_sums_features])
        print(energies_sums_features[energies_sums != energies_sums_features])
        assert (torch.abs(energies_sums - energies_sums_features) < 0.001).all()"""
        # print(model_output.shape, e_cor.shape, e_cor1.shape)
        # e_cor1 += 1.  # We regress the number around zero!!! # TODO: uncomment this if needed
        (loss, losses, loss_E, loss_E_frac_true,) = object_condensation_loss2(
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
            hgcalloss=self.args.hgcalloss,
        )
        loss = loss  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
        if self.trainer.is_global_zero:
            log_losses_wandb(True, batch_idx, 0, losses, loss, loss_ll)

        self.loss_final = loss + self.loss_final
        self.number_b = self.number_b + 1
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
        if self.args.correction:
            (
                model_output,
                e_cor1,
                true_e,
                sum_e,
                new_graphs,
                batch_id,
                graph_level_features,
            ) = self(batch_g, y, 1)
            loss_ll = 0
            e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        else:
            model_output, e_cor1, loss_ll = self(batch_g, y, 1)
            loss_ll = 0
            e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        preds = model_output.squeeze()
        (loss, losses, loss_E, loss_E_frac_true,) = object_condensation_loss2(
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
            hgcalloss=self.args.hgcalloss,
        )
        loss_ec = 0

        if self.trainer.is_global_zero:
            log_losses_wandb(
                True, batch_idx, 0, losses, loss, loss_ll, loss_ec, val=True
            )
        self.validation_step_outputs.append([model_output, e_cor, batch_g, y])
        if self.args.predict:
            model_output1 = torch.cat((model_output, e_cor.view(-1, 1)), dim=1)
            if self.args.correction:
                e_corr = e_cor1
            else:
                e_corr = None
            (df_batch_pandora, df_batch1, df_batch) = create_and_store_graph_output(
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
            )
            self.df_showers.append(df_batch)
            self.df_showers_pandora.append(df_batch_pandora)
            self.df_showes_db.append(df_batch1)

    def on_train_epoch_end(self):

        self.log("train_loss_epoch", self.loss_final / self.number_b)

    def on_train_epoch_start(self):
        if self.current_epoch == 0 and self.trainer.is_global_zero:
            stats_dict = create_stats_dict(self.beta.weight.device)
            self.stat_dict = stats_dict
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            self.ScaledGooeyBatchNorm2_1.momentum = 0
            # self.ScaledGooeyBatchNorm2_2.momentum = 0
            # for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
            #     gravnet_block.batchnorm_gravnet1.momentum = 0
            #     gravnet_block.batchnorm_gravnet2.momentum = 0

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            if self.args.predict:
                from src.layers.inference_oc import store_at_batch_end
                import pandas as pd

                self.df_showers = pd.concat(self.df_showers)
                self.df_showers_pandora = pd.concat(self.df_showers_pandora)
                self.df_showes_db = pd.concat(self.df_showes_db)
                store_at_batch_end(
                    path_save=os.path.join(
                        self.args.model_prefix, "showers_df_evaluation"
                    ),
                    df_batch=self.df_showers,
                    df_batch_pandora=self.df_showers_pandora,
                    df_batch1=self.df_showes_db,
                    step=0,
                    predict=True,
                    store=True,
                )
            else:
                model_output = self.validation_step_outputs[0][0]
                e_corr = self.validation_step_outputs[0][1]
                batch_g = self.validation_step_outputs[0][2]
                y = self.validation_step_outputs[0][3]
                model_output1 = torch.cat((model_output, e_corr.view(-1, 1)), dim=1)
                create_and_store_graph_output(
                    batch_g,
                    model_output1,
                    y,
                    0,
                    0,
                    0,
                    path_save=os.path.join(
                        self.args.model_prefix, "showers_df_evaluation"
                    ),
                    store=True,
                    predict=False,
                    tracks=self.args.tracks,
                )
        self.validation_step_outputs = []

    def configure_optimizers(self):
        print("Configuring optimizer!")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3
        )
        print("Optimizer params:", filter(lambda p: p.requires_grad, self.parameters()))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=3),
                "interval": "epoch",
                "monitor": "train_loss_epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
