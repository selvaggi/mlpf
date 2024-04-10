from os import path
import sys

# sys.path.append(
#     path.abspath("/afs/cern.ch/work/m/mgarciam/private/geometric-algebra-transformer/")
# )
# sys.path.append(path.abspath("/mnt/proj3/dd-23-91/cern/geometric-algebra-transformer/"))

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
from src.models.gravnet_calibration import (
    object_condensation_loss2,
    obtain_batch_numbers,
)
from src.models.gravnet_3_L import obtain_clustering_for_matched_showers
from src.utils.post_clustering_features import get_post_clustering_features, calculate_eta, calculate_phi
from src.models.energy_correction_NN import NetWrapper
from src.layers.inference_oc import create_and_store_graph_output
import lightning as L
from src.utils.nn.tools import log_losses_wandb_tracking
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from src.layers.inference_oc_tracks import (
    evaluate_efficiency_tracks,
    store_at_batch_end,
)
from src.models.gravnet_3_L_tracking import object_condensation_loss_tracking
from xformers.ops.fmha import BlockDiagonalMask
import os
import wandb

# from src.layers.obtain_statistics import (
#     obtain_statistics_graph_tracking,
#     create_stats_dict,
#     save_stat_dict,
#     plot_distributions,
# )
from src.utils.nn.tools import log_losses_wandb


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
    ):
        super().__init__()
        self.input_dim = 3
        self.output_dim = 4
        self.loss_final = 0
        self.number_b = 0
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []
        self.args = args
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
        # Load the energy correction module
        self.ec_model_wrapper_charged = NetWrapper("/afs/cern.ch/work/g/gkrzmanc/models/charged22000.pkl", dev)
        self.ec_model_wrapper_neutral = NetWrapper("/afs/cern.ch/work/g/gkrzmanc/models/neutral22000.pkl", dev)
        # freeze these models completely
        for param in self.ec_model_wrapper_charged.model.parameters():
            param.requires_grad = False
        for param in self.ec_model_wrapper_neutral.model.parameters():
            param.requires_grad = False
    def forward(self, g, y, step_count, eval="", return_train=False):
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor with shape (*batch_dimensions, num_points, 3)
            Point cloud input data

        Returns
        -------
        outputs : torch.Tensor with shape (*batch_dimensions, 1)
            Model prediction: a single scalar for the whole point cloud.
        """

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
        embedded_outputs, scalar_outputs = self.gatr(
            embedded_inputs, scalars=scalars, attention_mask=mask
        )  # (..., num_points, 1, 16)

        points = extract_point(embedded_outputs[:, 0, :])

        # Extract scalar and aggregate outputs from point cloud
        nodewise_outputs = extract_scalar(embedded_outputs)  # (..., num_points, 1, 1)

        x_point = points
        x_scalar = torch.cat(
            (nodewise_outputs.view(-1, 1), scalar_outputs.view(-1, 1)), dim=1
        )

        x_cluster_coord = self.clustering(x_point)
        beta = self.beta(x_scalar)
        if self.args.tracks:
            mask = g.ndata["hit_type"] == 1
            beta[mask] = 9
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
            graphs_new, true_new, sum_e = obtain_clustering_for_matched_showers(
                g,
                x,
                y,
                self.trainer.global_rank,
                use_gt_clusters=self.args.use_gt_clusters,
            )
            batch_num_nodes = graphs_new.batch_num_nodes()
            batch_idx = []
            for i, n in enumerate(batch_num_nodes):
                batch_idx.extend([i] * n)
            batch_idx = torch.tensor(batch_idx).to(self.device)
            graphs_new.ndata["h"][:, 0:3] = graphs_new.ndata["h"][:, 0:3] / 3300
            # TODO: add global features to each node here
            print("Using global features of the graphs as well")
            # graphs_num_nodes = graphs_new.batch_num_nodes
            # add num_nodes for each node
            graphs_sum_features = scatter_add(
                graphs_new.ndata["h"], batch_idx, dim=0
            )
            # now multiply graphs_sum_features so the shapes match
            graphs_sum_features = graphs_sum_features[batch_idx]
            # append the new features to "h" (graphs_sum_features)
            shape0 = graphs_new.ndata["h"].shape
            graphs_new.ndata["h"] = torch.cat(
                (graphs_new.ndata["h"], graphs_sum_features), dim=1
            )
            assert shape0[1] * 2 == graphs_new.ndata["h"].shape[1]
            # print("Also computing graph-level features")
            graphs_high_level_features = get_post_clustering_features(
                graphs_new, sum_e
            )
            node_features_avg = scatter_mean(
                graphs_new.ndata["h"], batch_idx, dim=0
            )
            node_features_avg = node_features_avg[:, 0:3]
            eta, phi = calculate_eta(node_features_avg[:, 0], node_features_avg[:, 1], node_features_avg[:, 2]), calculate_phi(node_features_avg[:, 0], node_features_avg[:, 1])
            graphs_high_level_features = torch.cat(
                (graphs_high_level_features, node_features_avg), dim=1
            )
            graphs_high_level_features = torch.cat(
                (graphs_high_level_features, eta.view(-1, 1)), dim=1
            )
            graphs_high_level_features = torch.cat(
                (graphs_high_level_features, phi.view(-1, 1)), dim=1
            )
            # print("Computed graph-level features")
            print("Shape", graphs_high_level_features.shape)
            # pred_energy_corr = self.GatedGCNNet(graphs_high_level_features)
            num_tracks = graphs_high_level_features[:, 7]
            charged_idx = torch.where(num_tracks >= 1)[0]
            neutral_idx = torch.where(num_tracks < 1)[0]
            # assert their union is the whole set
            assert len(charged_idx) + len(neutral_idx) == len(num_tracks)
            #assert (num_tracks > 1).sum() == 0
            if (num_tracks > 1).sum() > 0:
                print("! Particles with more than one track !")
                print((num_tracks > 1).sum().item(), "out of", len(num_tracks))
            assert (
                graphs_high_level_features.shape[0]
                == graphs_new.batch_num_nodes().shape[0]
            )
            charged_energies = self.ec_model_wrapper_charged.predict(graphs_high_level_features[charged_idx]).flatten()
            neutral_energies = self.ec_model_wrapper_neutral.predict(graphs_high_level_features[neutral_idx]).flatten()
            pred_energy_corr[charged_idx.flatten()] = charged_energies / sum_e.flatten()[charged_idx.flatten()]
            pred_energy_corr[neutral_idx.flatten()] = neutral_energies / sum_e.flatten()[neutral_idx.flatten()]
            #print("Pred energy corr:", pred_energy_corr)
            #print("Charged energy corr:", pred_energy_corr[charged_idx])
            #print("Neutral energy corr:", pred_energy_corr[neutral_idx])
            if return_train:
                return (
                    x,
                    pred_energy_corr,
                    true_new
                )
            else:
                return (
                    x,
                    pred_energy_corr,
                    true_new,
                    sum_e,
                    graphs_new,
                    batch_idx,
                    graphs_high_level_features,
                )
        else:
            pred_energy_corr = torch.ones_like(beta.view(-1, 1))
            return x, pred_energy_corr, 0
        return x, pred_energy_corr, 0

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
        if self.trainer.is_global_zero:
            model_output, e_cor, loss_ll = self(batch_g, y, batch_idx, return_train=True)
        else:
            model_output, e_cor, loss_ll = self(batch_g, y, 1, return_train=True)
            e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))

        # if self.global_step < 200:
        #     self.args.losstype = "hgcalimplementation"
        # else:
        #     self.args.losstype = "vrepweighted"
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
            loss_type=self.args.losstype,
        )
        loss = loss  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
        if self.trainer.is_global_zero:
            log_losses_wandb(True, batch_idx, 0, losses, loss, loss_ll)

        self.loss_final = loss.item() + self.loss_final
        self.number_b = self.number_b + 1
        del model_output
        del e_cor
        del losses
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
        # if self.global_step < 200:
        #     self.args.losstype = "hgcalimplementation"
        # else:
        #     self.args.losstype = "vrepweighted"
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
            loss_type=self.args.losstype,
        )
        loss_ec = 0

        print("starting validation step", batch_idx, loss)
        if self.trainer.is_global_zero:
            log_losses_wandb(
                True, batch_idx, 0, losses, loss, loss_ll, loss_ec, val=True
            )
        self.validation_step_outputs.append([model_output, e_cor, batch_g, y])
        if self.args.predict:
            model_output1 = torch.cat((model_output, e_cor.view(-1, 1)), dim=1)
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
                del model_output1
                del batch_g
        self.validation_step_outputs = []
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def configure_optimizers(self):
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
