from os import path
import sys

# sys.path.append(
#     path.abspath("/afs/cern.ch/work/m/mgarciam/private/geometric-algebra-transformer/")
# )
# sys.path.append(path.abspath("/mnt/proj3/dd-23-91/cern/geometric-algebra-transformer/"))

from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import (
    embed_point,
    extract_scalar,
    extract_point,
    embed_scalar,
    embed_translation,
)
import torch
import torch.nn as nn
from src.logger.plotting_tools import PlotCoordinates
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.layers.obj_cond_inf import calc_energy_loss
from src.models.gravnet_calibration import (
    obtain_batch_numbers,
)
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
from src.layers.obtain_statistics import (
    obtain_statistics_graph,
    create_stats_dict,
    save_stat_dict,
    plot_distributions,
)


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
        self.args = args
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        )
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        self.clustering = nn.Linear(16, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(16, 1)
        self.vector_like_data = False

    def forward(self, g, y, step_count, eval=""):
        inputs = g.ndata["pos_hits_xyz"]

        if self.trainer.is_global_zero and step_count % 1000 == 0:
            g.ndata["original_coords"] = g.ndata["pos_hits_xyz"]
            PlotCoordinates(
                g,
                path="input_coords",
                outdir=self.args.model_prefix,
                features_type="ones",
                predict=self.args.predict,
                epoch=str(self.current_epoch) + eval,
                step_count=step_count,
            )
        inputs_scalar = g.ndata["hit_type"].view(-1, 1)
        inputs = self.ScaledGooeyBatchNorm2_1(inputs)
        # inputs = inputs.unsqueeze(0)
        if self.vector_like_data:
            velocities = g.ndata["vector"]
            velocities = embed_translation(velocities)
            embedded_inputs = (
                embed_point(inputs) + embed_scalar(inputs_scalar) + velocities
            )
        else:
            embedded_inputs = embed_point(inputs) + embed_scalar(inputs_scalar)
        embedded_inputs = embedded_inputs.unsqueeze(
            -2
        )  # (batch_size*num_points, 1, 16)
        mask = self.build_attention_mask(g)
        scalars = torch.zeros((inputs.shape[0], 1))
        # Pass data through GATr
        embedded_outputs, _ = self.gatr(
            embedded_inputs, scalars=scalars, attention_mask=mask
        )  # (..., num_points, 1, 16)
        output = embedded_outputs[:, 0, :]
        x_point = output
        x_scalar = output

        x_cluster_coord = self.clustering(x_point)
        beta = self.beta(x_scalar)
        g.ndata["final_cluster"] = x_cluster_coord
        g.ndata["beta"] = beta.view(-1)
        if self.trainer.is_global_zero and step_count % 1000 == 0:
            PlotCoordinates(
                g,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
                epoch=str(self.current_epoch) + eval,
                step_count=step_count,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        return x, embedded_outputs

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
        # if self.trainer.is_global_zero and self.current_epoch == 0:
        #     self.stat_dict = obtain_statistics_graph(
        #         self.stat_dict, y, batch_g, pf=False
        #     )
        if self.trainer.is_global_zero:
            model_output, embedded_outputs = self(batch_g, y, batch_idx)
        else:
            model_output, embedded_outputs = self(batch_g, y, 1)

        (loss, losses) = object_condensation_loss_tracking(
            batch_g,
            model_output,
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
            # loss_type=self.args.losstype,
            tracking=True,
        )
        loss = loss
        # print("training step", batch_idx, loss)
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss)

        self.loss_final = loss
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs = []
        y = batch[1]

        batch_g = batch[0]

        model_output, embedded_outputs = self(batch_g, y, batch_idx, eval="_val")
        preds = model_output.squeeze()

        (loss, losses) = object_condensation_loss_tracking(
            batch_g,
            model_output,
            y,
            q_min=self.args.qmin,
            frac_clustering_loss=0,
            clust_loss_only=self.args.clustering_loss_only,
            use_average_cc_pos=self.args.use_average_cc_pos,
            # loss_type=self.args.losstype,
            tracking=True,
        )
        loss = loss  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
        # print("validation step", batch_idx, loss)
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss, val=True)
        # self.validation_step_outputs.append([model_output, batch_g, y])
        if self.trainer.is_global_zero:
            df_batch = evaluate_efficiency_tracks(
                batch_g,
                model_output,
                embedded_outputs,
                y,
                0,
                batch_idx,
                0,
                path_save=self.args.model_prefix + "showers_df_evaluation",
                store=True,
                predict=False,
            )
            if self.args.predict:
                self.df_showers.append(df_batch)

    def on_train_epoch_end(self):
        # if self.current_epoch == 0 and self.trainer.is_global_zero:
        #     save_stat_dict(
        #         self.stat_dict,
        #         os.path.join(self.args.model_prefix, "showers_df_evaluation"),
        #     )
        #     plot_distributions(
        #         self.stat_dict,
        #         os.path.join(self.args.model_prefix, "showers_df_evaluation"),
        #         pf=True,
        #     )
        # self.stat_dict = {}
        # log epoch metric
        self.log("train_loss_epoch", self.loss_final)

    def on_train_epoch_start(self):
        # if self.current_epoch == 0 and self.trainer.is_global_zero:
        #     stats_dict = create_stats_dict(self.beta.weight.device)
        #     self.stat_dict = stats_dict
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 2 or self.args.predict:
            self.ScaledGooeyBatchNorm2_1.momentum = 0
            # self.ScaledGooeyBatchNorm2_2.momentum = 0
            # for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
            #     gravnet_block.batchnorm_gravnet1.momentum = 0
            #     gravnet_block.batchnorm_gravnet2.momentum = 0

    def on_validation_epoch_end(self):
        # print("VALIDATION END NEXT EPOCH", self.trainer.global_rank)
        print("end of val predictiong")
        if self.args.predict:
            store_at_batch_end(
                self.args.model_prefix + "showers_df_evaluation",
                self.df_showers,
                0,
                0,
                0,
                predict=True,
            )
        # if self.trainer.is_global_zero:

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.start_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": StepLR(
                    optimizer, step_size=4, gamma=0.1
                ),  # ReduceLROnPlateau(optimizer),
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
