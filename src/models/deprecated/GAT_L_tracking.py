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
from src.layers.obj_cond_inf import calc_energy_loss
from src.models.gravnet_calibration import (
    obtain_batch_numbers,
)
from src.layers.inference_oc import create_and_store_graph_output
import lightning as L
from src.utils.nn.tools import log_losses_wandb_tracking
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.layers.inference_oc_tracks import evaluate_efficiency_tracks
from src.layers.object_cond import calc_LV_Lbeta
from src.models.gatconv import GATConv

class GravnetModel(L.LightningModule):
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
    ):

        super(GravnetModel, self).__init__()
        self.emb_mlp = torch.nn.Linear(9, 128).to(dev)
        self.emb_out = torch.nn.Linear(128, 4).to(dev)
        self.mod = GATConv(in_channels=128, out_channels=128, heads=3, concat=False).to(dev)
        self.mod2 = GATConv(in_channels=128, out_channels=128, heads=3, concat=False).to(dev)
        self.mod3 = GATConv(in_channels=128, out_channels=128, heads=3, concat=False).to(dev)
        self.mod.input_dim = 9
        self.mod.output_dim = 4  # to be used by the loss model
        self.mod.clust_space_norm = "none"
        self.mod.post_pid_pool_module = torch.nn.Identity()

    def forward(self, g, step_count):
        x = g.ndata["h"]
        x = self.emb_mlp(x)
        edge_index = torch.stack(g.edges())
        x_prev = x
        for model in [self.mod, self.mod2, self.mod3]:
            x_new = model(x=x_prev, edge_index=edge_index) + x_prev
            x_prev = x_new
        return self.emb_out(x)
 

    # def on_after_backward(self):
    #     for name, p in self.named_parameters():
    #         if p.grad is None:
    #             print(name)

    def training_step(self, batch, batch_idx):
        y = batch[1]

        batch_g = batch[0]
        if self.trainer.is_global_zero:
            model_output = self(batch_g, batch_idx)
        else:
            model_output = self(batch_g, 1)

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
            hgcalloss=self.args.hgcalloss,
            tracking=True,
        )
        loss = loss

        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss)

        self.loss_final = loss
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs = []
        y = batch[1]

        batch_g = batch[0]

        model_output = self(batch_g, 1)
        preds = model_output.squeeze()

        (loss, losses) = object_condensation_loss_tracking(
            batch_g,
            model_output,
            y,
            q_min=self.args.qmin,
            frac_clustering_loss=0,
            clust_loss_only=self.args.clustering_loss_only,
            use_average_cc_pos=self.args.use_average_cc_pos,
            hgcalloss=self.args.hgcalloss,
            tracking=True,
        )
        loss = loss  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
        if self.trainer.is_global_zero:
            log_losses_wandb_tracking(True, batch_idx, 0, losses, loss, val=True)
        self.validation_step_outputs.append([model_output, batch_g, y])
        if self.trainer.is_global_zero:
            evaluate_efficiency_tracks(
                batch_g,
                model_output,
                y,
                0,
                batch_idx,
                0,
                path_save=self.args.model_prefix + "showers_df_evaluation",
                store=True,
                predict=True,
            )

    def on_train_epoch_end(self):

        # log epoch metric
        self.log("train_loss_epoch", self.loss_final)

    def on_train_epoch_start(self):
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 2 or self.args.predict:
            self.ScaledGooeyBatchNorm2_1.momentum = 0
            self.ScaledGooeyBatchNorm2_2.momentum = 0
            for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
                gravnet_block.batchnorm_gravnet1.momentum = 0
                gravnet_block.batchnorm_gravnet2.momentum = 0

    # def on_validation_epoch_end(self):
    #     if self.trainer.is_global_zero:

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "interval": "epoch",
                "monitor": "train_loss_epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


class GravNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 96,
        space_dimensions: int = 3,
        propagate_dimensions: int = 22,
        k: int = 40,
        # batchnorm: bool = True
        weird_batchnom=False,
    ):
        super(GravNetBlock, self).__init__()
        self.d_shape = 64
        out_channels = self.d_shape
        propagate_dimensions = self.d_shape
        self.gravnet_layer = GravNetConv(
            self.d_shape,
            out_channels,
            space_dimensions,
            propagate_dimensions,
            k,
            weird_batchnom,
        ).jittable()

        self.post_gravnet = nn.Sequential(
            nn.Linear(
                out_channels + space_dimensions + self.d_shape, self.d_shape
            ),  #! Dense 3
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 4
            nn.ELU(),
        )
        self.pre_gravnet = nn.Sequential(
            nn.Linear(in_channels, self.d_shape),  #! Dense 1
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 2
            nn.ELU(),
        )

    def forward(
        self,
        g,
        x: Tensor,
        batch: Tensor,
        original_coords: Tensor,
        step_count,
        outdir,
        num_layer,
    ) -> Tensor:
        x = self.pre_gravnet(x)
        x_input = x
        xgn, graph, gncoords, loss_regularizing_neig, ll_r = self.gravnet_layer(
            g, x, original_coords, batch
        )
        g.ndata["gncoords"] = gncoords
        # if step_count % 50:
        #     PlotCoordinates(
        #         g, path="gravnet_coord", outdir=outdir, num_layer=str(num_layer)
        #     )
        # gncoords = gncoords.detach()
        x = torch.cat((xgn, gncoords, x_input), dim=1)
        x = self.post_gravnet(x)
        return x, graph, loss_regularizing_neig, ll_r


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)


def object_condensation_loss_tracking(
    batch,
    pred,
    y,
    return_resolution=False,
    clust_loss_only=True,
    add_energy_loss=False,
    calc_e_frac_loss=False,
    q_min=0.1,
    frac_clustering_loss=0.1,
    attr_weight=1.0,
    repul_weight=1.0,
    fill_loss_weight=1.0,
    use_average_cc_pos=0.0,
    hgcalloss=False,
    output_dim=4,
    clust_space_norm="none",
    tracking=False,
):

    _, S = pred.shape
    if clust_loss_only:
        clust_space_dim = output_dim - 1
    else:
        clust_space_dim = output_dim - 28

    bj = torch.sigmoid(torch.reshape(pred[:, clust_space_dim], [-1, 1]))  # 3: betas
    original_coords = batch.ndata["h"][:, 0:clust_space_dim]
    xj = pred[:, 0:clust_space_dim]  # xj: cluster space coords

    dev = batch.device
    clustering_index_l = batch.ndata["particle_number"]

    len_batch = len(batch.batch_num_nodes())
    batch_numbers = torch.repeat_interleave(
        torch.range(0, len_batch - 1).to(dev), batch.batch_num_nodes()
    ).to(dev)

    a = calc_LV_Lbeta(
        original_coords,
        batch,
        y,
        None,
        None,
        momentum=None,
        predicted_pid=None,
        beta=bj.view(-1),
        cluster_space_coords=xj,  # Predicted by model
        cluster_index_per_event=clustering_index_l.view(
            -1
        ).long(),  # Truth hit->cluster index
        batch=batch_numbers.long(),
        qmin=q_min,
        return_regression_resolution=return_resolution,
        post_pid_pool_module=None,
        clust_space_dim=clust_space_dim,
        frac_combinations=frac_clustering_loss,
        attr_weight=attr_weight,
        repul_weight=repul_weight,
        fill_loss_weight=fill_loss_weight,
        use_average_cc_pos=use_average_cc_pos,
        hgcal_implementation=hgcalloss,
        tracking=tracking,
    )

    loss = a[0] + a[1]  # + 5 * a[14]

    return loss, a
