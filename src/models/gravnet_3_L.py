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
from src.models.gravnet_model import (
    scatter_count,
    obtain_batch_numbers,
    global_exchange,
)
from src.layers.inference_oc import create_and_store_graph_output
from src.models.gravnet_calibration import object_condensation_loss2
import lightning as L
from src.utils.nn.tools import log_losses_wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GravnetModel(L.LightningModule):
    def __init__(
        self,
        args,
        dev,
        input_dim: int = 9,
        output_dim: int = 4,
        n_postgn_dense_blocks: int = 3,
        n_gravnet_blocks: int = 4,
        clust_space_norm: str = "twonorm",
        k_gravnet: int = 7,
        activation: str = "elu",
        weird_batchnom=False,
    ):

        super(GravnetModel, self).__init__()
        self.loss_final = 100
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

        N_NEIGHBOURS = [16, 128, 16, 256]
        TOTAL_ITERATIONS = len(N_NEIGHBOURS)
        self.return_graphs = False
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = TOTAL_ITERATIONS
        self.n_postgn_dense_blocks = n_postgn_dense_blocks
        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_1 = WeirdBatchNorm(self.input_dim)
        else:
            self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.01)

        self.Dense_1 = nn.Linear(input_dim, 64, bias=False)
        self.Dense_1.weight.data.copy_(torch.eye(64, input_dim))
        print("clust_space_norm", clust_space_norm)
        assert clust_space_norm in ["twonorm", "tanh", "none"]
        self.clust_space_norm = clust_space_norm

        self.d_shape = 32
        self.gravnet_blocks = nn.ModuleList(
            [
                GravNetBlock(
                    64 if i == 0 else (self.d_shape * i + 64),
                    k=N_NEIGHBOURS[i],
                    weird_batchnom=weird_batchnom,
                )
                for i in range(self.n_gravnet_blocks)
            ]
        )

        # Post-GravNet dense layers
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend(
                [
                    nn.Linear(4 * self.d_shape + 64 if i == 0 else 64, 64),
                    self.act,  # ,
                ]
            )
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)

        # Output block
        # self.output = nn.Sequential(
        #     nn.Linear(64, 64),
        #     self.act,
        #     nn.Linear(64, 64),
        #     self.act,
        #     nn.Linear(64, 64),
        # )

        # self.post_pid_pool_module = nn.Sequential(  # to project pooled "particle type" embeddings to a common space
        #     nn.Linear(22, 64),
        #     self.act,
        #     nn.Linear(64, 64),
        #     self.act,
        #     nn.Linear(64, 22),
        #     nn.Softmax(dim=-1),
        # )
        self.clustering = nn.Linear(64, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(64, 1)

        init_weights_ = True
        if init_weights_:
            # init_weights(self.clustering)
            init_weights(self.beta)
            init_weights(self.postgn_dense)
            # init_weights(self.output)

        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_2 = WeirdBatchNorm(64)
        else:
            self.ScaledGooeyBatchNorm2_2 = nn.BatchNorm1d(64, momentum=0.01)

    def forward(self, g, step_count):
        x = g.ndata["h"]
        original_coords = x[:, 0:3]
        g.ndata["original_coords"] = original_coords
        device = x.device
        batch = obtain_batch_numbers(x, g)
        x = self.ScaledGooeyBatchNorm2_1(x)
        x = self.Dense_1(x)
        assert x.device == device

        allfeat = []  # To store intermediate outputs
        allfeat.append(x)
        graphs = []
        loss_regularizing_neig = 0.0
        loss_ll = 0
        if step_count % 100:
            PlotCoordinates(g, path="input_coords", outdir=self.args.model_prefix)
        for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
            #! first time dim x is 64
            #! second time is 64+d
            x, graph, loss_regularizing_neig_block, loss_ll_ = gravnet_block(
                g,
                x,
                batch,
                original_coords,
                step_count,
                self.args.model_prefix,
                num_layer,
            )

            allfeat.append(x)
            graphs.append(graph)
            loss_regularizing_neig = (
                loss_regularizing_neig_block + loss_regularizing_neig
            )
            loss_ll = loss_ll_ + loss_ll
            if len(allfeat) > 1:
                x = torch.concatenate(allfeat, dim=1)

        x = torch.cat(allfeat, dim=-1)
        assert x.device == device

        x = self.postgn_dense(x)
        x = self.ScaledGooeyBatchNorm2_2(x)
        x_cluster_coord = self.clustering(x)
        beta = self.beta(x)
        g.ndata["final_cluster"] = x_cluster_coord
        g.ndata["beta"] = beta.view(-1)
        if step_count % 100:
            PlotCoordinates(
                g,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        pred_energy_corr = torch.ones_like(beta.view(-1, 1))
        assert x.device == device

        return x, pred_energy_corr, loss_ll

    def on_after_backward(self):
        for name, p in self.named_parameters():
            if p.grad is None:
                print(name)

    def training_step(self, batch, batch_idx):
        y = batch[1]

        batch_g = batch[0]
        if self.trainer.is_global_zero:
            model_output, e_cor, loss_ll = self(batch_g, batch_idx)
        else:
            model_output, e_cor, loss_ll = self(batch_g, 1)

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
        loss = loss + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +

        if self.trainer.is_global_zero:
            log_losses_wandb(True, batch_idx, 0, losses, loss, loss_ll)
            self.log("train_loss", loss)
        self.loss_final = loss
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs = []
        y = batch[1]

        batch_g = batch[0]

        model_output, e_cor, loss_ll = self(batch_g, 1)
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
        loss = loss + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
        if self.trainer.is_global_zero:
            log_losses_wandb(True, batch_idx, 0, losses, loss, loss_ll, val=True)
        self.validation_step_outputs.append([model_output, e_cor, batch_g, y])
        if self.args.predict:
            model_output1 = torch.cat((model_output, e_cor.view(-1, 1)), dim=1)
            (df_batch, df_batch_pandora, df_batch1,) = create_and_store_graph_output(
                batch_g,
                model_output1,
                y,
                0,
                batch_idx,
                0,
                path_save=self.args.model_prefix + "showers_df_evaluation",
                store=False,
                predict=True,
                tracks=self.args.tracks,
            )
            self.df_showers.append(df_batch)
            self.df_showers_pandora.append(df_batch_pandora)
            self.df_showes_db.append(df_batch1)

    def on_train_epoch_end(self):
        # log epoch metric
        self.log("train_loss_epoch", self.loss_final)

    def on_validation_epoch_start(self):
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            if self.args.predict:
                from src.layers.inference_oc import store_at_batch_end
                import pandas as pd

                df_showers = pd.concat(df_showers)
                df_showers_pandora = pd.concat(df_showers_pandora)
                df_showes_db = pd.concat(df_showes_db)
                store_at_batch_end(
                    path_save=self.args.model_prefix + "showers_df_evaluation",
                    df_batch=df_showers,
                    df_batch_pandora=df_showers_pandora,
                    df_batch1=df_showes_db,
                    step=0,
                    predict=True,
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
                    path_save=self.args.model_prefix + "showers_df_evaluation",
                    store=True,
                    predict=False,
                    tracks=self.args.tracks,
                )

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
        self.d_shape = 32
        out_channels = self.d_shape
        if weird_batchnom:
            self.batchnorm_gravnet1 = WeirdBatchNorm(self.d_shape)
        else:
            self.batchnorm_gravnet1 = nn.BatchNorm1d(self.d_shape, momentum=0.01)
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
        # self.output = nn.Sequential(nn.Linear(self.d_shape, self.d_shape), nn.ELU())

        # init_weights(self.output)
        init_weights(self.post_gravnet)
        init_weights(self.pre_gravnet)

        if weird_batchnom:
            self.batchnorm_gravnet2 = WeirdBatchNorm(self.d_shape)
        else:
            self.batchnorm_gravnet2 = nn.BatchNorm1d(self.d_shape, momentum=0.01)

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
        x = self.batchnorm_gravnet1(x)
        x_input = x
        xgn, graph, gncoords, loss_regularizing_neig, ll_r = self.gravnet_layer(
            g, x, original_coords, batch
        )
        g.ndata["gncoords"] = gncoords
        if step_count % 50:
            PlotCoordinates(
                g, path="gravnet_coord", outdir=outdir, num_layer=str(num_layer)
            )
        # gncoords = gncoords.detach()
        x = torch.cat((xgn, gncoords, x_input), dim=1)
        x = self.post_gravnet(x)
        x = self.batchnorm_gravnet2(x)  #! batchnorm 2
        # x = global_exchange(x, batch)
        # x = self.output(x)
        return x, graph, loss_regularizing_neig, ll_r


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)
