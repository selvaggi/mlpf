import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_add
from torch.nn import Parameter
from src.layers.GravNetConv3 import WeirdBatchNorm, knn_per_graph
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.layers.object_cond import (
    calc_LV_Lbeta,
    get_clustering,
    calc_LV_Lbeta_inference,
)
from typing import Optional, Union
from src.layers.obj_cond_inf import calc_energy_loss
from src.layers.inference_oc import create_and_store_graph_output
from src.models.gravnet_calibration import (
    object_condensation_loss2,
    obtain_batch_numbers,
)
from torch_geometric.nn.conv import MessagePassing
import lightning as L
from src.utils.nn.tools import log_losses_wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import Tensor
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
from dgl.nn import EdgeWeightNorm
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor
from src.layers.graph_transformer_layer import GraphTransformerLayer
from src.layers.mlp_readout_layer import MLPReadout


class GraphT(L.LightningModule):
    def __init__(
        self,
        args,
        dev,
        input_dim: int = 4,
        output_dim: int = 4,
        n_postgn_dense_blocks: int = 3,
        n_gravnet_blocks: int = 4,
        clust_space_norm: str = "twonorm",
        k_gravnet: int = 7,
        activation: str = "elu",
        weird_batchnom=False,
    ):

        super(GraphT, self).__init__()
        self.loss_final = 100
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []
        self.args = args
        self.validation_step_outputs = []
        in_dim_node = 8  # node_dim (feat is an integer)
        hidden_dim = 80  # before 80
        out_dim = 80
        n_classes = 4
        num_heads = 8
        in_feat_dropout = 0.0
        dropout = 0.0
        n_layers = 10
        self.n_layers = n_layers
        self.layer_norm = False
        self.batch_norm = False
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        self.lap_pos_enc = False
        self.wl_pos_enc = False
        max_wl_role_index = 100
        self.readout = "sum"
        self.output_dim = n_classes
        self.batchnorm1 = nn.BatchNorm1d(in_dim_node)
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer
        # self.embedding_h.weight.data.copy_(torch.eye(hidden_dim, in_dim_node))
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    self.layer_norm,
                    self.batch_norm,
                    self.residual,
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.layers.append(
            GraphTransformerLayer(
                hidden_dim,
                out_dim,
                num_heads,
                dropout,
                self.layer_norm,
                self.batch_norm,
                self.residual,
            )
        )
        self.MLP_layer = MLPReadout(out_dim, 64)
        self.clustering = nn.Linear(64, 4 - 1, bias=False)
        self.beta = nn.Linear(64, 1)

    def forward(self, g, step_count, eval=""):
        original_coords = g.ndata["pos_hits_xyz"]
        if self.trainer.is_global_zero and step_count % 100 == 0:
            g.ndata["original_coords"] = original_coords
            PlotCoordinates(
                g,
                path="input_coords",
                outdir=self.args.model_prefix,
                features_type="ones",
                predict=self.args.predict,
                epoch=str(self.current_epoch) + eval,
                step_count=step_count,
            )
        ############################## Embeddings #############################################
        # h = torch.cat((g.ndata["pos_hits_xyz"], g.ndata["hit_type"].view(-1, 1)), dim=1)
        h = g.ndata["h"]
        # input embedding
        h = self.batchnorm1(h)
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # GraphTransformer Layers
        gu = knn_per_graph(g, original_coords, 7)
        gu.ndata["h"] = h
        for conv in self.layers:
            h = conv(gu, h)
        x = self.MLP_layer(h)
        x_cluster_coord = self.clustering(x)
        beta = self.beta(x)
        if self.args.tracks:
            mask = g.ndata["hit_type"] == 1
            beta[mask] = 9
        g.ndata["final_cluster"] = x_cluster_coord
        g.ndata["beta"] = beta.view(-1)
        if self.trainer.is_global_zero and step_count % 100 == 0:
            PlotCoordinates(
                g,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
                epoch=str(self.current_epoch) + eval,
                step_count=step_count,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        pred_energy_corr = torch.ones_like(beta.view(-1, 1))
        return x, pred_energy_corr, 0

    # def on_after_backward(self):
    #     for name, p in self.named_parameters():
    #         if p.grad is None:
    #             print(name)

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
        loss = loss  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +

        if self.trainer.is_global_zero:
            log_losses_wandb(True, batch_idx, 0, losses, loss, loss_ll)

        self.loss_final = loss
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_step_outputs = []
        y = batch[1]

        batch_g = batch[0]

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
        loss = loss  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
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
                store=True,
                predict=True,
                tracks=self.args.tracks,
            )
            self.df_showers.append(df_batch)
            self.df_showers_pandora.append(df_batch_pandora)
            self.df_showes_db.append(df_batch1)

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
            # self.ScaledGooeyBatchNorm2_2.momentum = 0
            # for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
            #     # gravnet_block.batchnorm_gravnet1.momentum = 0
            #     gravnet_block.batchnorm_gravnet2.momentum = 0
            #     gravnet_block.batchnorm_gravnet3.momentum = 0

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            if self.args.predict:
                from src.layers.inference_oc import store_at_batch_end
                import pandas as pd

                self.df_showers = pd.concat(self.df_showers)
                self.df_showers_pandora = pd.concat(self.df_showers_pandora)
                self.df_showes_db = pd.concat(self.df_showes_db)
                store_at_batch_end(
                    path_save=self.args.model_prefix + "showers_df_evaluation",
                    df_batch=self.df_showers,
                    df_batch_pandora=self.df_showers_pandora,
                    df_batch1=self.df_showes_db,
                    step=0,
                    predict=True,
                )
            else:
                if len(self.validation_step_outputs) > 0:
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
                "scheduler": ReduceLROnPlateau(optimizer, patience=3),
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
        )

        self.post_gravnet = nn.Sequential(
            nn.Linear(out_channels, self.d_shape),
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 4
        )

        self.batchnorm_gravnet2 = nn.BatchNorm1d(self.d_shape)  # , momentum=0.01)
        self.batchnorm_gravnet3 = nn.BatchNorm1d(self.d_shape)  # , momentum=0.01)

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
        x_input = x
        xgn, gncoords = self.gravnet_layer(g, x, original_coords, batch)
        g.ndata["gncoords"] = gncoords
        x = xgn + x_input
        x = self.batchnorm_gravnet2(x)
        x_in2 = x
        x = self.post_gravnet(x)
        x = x_in2 + x
        x = self.batchnorm_gravnet3(x)  #! batchnorm 2
        return x


class GravNetConv(nn.Module):
    """
    Param: [in_dim, out_dim]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        space_dimensions: int,
        propagate_dimensions: int,
        k: int,
        num_workers: int = 1,
        weird_batchnom=False,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.num_workers = num_workers
        self.lin_s = Linear(in_channels, space_dimensions, bias=False)
        self.lin_h = Linear(in_channels, propagate_dimensions)
        self.lin = Linear(2 * propagate_dimensions, out_channels)

    def forward(self, g, x, original_coords, batch):
        h_l: Tensor = self.lin_h(x)  #! input_feature_transform
        s_l: Tensor = self.lin_s(x)
        graph = knn_per_graph(g, s_l, self.k)
        graph.ndata["s_l"] = s_l
        row = graph.edges()[0]
        col = graph.edges()[1]
        edge_index = torch.stack([row, col], dim=0)

        edge_weight = (s_l[edge_index[0]] - s_l[edge_index[1]]).pow(2).sum(-1)
        edge_weight = torch.sqrt(edge_weight + 1e-6)
        edge_weight = torch.exp(-torch.square(edge_weight))
        graph.edata["edge_weight"] = edge_weight.view(-1, 1)
        graph.ndata["h"] = h_l
        graph.update_all(self.message_func, self.reduce_func)
        out = graph.ndata["h"]

        out = self.lin(out)

        return (out, s_l)

    def message_func(self, edges):
        e_ij = edges.data["edge_weight"] * edges.src["h"]
        return {"e": e_ij}

    def reduce_func(self, nodes):
        mean_ = torch.mean(nodes.mailbox["e"], dim=-2)
        max_ = torch.max(nodes.mailbox["e"], dim=-2)[0]
        h = torch.cat((mean_, max_), dim=-1)
        return {"h": h}
