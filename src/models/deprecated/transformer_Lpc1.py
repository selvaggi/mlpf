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
from src.layers.graph_transformer_layer_pc import GraphTransformerLayer
from src.layers.mlp_readout_layer import MLPReadout
import os
from dgl.geometry import farthest_point_sampler
import torch_cmspepr
import torch.nn.functional as F


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
        self.dev = dev
        self.loss_final = 0
        self.number_b = 0
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
        # node feat is an integer
        # self.embedding_h.weight.data.copy_(torch.eye(hidden_dim, in_dim_node))

        n_blocks = 3
        self.backbone = PointTransformer(
            in_dim_node,
            n_blocks,
            1,
            32,
            None,
            16,
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 2**n_blocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2**n_blocks),
        )

        self.ptb = GraphTransformerLayer(
            32 * 2**n_blocks,
            32 * 2**n_blocks,
            1,
            0.0,
        )

        self.n_blocks = n_blocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(n_blocks)):
            block_hidden_dim = 32 * 2**i
            self.transition_ups.append(
                TransitionUp(block_hidden_dim * 2, block_hidden_dim, block_hidden_dim)
            )
            self.transformers.append(
                GraphTransformerLayer(block_hidden_dim, block_hidden_dim, 1, 0.0)
            )
        self.MLP_layer = MLPReadout(32, 32)
        self.clustering = nn.Linear(32, 4 - 1, bias=False)
        self.beta = nn.Linear(32, 1)

    def forward(self, g, y, step_count, eval=""):
        original_coords = g.ndata["pos_hits_xyz"]
        graph_up, h, hidden_state = self.backbone(g)
        pos, h, g1 = hidden_state[-1]
        h = self.fc(h)
        graph_up.ndata["h"] = h
        h = self.ptb(graph_up, h)

        for i in range(self.n_blocks):
            h = self.transition_ups[i](
                pos,
                h,
                hidden_state[-i - 2][0],
                hidden_state[-i - 2][1],
                hidden_state[-i - 1][2],
            )
            pos = hidden_state[-i - 2][0]
            h = self.transformers[i](hidden_state[-i - 1][2], h)

        x = self.MLP_layer(h)
        x_cluster_coord = self.clustering(x)
        beta = self.beta(x)
        if self.args.tracks:
            mask = g.ndata["hit_type"] == 1
            beta[mask] = 9
        x_cluster_coord = x_cluster_coord + original_coords / 10000
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
        # if self.trainer.is_global_zero and self.current_epoch == 0:
        #     self.stat_dict = obtain_statistics_graph(self.stat_dict, y, batch_g)
        if self.trainer.is_global_zero:
            model_output, e_cor, loss_ll = self(batch_g, y, batch_idx)
        else:
            model_output, e_cor, loss_ll = self(batch_g, y, 1)
            e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
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
        print("end of validation step ")

    def on_train_epoch_end(self):

        self.log("train_loss_epoch", self.loss_final / self.number_b)

    def on_train_epoch_start(self):
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
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
            # else:
            #     model_output = self.validation_step_outputs[0][0]
            #     e_corr = self.validation_step_outputs[0][1]
            #     batch_g = self.validation_step_outputs[0][2]
            #     y = self.validation_step_outputs[0][3]
            #     model_output1 = torch.cat((model_output, e_corr.view(-1, 1)), dim=1)
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
            #         tracks=self.args.tracks,
            #     )
        self.validation_step_outputs = []

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


class PointTransformer(nn.Module):
    def __init__(
        self,
        feature_dim=3,
        n_blocks=4,
        downsampling_rate=2,
        hidden_dim=32,
        transformer_dim=None,
        n_neighbors=16,
    ):
        super(PointTransformer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ptb = GraphTransformerLayer(
            hidden_dim,
            hidden_dim,
            1,
            0.0,
        )
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(n_blocks):
            block_hidden_dim = hidden_dim * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(
                    [
                        block_hidden_dim // 2 + 3,
                        block_hidden_dim,
                        block_hidden_dim,
                    ],
                    n_neighbors=n_neighbors,
                )
            )
            self.transformers.append(
                GraphTransformerLayer(
                    block_hidden_dim,
                    block_hidden_dim,
                    1,
                    0.0,
                )
            )
        self.batchnorm1 = nn.BatchNorm1d(feature_dim, momentum=0.1)
        self.batchnorm1.running_var = torch.Tensor(
            [10000, 10000, 10000, 1, 1, 1, 1, 1]
        ).to(self.batchnorm1.running_mean.device)

    def forward(self, g):
        original_coords = g.ndata["pos_hits_xyz"]
        gu = knn_per_graph(g, original_coords, 7)
        gu.ndata["h"] = g.ndata["h"]
        gu.ndata["pos_hits_xyz"] = g.ndata["pos_hits_xyz"]
        h = gu.ndata["h"]
        h = self.batchnorm1(h)
        h = self.fc(h)
        h = self.ptb(gu, h)
        pos = gu.ndata["pos_hits_xyz"]
        gu.ndata["h"] = h
        hidden_state = [(pos, h, gu)]
        graph_up = gu
        for td, tf in zip(self.transition_downs, self.transformers):
            pos, h, graph_up, gi = td(graph_up)
            h = tf(graph_up, h)
            graph_up.ndata["h"] = h
            graph_up.ndata["pos_hits_xyz"] = pos
            hidden_state.append((pos, h, gi))

        return graph_up, h, hidden_state


class TransitionDown(nn.Module):
    """
    The Transition Down Module
    """

    def __init__(self, mlp_sizes, n_neighbors=64):
        super(TransitionDown, self).__init__()
        self.frnn_graph = KNNGraphBuilder(n_neighbors)
        self.message = RelativePositionMessage(n_neighbors)
        self.conv = KNNConv(mlp_sizes, 1)

    def forward(self, g):

        g, graph_up1 = self.frnn_graph(g)
        g.update_all(self.message, self.conv)
        # now the nodes with the up features have been upgraded
        mask = g.ndata["center"] == 1
        pos_res = g.ndata["pos_hits_xyz"][mask]
        feat_res = g.ndata["new_feat"][mask]
        # graph_up1 = knn_per_graph2(graph_up, pos_res, 7, feat_res)
        return pos_res, feat_res, graph_up1, g


# def knn_per_graph2(g, sl, k, feat_res):
#     graphs_list = dgl.unbatch(g)
#     node_counter = 0
#     new_graphs = []
#     for graph in graphs_list:
#         non = graph.number_of_nodes()
#         sls_graph = sl[node_counter : node_counter + non]
#         feat_res_g = feat_res[node_counter : node_counter + non]
#         # new_graph = dgl.knn_graph(sls_graph, k, exclude_self=True)
#         edge_index = torch_cmspepr.knn_graph(sls_graph, k=k)
#         new_graph = dgl.graph(
#             (edge_index[0], edge_index[1]), num_nodes=sls_graph.shape[0]
#         )
#         new_graph = dgl.remove_self_loop(new_graph)
#         # new_graph["pos_hits_xyz"] = sls_graph
#         # new_graph["h"] = feat_res_g
#         new_graphs.append(new_graph)

#         node_counter = node_counter + non
#     return dgl.batch(new_graphs)


class KNNGraphBuilder(nn.Module):
    """
    Build NN graph
    """

    def __init__(self, n_neighbor):
        super(KNNGraphBuilder, self).__init__()
        self.n_neighbor = n_neighbor

    def forward(self, g, feat=None):
        dev = g.ndata["pos_hits_xyz"].device
        glist = []
        graphs_up = []
        list_graphs = dgl.unbatch(g)
        for i, g_i in enumerate(list_graphs):
            n_points = int(np.floor(len(g_i.ndata["pos_hits_xyz"]) * 0.25).astype(int))
            centroids = farthest_point_sampler(
                g_i.ndata["pos_hits_xyz"].unsqueeze(0), n_points
            )
            N = len(g_i.ndata["pos_hits_xyz"])
            center = torch.zeros((N)).to(dev)
            center[centroids[0]] = 1
            s_l_i = g_i.ndata["pos_hits_xyz"]
            mask = center.bool()
            M_i = 7
            nodes = torch.range(start=0, end=N - 1, step=1).to(dev)
            nodes_up = nodes[mask]
            nodes_down = nodes[~mask]
            neigh_indices, neigh_dist_sq = torch_cmspepr.select_knn_directional(
                s_l_i[mask], s_l_i[~mask], M_i
            )
            j = nodes_down[neigh_indices]
            j = j.view(-1)
            i = torch.tile(nodes_up.view(-1, 1), (1, M_i)).reshape(-1)
            gi = dgl.graph((j.long(), i.long()), num_nodes=N).to(dev)
            gi.ndata["pos_hits_xyz"] = s_l_i
            gi.ndata["center"] = center * 1
            gi.ndata["feat"] = g_i.ndata["h"]
            glist.append(gi)
            sls_up = s_l_i[mask]
            edge_index = torch_cmspepr.knn_graph(sls_up, k=7)
            graph_up = dgl.graph(
                (edge_index[0], edge_index[1]), num_nodes=sls_up.shape[0]
            ).to(dev)
            # graph_up = dgl.graph(num_nodes=len(nodes_up)).to(dev)

            graphs_up.append(graph_up)
        bg = dgl.batch(glist)
        graphs_up = dgl.batch(graphs_up)
        return bg, graphs_up


class RelativePositionMessage(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self, n_neighbor):
        super(RelativePositionMessage, self).__init__()
        self.n_neighbor = n_neighbor

    def forward(self, edges):
        pos = edges.src["pos_hits_xyz"] - edges.dst["pos_hits_xyz"]
        res = torch.cat([pos, edges.src["feat"]], 1)

        return {"agg_feat": res}


class KNNConv(nn.Module):
    """
    Feature aggregation
    """

    def __init__(self, sizes, batch_size):
        super(KNNConv, self).__init__()
        self.batch_size = batch_size
        self.conv = nn.ModuleList()
        self.sizes = sizes
        self.bn = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.conv.append(nn.Linear(sizes[i - 1], sizes[i]))

    def forward(self, nodes):
        h = nodes.mailbox["agg_feat"]
        for conv in self.conv:
            h = conv(h)
            h = F.relu(h)
        h = torch.max(h, dim=1)[0]
        return {"new_feat": h}


class TransitionUp(nn.Module):
    """
    The Transition Up Module
    """

    def __init__(self, dim1, dim2, dim_out):
        super(TransitionUp, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            nn.BatchNorm1d(dim_out),  # TODO
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            nn.BatchNorm1d(dim_out),  # TODO
            nn.ReLU(),
        )
        self.fp = FeaturePropagation(-1, [])

    def forward(self, pos1, feat1, pos2, feat2, g):
        h1 = self.fc1(feat1)
        h2 = self.fc2(feat2)
        h1 = self.fp(pos2, pos1, None, h1, g)
        return h1 + h2


class FeaturePropagation(nn.Module):
    """
    The FeaturePropagation Layer
    """

    def __init__(self, input_dims, sizes):
        super(FeaturePropagation, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        sizes = [input_dims] + sizes
        for i in range(1, len(sizes)):
            self.convs.append(nn.Conv1d(sizes[i - 1], sizes[i], 1))
            self.bns.append(nn.BatchNorm1d(sizes[i]))
        self.NormDistanceFeature = NormDistanceFeature()
        self.Sum_dis = Sum_dis()

    def forward(self, x1, x2, feat1, feat2, g):
        """
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
            Input:
                x1: input points position data, [B, N, C]
                x2: sampled input points position data, [B, S, C]
                feat1: input points data, [B, N, D]
                feat2: input points data, [B, S, D]
            Return:
                new_feat: upsampled points data, [B, D', N]
        """
        # give each node 3 upper nodes of the same graph
        glist = []
        graphs_up = []
        g.ndata["x1"] = x1
        list_graphs = dgl.unbatch(g)
        dev = g.ndata["center"].device

        nup = 0
        for i, g_i in enumerate(list_graphs):
            up_nodes_mask = g_i.ndata["center"].bool()
            number_up_points = torch.sum(up_nodes_mask)
            x2_i = x2[nup : number_up_points + nup]
            nup = nup + number_up_points
            x1_i = g_i.ndata["x1"][~up_nodes_mask]
            neigh_indices, neigh_dist_sq = torch_cmspepr.select_knn_directional(
                x1_i, x2_i, 3
            )  # from up nodes to 3 down nodes
            N = len(g_i.ndata["center"])

            nodes = torch.range(start=0, end=N - 1, step=1).to(dev)
            nodes_up = nodes[up_nodes_mask]
            nodes_down = nodes[~up_nodes_mask]
            j = nodes_up[neigh_indices]
            j = j.view(-1)
            M_i = 3
            i = torch.tile(nodes_down.view(-1, 1), (1, M_i)).reshape(-1)
            g_up_to_down = dgl.graph((j.long(), i.long()), num_nodes=N)  # .to(dev)
            glist.append(g_up_to_down)
            # graph_up = dgl.graph(num_nodes=len(nodes_up)).to(dev)
        bg = dgl.batch(glist)
        # interpolate feature times weighted distance
        bg.ndata["interp"] = torch.zeros(len(x1), feat2.shape[1]).to(dev)
        up_points = g.ndata["center"].bool()
        bg.ndata["interp"][up_points] = feat2
        bg.ndata["pos"] = x1
        # bg.apply_edges(src_dot_dst("pos", "pos", "dis"))
        bg.update_all(self.NormDistanceFeature, self.Sum_dis)

        new_feat = bg.ndata["z"]
        for i, conv in enumerate(self.convs):
            bn = self.bns[i]
            new_feat = F.relu(bn(conv(new_feat)))
        return new_feat


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        distance = (edges.src[src_field] - edges.dst[dst_field]).pow(2).sum(-1)
        distance = 1 / (distance + 1e-6)
        return {out_field: distance}

    return func


class NormDistanceFeature(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self):
        super(NormDistanceFeature, self).__init__()

    def forward(self, edges):

        distance = (edges.src["pos"] - edges.dst["pos"]).pow(2).sum(-1)
        distance = 1 / (distance + 1e-6)
        interpolated_feature = edges.src["interp"]
        return {"interpolated_feature": interpolated_feature, "distance": distance}


class Sum_dis(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self):
        super(Sum_dis, self).__init__()

    def forward(self, nodes):
        norm = torch.sum(nodes.mailbox["distance"], dim=1)
        distance = nodes.mailbox["distance"] / torch.tile(norm.view(-1, 1), (1, 3))
        z = torch.sum(
            torch.mul(distance.unsqueeze(-1), nodes.mailbox["interpolated_feature"]),
            dim=1,
        )
        return {"z": z}
