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
from src.layers.inference_oc import create_and_store_graph_output
from src.models.gravnet_calibration import (
    object_condensation_loss2,
    obtain_batch_numbers,
)
import lightning as L
from src.utils.nn.tools import log_losses_wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_cmspepr


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
    ):

        super(GravnetModel, self).__init__()
        self.loss_final = 100
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []
        self.args = args
        self.validation_step_outputs = []
        activation = "elu"
        acts = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
        }
        self.act = acts[activation]
        in_dim_node = 6
        num_heads = 8
        hidden_dim = 128
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True
        dropout = 0.05
        self.number_of_layers = 5
        self.num_classes = 13
        num_neigh = [16, 16, 16, 16, 16]
        n_layers = [2, 4, 9, 4, 4]
        # self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.embedding_h = nn.Sequential(
            nn.Linear(in_dim_node, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_1 = WeirdBatchNorm(self.input_dim)
        else:
            self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim)
        self.layers = nn.ModuleList(
            [
                Swin3D(
                    in_dim_node=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    layer_norm=self.layer_norm,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=dropout,
                    M=0.5,
                    k_in=num_neigh[ii],
                    n_layers=n_layers[ii],
                )
                for ii in range(self.number_of_layers)
            ]
        )
        # self.batch_norm1 = nn.BatchNorm1d(in_dim_node, momentum=0.01)
        hidden_dim = hidden_dim
        out_dim = hidden_dim * (self.number_of_layers + 1)
        self.n_postgn_dense_blocks = 3
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend(
                [
                    nn.Linear(out_dim if i == 0 else 64, 64),
                    self.act,  # ,
                ]
            )
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)
        self.ScaledGooeyBatchNorm2_2 = nn.BatchNorm1d(64, momentum=0.1)
        self.step_count = 0
        self.clustering = nn.Linear(64, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(64, 1)

    def forward(self, g, step_count):
        x = g.ndata["h"]
        original_coords = x[:, 0:3]
        g.ndata["original_coords"] = original_coords
        g.ndata["c"] = original_coords
        #! this is to have the GT for the loss
        g.ndata["object"] = object
        x = g.ndata["h"]
        c = g.ndata["c"]
        x = self.ScaledGooeyBatchNorm2_1(x)
        h = self.embedding_h(h)

        g1 = g

        if self.trainer.is_global_zero and (step_count % 100 == 0):
            PlotCoordinates(g, path="input_coords", outdir=self.args.model_prefix)
        full_res_features = []
        losses = 0
        depth_label = 0
        full_up_points = []
        ij_pairs = []
        latest_depth_rep = []
        for l, swin3 in enumerate(self.layers):
            features, up_points, g, i, j, s_l = swin3(g, h, c)
            if l == 0:
                full_res_features.append(features)
            c = s_l
            up_points = up_points.view(-1)
            if l == 0:
                g1.ndata["up_points"] = up_points + 1
            ij_pairs.append([i, j])
            full_up_points.append(up_points)
            h = features[up_points]
            c = c[up_points]
            depth_label = depth_label + 1
            # losses = losses + loss_ud
            features_down = features
            for it in range(0, depth_label):
                h_up_down = self.push_info_down(features_down, i, j)
                try:
                    latest_depth_rep[l - it] = h_up_down
                except:
                    latest_depth_rep.append(h_up_down)
                if depth_label > 1 and (l - it - 1) >= 0:
                    # print(l, it)
                    h_up_down_previous = latest_depth_rep[l - it - 1]
                    up_points_down = full_up_points[l - it - 1]
                    h_up_down_previous[up_points_down] = h_up_down
                    features_down = h_up_down_previous
                    i, j = ij_pairs[l - it - 1]
            full_res_features.append(h_up_down)

        all_resolutions = torch.concat(full_res_features, dim=1)
        x = self.postgn_dense(all_resolutions)
        x = self.ScaledGooeyBatchNorm2_2(x)
        x_cluster_coord = self.clustering(x)
        beta = self.beta(x)
        if self.args.tracks:
            mask = g.ndata["hit_type"] == 1
            beta[mask] = 9
        g1.ndata["final_cluster"] = x_cluster_coord
        g1.ndata["beta"] = beta.view(-1)
        if self.trainer.is_global_zero and (step_count % 100 == 0):
            PlotCoordinates(
                g1,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        pred_energy_corr = torch.ones_like(beta.view(-1, 1))

        return x, pred_energy_corr, 0

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
        loss = loss  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
        print("starting validation step", batch_idx, loss)
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
                    path_save=self.args.model_prefix + "showers_df_evaluation",
                    df_batch=self.df_showers,
                    df_batch_pandora=self.df_showers_pandora,
                    df_batch1=self.df_showes_db,
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
        self.validation_step_outputs = []

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


class Swin3D(nn.Module):
    """MAIN block
    1) Find coordinates and score for the graph
    2) Do knn down graph
    3) Message passing on the down graph SWIN3D_Blocks
    4) Downsample:
            - find up points
            - find neigh of from down to up
    """

    def __init__(
        self,
        in_dim_node,
        hidden_dim,
        num_heads,
        layer_norm,
        batch_norm,
        residual,
        dropout,
        M,
        k_in,
        n_layers,
    ):
        super().__init__()
        self.k = k_in
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual

        # self.send_scores = SendScoresMessage()
        # self.find_up = FindUpPoints()
        self.sigmoid_scores = nn.Sigmoid()
        self.funky_coordinate_space = True
        if self.funky_coordinate_space:
            self.embedding_coordinates = nn.Linear(
                in_dim_node, 3
            )  # node feat is an integer
        self.M = M  # number of points up to connect to
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.SWIN3D_Blocks = SWIN3D_Blocks(
            n_layers,
            hidden_dim,
            num_heads,
            dropout,
            self.layer_norm,
            self.batch_norm,
            self.residual,
            possible_empty=True,
        )
        self.Downsample = Downsample_maxpull(hidden_dim, M)

    def forward(self, g, h, c):
        object = g.ndata["object"]
        # 1) Find coordinates and score for the graph
        # embedding to calculate the coordinates in the embedding space #! this could also be kept to the original coordinates
        if self.funky_coordinate_space:
            s_l = self.embedding_coordinates(h)
        else:
            s_l = c
        h = self.embedding_h(h)
        scores = torch.rand(h.shape[0]).to(h.device)

        # 2) Do knn down graph
        g.ndata["s_l"] = s_l
        g = knn_per_graph(
            g, s_l, 7
        )  #! if these are learnt then they should be added to the gradients, they are not at the moment
        g.ndata["h"] = h

        # 3) Message passing on the down graph SWIN3D_Blocks
        h = self.SWIN3D_Blocks(g)

        g.ndata["scores"] = scores
        g.ndata["object"] = object
        g.ndata["s_l"] = s_l
        g.ndata["h"] = h

        # calculate loss of score
        # g.update_all(self.send_scores, self.find_up)
        # loss_ud = self.find_up.loss_ud

        # 4) Downsample:
        features, up_points, new_graphs_up, i, j = self.Downsample(g)
        return features, up_points, new_graphs_up, i, j, s_l


class SWIN3D_Blocks(nn.Module):
    """Point 3)
    Just multiple blocks of sparse attention over the down graph
    """

    def __init__(
        self,
        n_layers,
        hidden_dim,
        num_heads,
        layer_norm,
        batch_norm,
        residual,
        dropout,
        possible_empty=True,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.layers_message_passing = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    self.layer_norm,
                    self.batch_norm,
                    self.residual,
                    possible_empty=True,
                )
                for zz in range(n_layers)
            ]
        )

    def forward(self, g):
        h = g.ndata["h"]
        for ii, conv in enumerate(self.layers_message_passing):
            h = conv(g, h)

        return h


class Downsample_maxpull(nn.Module):
    """Point 4)
    - find up points
    - find neigh of from down to up
    """

    def __init__(self, hidden_dim, M):
        super().__init__()
        self.M = M
        self.embedding_features_to_att = nn.Linear(hidden_dim + 3, hidden_dim)
        self.MLP_difs = MLP_difs_maxpool(hidden_dim, hidden_dim)

    def forward(self, g):
        features = g.ndata["h"]
        list_graphs = dgl.unbatch(g)
        s_l = g.ndata["s_l"]
        graphs_UD = []
        graphs_U = []
        up_points = []
        for i in range(0, len(list_graphs)):
            graph_i = list_graphs[i]
            number_nodes_graph = graph_i.number_of_nodes()

            # find up nodes
            s_l_i = graph_i.ndata["s_l"]
            scores_i = graph_i.ndata["scores"].view(-1)
            device = scores_i.device
            number_up = np.floor(number_nodes_graph * 0.25).astype(int)
            up_points_i_index = torch.flip(torch.sort(scores_i, dim=0)[1], [0])[
                0:number_up
            ]
            up_points_i = torch.zeros_like(scores_i)
            up_points_i[up_points_i_index.long()] = 1
            up_points_i = up_points_i.bool()

            up_points.append(up_points_i)

            # connect down to up
            number_up_points_i = torch.sum(up_points_i)
            if number_up_points_i > 5:
                M_i = 5
            else:
                M_i = number_up_points_i
            nodes = torch.range(start=0, end=number_nodes_graph - 1, step=1).to(device)
            nodes_up = nodes[up_points_i]
            nodes_down = nodes[~up_points_i]

            neigh_indices, neigh_dist_sq = torch_cmspepr.select_knn_directional(
                s_l_i[~up_points_i], s_l_i[up_points_i], M_i
            )
            j = nodes_up[neigh_indices]
            j = j.view(-1)
            i = torch.tile(nodes_down.view(-1, 1), (1, M_i)).reshape(-1)

            g_i = dgl.graph((i.long(), j.long()), num_nodes=number_nodes_graph).to(
                device
            )
            g_i.ndata["h"] = graph_i.ndata["h"]
            g_i.ndata["s_l"] = graph_i.ndata["s_l"]
            g_i.ndata["object"] = graph_i.ndata["object"]
            # find index in original numbering
            graphs_UD.append(g_i)
            # use this way if no message passing between nodes
            # edge_index = torch_cmspepr.knn_graph(s_l_i[up_points_i], k=7)
            # graph_up = dgl.graph(
            #     (edge_index[0], edge_index[1]), num_nodes=len(nodes_up)
            # ).to(device)
            graph_up = dgl.DGLGraph().to(device)
            graph_up.add_nodes(len(nodes_up))
            graph_up.ndata["object"] = g_i.ndata["object"][up_points_i]
            graph_up.ndata["s_l"] = g_i.ndata["s_l"][up_points_i]
            graphs_U.append(graph_up)

        graphs_UD = dgl.batch(graphs_UD)
        i, j = graphs_UD.edges()
        graphs_U = dgl.batch(graphs_U)
        # naive way of giving the coordinates gradients
        features = torch.cat((features, s_l), dim=1)
        features = self.embedding_features_to_att(features)

        # do attention in g connected to up, this features have only been updated for points that have neighbourgs pointing to them: up-points
        features = self.MLP_difs(graphs_UD, features)

        up_points = torch.concat(up_points, dim=0).view(-1)

        return features, up_points, graphs_U, i, j


class MLP_difs_maxpool(nn.Module):
    """
    Param:
    """

    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.edgedistancespassing = EdgePassing(self.in_channels, self.out_channels)
        self.meanmaxaggregation = Max_aggregation(self.out_channels)
        # self.batch_norm = nn.BatchNorm1d(out_dim)
        # self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        # self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)
        # self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h_in = h
        g.ndata["features"] = h
        g.update_all(self.edgedistancespassing, self.meanmaxaggregation)
        h = g.ndata["h_updated"]
        # h = h_in + h
        # h = self.batch_norm(h)
        # h_in2 = h
        # h = self.FFN_layer1(h)
        # h = F.relu(h)
        # h = self.FFN_layer2(h)
        # h = h_in2 + h  # residual connection
        # h = self.batch_norm2(h)
        return h


class EdgePassing(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self, in_dim, out_dim):
        super(EdgePassing, self).__init__()
        # self.MLP = nn.Sequential(
        #     nn.Linear(in_dim, out_dim),  #! Dense 3
        #     nn.ReLU(),
        #     nn.Linear(out_dim, 1),  #! Dense 4
        #     nn.ReLU(),
        # )

    def forward(self, edges):
        dif = edges.src["features"]
        # att_weight = self.MLP(dif)
        # att_weight = torch.sigmoid(att_weight)  #! try sigmoid
        # feature = att_weight * edges.src["features"]
        return {"feature_n": dif}


class Max_aggregation(nn.Module):
    """
    Feature aggregation in a DGL graph
    """

    def __init__(self, out_dim):
        super(Max_aggregation, self).__init__()

    def forward(self, nodes):
        max_agg = torch.max(nodes.mailbox["feature_n"], dim=1)[0]

        return {"h_updated": max_agg}


def knn_per_graph(g, sl, k):
    """Build knn for each graph in the batch

    Args:
        g (_type_): original batch of dgl graphs
        sl (_type_): coordinates
        k (_type_): number of neighbours

    Returns:
        _type_: updates batch of dgl graphs with edges
    """
    graphs_list = dgl.unbatch(g)
    node_counter = 0
    new_graphs = []
    for graph in graphs_list:
        non = graph.number_of_nodes()
        sls_graph = sl[node_counter : node_counter + non]
        edge_index = torch_cmspepr.knn_graph(sls_graph, k=k)
        new_graph = dgl.graph(
            (edge_index[0], edge_index[1]), num_nodes=sls_graph.shape[0]
        )
        new_graphs.append(new_graph)
        node_counter = node_counter + non
    return dgl.batch(new_graphs)
