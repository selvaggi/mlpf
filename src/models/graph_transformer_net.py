import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import sys
from functools import partial
import os.path as osp
import time
import numpy as np

"""
    Graph Transformer
    
"""
from src.layers.graph_transformer_layer import GraphTransformerLayer
from src.layers.mlp_readout_layer import MLPReadout
from src.layers.object_cond import (
    calc_LV_Lbeta,
    get_clustering,
    calc_LV_Lbeta_inference,
)
from src.layers.obj_cond_inf import calc_energy_loss


class GraphTransformerNet(nn.Module):
    def __init__(self, dev):
        super().__init__()

        in_dim_node = 9  # node_dim (feat is an integer)
        self.clust_space_norm = "none"
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
        self.device = dev
        self.lap_pos_enc = False
        self.wl_pos_enc = False
        max_wl_role_index = 100
        self.readout = "sum"
        self.output_dim = n_classes

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer
        self.embedding_h.weight.data.copy_(torch.eye(hidden_dim, in_dim_node))
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
        self.MLP_layer = MLPReadout(out_dim, 4)

        self.post_pid_pool_module = nn.Sequential(  # to project pooled "particle type" embeddings to a common space
            nn.Linear(22, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 22),
            nn.Softmax(dim=-1),
        )

    def forward(self, g_batch):
        g = g_batch

        ############################## Embeddings #############################################
        h = g.ndata["h"]
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)

        return self.MLP_layer(h)

    def object_condensation_loss2(
        self,
        batch,
        pred,
        y,
        return_resolution=False,
        clust_loss_only=False,
        add_energy_loss=False,
        calc_e_frac_loss=False,
        q_min=0.1,
        frac_clustering_loss=0.1,
        attr_weight=1.0,
        repul_weight=1.0,
        fill_loss_weight=1.0,
    ):
        """

        :param batch:
        :param pred:
        :param y:
        :param return_resolution: If True, it will only output resolution data to plot for regression (only used for evaluation...)
        :param clust_loss_only: If True, it will only add the clustering terms to the loss
        :return:
        """
        _, S = pred.shape
        if clust_loss_only:
            clust_space_dim = self.output_dim - 1
        else:
            clust_space_dim = self.output_dim - 28

        # xj = torch.nn.functional.normalize(
        #     pred[:, 0:clust_space_dim], dim=1
        # )  # 0, 1, 2: cluster space coords

        bj = torch.sigmoid(torch.reshape(pred[:, clust_space_dim], [-1, 1]))  # 3: betas

        xj = pred[:, 0:clust_space_dim]  # xj: cluster space coords
        if self.clust_space_norm == "twonorm":
            xj = torch.nn.functional.normalize(
                xj, dim=1
            )  # 0, 1, 2: cluster space coords
        elif self.clust_space_norm == "tanh":
            xj = torch.tanh(xj)
        elif self.clust_space_norm == "none":
            pass
        else:
            raise NotImplementedError
        if clust_loss_only:
            distance_threshold = torch.zeros((xj.shape[0], 3)).to(xj.device)
            energy_correction = torch.zeros_like(bj)
            momentum = torch.zeros_like(bj)
            pid_predicted = torch.zeros((distance_threshold.shape[0], 22)).to(
                momentum.device
            )
        else:
            distance_threshold = torch.reshape(
                pred[:, 1 + clust_space_dim : 4 + clust_space_dim], [-1, 3]
            )  # 4, 5, 6: distance thresholds
            energy_correction = torch.nn.functional.relu(
                torch.reshape(pred[:, 4 + clust_space_dim], [-1, 1])
            )  # 7: energy correction factor
            momentum = torch.nn.functional.relu(
                torch.reshape(pred[:, 27 + clust_space_dim], [-1, 1])
            )
            pid_predicted = pred[
                :, 5 + clust_space_dim : 27 + clust_space_dim
            ]  # 8:30: predicted particle one-hot encoding
        dev = batch.device
        clustering_index_l = batch.ndata["particle_number"]

        len_batch = len(batch.batch_num_nodes())
        batch_numbers = torch.repeat_interleave(
            torch.range(0, len_batch - 1).to(dev), batch.batch_num_nodes()
        ).to(dev)

        a = calc_LV_Lbeta(
            batch,
            y,
            distance_threshold,
            energy_correction,
            momentum=momentum,
            predicted_pid=pid_predicted,
            beta=bj.view(-1),
            cluster_space_coords=xj,  # Predicted by model
            cluster_index_per_event=clustering_index_l.view(
                -1
            ).long(),  # Truth hit->cluster index
            batch=batch_numbers.long(),
            qmin=q_min,
            return_regression_resolution=return_resolution,
            post_pid_pool_module=self.post_pid_pool_module,
            clust_space_dim=clust_space_dim,
            frac_combinations=frac_clustering_loss,
            attr_weight=attr_weight,
            repul_weight=repul_weight,
            fill_loss_weight=fill_loss_weight,
        )
        if return_resolution:
            return a
        if clust_loss_only:
            loss = a[0] + a[1]
            if calc_e_frac_loss:
                loss_E_frac, loss_E_frac_true = calc_energy_loss(
                    batch, xj, bj.view(-1), qmin=q_min
                )
            if add_energy_loss:
                loss += a[2]  # TODO add weight as argument

        else:
            loss = (
                a[0]
                + a[1]
                + 20 * a[2]
                + 0.001 * a[3]
                + 0.001 * a[4]
                + 0.001
                * a[
                    5
                ]  # TODO: the last term is the PID classification loss, explore this yet
            )  # L_V / batch_size, L_beta / batch_size, loss_E, loss_x, loss_particle_ids, loss_momentum, loss_mass)
        if clust_loss_only:
            if calc_e_frac_loss:
                return loss, a, loss_E_frac, loss_E_frac_true
            else:
                return loss, a, 0, 0
        return loss, a, 0, 0
