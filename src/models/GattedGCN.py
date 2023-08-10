import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np
from torch import Tensor
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_add

from src.layers.GravNetConv import GravNetConv

from typing import Tuple, Union, List
import dgl

from src.layers.object_cond import (
    calc_LV_Lbeta,
    get_clustering,
    calc_LV_Lbeta_inference,
)
from src.layers.obj_cond_inf import calc_energy_loss

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from src.layers.gated_gcn_layer import GatedGCNLayer
from src.layers.mlp_readout_layer import MLPReadout

from torch_scatter import scatter_max, scatter_add, scatter_mean
from src.layers.object_cond import calc_LV_Lbeta
from src.layers.object_cond_infonet import infonet_updated


class GatedGCNNet(nn.Module):
    def __init__(
        self,
        dev,
        input_dim: int = 9,
        output_dim: int = 4,
        hidden_dim: int = 80,
        n_layers: int = 10,
        **kwargs
    ):
        super().__init__()
        print(
            "GatedGCN with params:", input_dim, output_dim, hidden_dim, n_layers, kwargs
        )
        in_dim_node = input_dim  # node_dim (feat is an integer)
        in_dim_edge = 1  # edge_dim (feat is a float)
        n_classes = output_dim
        self.output_dim = n_classes
        dropout = 0.0
        # n_layers = 10
        self.clust_space_norm = "none"
        self.readout = "mean"
        self.batch_norm = True
        self.residual = True
        self.n_classes = n_classes
        self.device = dev
        self.pos_enc = False
        if self.pos_enc:
            pos_enc_dim = 2
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)  # edge feat is a float
        self.layers = nn.ModuleList(
            [
                GatedGCNLayer(
                    hidden_dim, hidden_dim, dropout, self.batch_norm, self.residual
                )
                for _ in range(n_layers)
            ]
        )
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)
        self.post_pid_pool_module = nn.Sequential(  # to project pooled "particle type" embeddings to a common space
            nn.Linear(22, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 22),
            nn.Softmax(dim=-1),
        )

    def forward(self, g):
        h = g.ndata["h"]
        e = g.edata["h"]
        # input embedding
        h = self.embedding_h(h)
        if self.pos_enc:
            h_pos_enc = g.ndata["pos_enc"]
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc
        e = self.embedding_e(e)

        # res gated convnets
        for conv in self.layers:
            h, e = conv(g, h, e)

        # output
        h_out = self.MLP_layer(h)
        h_out = torch.nn.functional.normalize(h_out, dim=1)

        return h_out

    def loss_nce(self, batch, pred, y, q_min=0.1):
        _, S = pred.shape

        clust_space_dim = self.output_dim - 1

        bj = torch.sigmoid(torch.reshape(pred[:, clust_space_dim], [-1, 1]))  # 3: betas

        relu = torch.nn.ReLU()  # xj: cluster space coords
        xj = torch.nn.functional.normalize(pred[:, 0:clust_space_dim])

        loss_total_, Loss_beta, Loss_beta2, loss_total = infonet_updated(
            batch, q_min, xj, bj
        )
        return loss_total_, Loss_beta, Loss_beta2, loss_total

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
        use_average_cc_pos=0.0,
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
        original_coords = batch.ndata["h"][:, 0:clust_space_dim]
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
            original_coords,
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
            use_average_cc_pos=use_average_cc_pos,
        )
        if return_resolution:
            return a
        if clust_loss_only:
            loss = a[0] + a[1]
            # loss = a[0] + 2. * a[1] #+ a[10] # temporarily add inter-clustering loss too
            # loss = a[10]  # ONLY INTERCLUSTERING LOSS - TEMPORARY!!!!
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
