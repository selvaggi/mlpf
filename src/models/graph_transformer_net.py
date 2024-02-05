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
from src.layers.GravNetConv3 import knn_per_graph
from src.layers.obj_cond_inf import calc_energy_loss
from src.logger.plotting_tools import PlotCoordinates


class GraphTransformerNet(nn.Module):
    def __init__(
        self,
        args,
        dev,
        output_dim=4,
        input_dim=9,
        n_postgn_dense_blocks=0,
        clust_space_norm="none",
    ):
        super().__init__()

        in_dim_node = 9  # node_dim (feat is an integer)
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
        self.args = args
        self.batchnorm1 = nn.BatchNorm1d(in_dim_node, momentum=0.01)
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
        self.post_pid_pool_module = nn.Sequential(  # to project pooled "particle type" embeddings to a common space
            nn.Linear(22, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 22),
            nn.Softmax(dim=-1),
        )

    def forward(self, g_batch, step_count):
        g = g_batch
        original_coords = g.ndata["h"][:, 0:3]
        g.ndata["original_coords"] = original_coords
        if step_count % 100:
            PlotCoordinates(g, path="input_coords", outdir=self.args.model_prefix)
        ############################## Embeddings #############################################
        h = g.ndata["h"]
        # input embedding
        h = self.batchnorm1(h)
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # GraphTransformer Layers
        gu = knn_per_graph(g, original_coords, 8)
        gu.ndata["h"] = h
        for conv in self.layers:
            h = conv(gu, h)
        x = self.MLP_layer(h)
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
        pred_energy_corr = torch.ones_like(x[:, 0].view(-1, 1))
        return x, pred_energy_corr, torch.Tensor([0]).to(x.device)
