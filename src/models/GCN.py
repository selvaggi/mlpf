import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as np
import dgl
import dgl.function as fn
import sys
from functools import partial
import os.path as osp
import time
import numpy as np

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from src.layers.gcn_layer import GCNLayer
from src.layers.mlp_readout_layer import MLPReadout


class GCNNet(nn.Module):
    def __init__(self, dev):
        super().__init__()
        num_atom_type = 6

        hidden_dim = 80
        out_dim = 80
        in_feat_dropout = 0.0
        dropout = 0.0
        n_layers = 10
        self.readout = "max"
        self.batch_norm = True
        self.residual = True

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.embedding_h = nn.Linear(num_atom_type, hidden_dim)

        self.layers = nn.ModuleList(
            [
                GCNLayer(
                    hidden_dim,
                    hidden_dim,
                    F.relu,
                    dropout,
                    self.batch_norm,
                    self.residual,
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.layers.append(
            GCNLayer(
                hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual
            )
        )
        self.MLP_layer = MLPReadout(out_dim, 3)  # 1 out dim since regression problem

    def forward(self, g):
        h = g.ndata["h"]
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for conv in self.layers:
            h = conv(g, h)
        g.ndata["h"] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, "h")
        elif self.readout == "max":
            hg = dgl.max_nodes(g, "h")
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, "h")
        else:
            hg = dgl.mean_nodes(g, "h")  # default readout is mean nodes

        return self.MLP_layer(hg)
