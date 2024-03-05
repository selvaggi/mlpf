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
from src.layers.gcn_layer import GCNLayer


class GraphTransformerNet(nn.Module):
    def __init__(self, dev, activation='elu', in_dim_node = 9):
        super().__init__()
        # in_dim_node = 9  # node_dim (feat is an integer)
        self.clust_space_norm = "twonorm"
        hidden_dim = 80
        out_dim = 80
        n_classes = 4
        num_heads = 4
        in_feat_dropout = 0.0
        dropout = 0.0
        n_layers = 3
        self.n_layers = n_layers
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = dev
        self.lap_pos_enc = False
        self.wl_pos_enc = False
        max_wl_role_index = 100
        self.readout = "mean"
        self.output_dim = n_classes
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer
        self.embedding_h.weight.data.copy_(torch.eye(hidden_dim, in_dim_node))
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.batchnorm1 = nn.BatchNorm1d(in_dim_node)
        #self.batchnorm1 = nn.Identity()  #!!!!! #For experiments with and without batch norm
        #print("No batch norm!!")
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
        self.MLP_layer = MLPReadout(out_dim, 1)
        if activation=='elu':
            self.act = nn.ELU()
        else:
            print("Using identity activation for E corr. factor calibration!")
            self.act = nn.Identity()

    def forward(self, g_batch):
        g = g_batch
        ############################## Embeddings #############################################
        h = g.ndata["h"]
        # input embedding
        h = self.batchnorm1(h)
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)
        g.ndata["h1"] = h
        hg = dgl.mean_nodes(g, "h1")
        hg = self.MLP_layer(hg)
        hg = self.act(hg)
        return hg

class GCNNet(nn.Module):
    # A very simple GCN for debugging/testing
    def __init__(self, dev, activation='elu', in_dim_node = 9):
        super().__init__()

        hidden_dim = 30
        out_dim = 30
        in_feat_dropout = 0.0
        dropout = 0.0
        n_layers = 2
        self.readout = "mean"
        self.batch_norm = False
        self.residual = True

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.batchnorm1 = nn.BatchNorm1d(in_dim_node)
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)

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
        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem
        if activation == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.Identity()

    def forward(self, g):
        h = g.ndata["h"]
        h = self.batchnorm1(h)
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
        hg = self.MLP_layer(hg)
        hg = self.act(hg)
        return hg


class LinearGNNLayer(nn.Module):
    # A very simple linear transformation for debugging
    def __init__(self, dev, activation='elu', in_dim_node = 9):
        super().__init__()
        self.proj = nn.Linear(in_dim_node, 50)
        self.proj1 = nn.Linear(50, 50)
        self.proj2 = nn.Linear(50, 1)
        self.act = nn.LeakyReLU()

    def forward(self, g):
        hg = dgl.mean_nodes(g, "h") # a quick neural network to test the pipeline
        return self.act(self.proj2(self.act(self.proj1(self.act(self.proj(hg))))))


