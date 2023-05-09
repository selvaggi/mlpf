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
from src.layers.mlp_readout_layer import MLPReadout

"""
    Graph Transformer
    
"""
from src.layers.graph_transformer_edge_layer import GraphTransformerLayer


class GraphTransformerNet(nn.Module):
    def __init__(self, dev):
        super().__init__()

        in_dim_node = 4  # node_dim (feat is an integer)
        hidden_dim = 80
        out_dim = 80
        n_classes = 3
        num_heads = 8
        in_feat_dropout = 0.0
        dropout = 0.0
        n_layers = 2

        self.readout = "sum"
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = dev
        self.lap_pos_enc = False
        self.wl_pos_enc = False
        max_wl_role_index = 100

        if self.lap_pos_enc:
            pos_enc_dim = 10
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer

        self.embedding_e = nn.Linear(2, hidden_dim)  # node feat is an integer

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
        self.MLP_layer = MLPReadout(out_dim, 2)

    def forward(self, g):
        ####################################### Convert to graphs ##############################
        batched_graph = g
        g_base = batched_graph
        ############################## Embeddings #############################################
        h = g_base.ndata["h"]
        e = g_base.edata["h"]

        # input embedding
        h = self.embedding_h(h)
        e = self.embedding_e(e)
        # if self.lap_pos_enc:
        #    h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
        #    h = h + h_lap_pos_enc
        # if self.wl_pos_enc:
        #    h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
        #    h = h + h_wl_pos_enc
        # h = self.in_feat_dropout(h)

        # GraphTransformer Layers
        for conv in self.layers:
            h, e = conv(g_base, h, e)

        # output
        # g_base.ndata['h'] = h
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

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
