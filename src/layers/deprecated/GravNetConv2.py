from typing import Optional, Union
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor

import torch
from torch import Tensor
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
from dgl.nn import EdgeWeightNorm
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_cmspepr
from src.layers.GravNetConv3 import knn_per_graph


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {
            out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(
                -1, keepdim=True
            )
        }

    return func


def src_dot_distance(src_field, dst_field, out_field):
    def func(edges):
        dij = (edges.src[src_field] - edges.dst[dst_field]).pow(2).sum(-1, keepdim=True)
        edge_weight = torch.sqrt(dij + 1e-6)
        edge_weight = torch.exp(-torch.square(dij))
        return {out_field: edge_weight}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


def score_dij(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: edges.data["score"].view(-1) * edges.data["dij"].view(-1)}

    return func


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst("K_h", "Q_h", "score"))  # , edges)
        g.apply_edges(scaled_exp("score", np.sqrt(self.out_dim)))

        g.apply_edges(src_dot_distance("s_l", "s_l", "dij"))
        g.apply_edges(score_dij("news"))
        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e("V_h", "news", "V_h"), fn.sum("V_h", "wV"))
        g.send_and_recv(eids, fn.copy_e("score", "score"), fn.sum("score", "z"))

    def forward(self, g, h):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata["Q_h"] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["K_h"] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["V_h"] = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(g)
        g.ndata["z"] = g.ndata["z"].tile((1, 1, self.out_dim))
        mask_empty = g.ndata["z"] > 0
        head_out = g.ndata["wV"]
        head_out[mask_empty] = head_out[mask_empty] / (g.ndata["z"][mask_empty])
        g.ndata["z"] = g.ndata["z"][:, :, 0].view(
            g.ndata["wV"].shape[0], self.num_heads, 1
        )
        return head_out


class GraphTransformerLayer(nn.Module):
    """
    Param:
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        k,
        dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=False,
        use_bias=False,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.k = k
        space_dimensions = 3
        self.lin_s = Linear(self.in_channels, space_dimensions, bias=False)
        self.lin_h = Linear(self.in_channels, self.out_channels)
        self.lin = Linear(self.in_channels + self.out_channels, self.out_channels)
        self.attention = MultiHeadAttentionLayer(
            in_dim, out_dim // num_heads, num_heads, use_bias
        )

        self.O = nn.Linear(out_dim, out_dim)

        # if self.layer_norm:
        #     self.layer_norm1 = nn.LayerNorm(out_dim)

        # if self.batch_norm:
        #     self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # # FFN
        # self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        # self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        # if self.layer_norm:
        #     self.layer_norm2 = nn.LayerNorm(out_dim)

        # if self.batch_norm:
        #     self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h_l = self.lin_h(h)
        s_l = self.lin_s(h)
        graph = knn_per_graph(g, s_l, self.k)
        graph.ndata["s_l"] = s_l
        h_in1 = h_l  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(graph, h_l)
        h = attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O(h)

        h = self.lin(torch.cat((h_l, h), dim=1))
        # if self.residual:
        #     h = h_in1 + h  # residual connection

        # if self.layer_norm:
        #     h = self.layer_norm1(h)

        # if self.batch_norm:
        #     h = self.batch_norm1(h)

        # h_in2 = h  # for second residual connection

        # # FFN
        # h = self.FFN_layer1(h)
        # h = F.relu(h)
        # h = F.dropout(h, self.dropout, training=self.training)
        # h = self.FFN_layer2(h)

        # if self.residual:
        #     h = h_in2 + h  # residual connection

        # if self.layer_norm:
        #     h = self.layer_norm2(h)

        # if self.batch_norm:
        #     h = self.batch_norm2(h)

        return h, s_l
