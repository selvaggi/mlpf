import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
from src.layers.GravNetConv3 import WeirdBatchNorm, knn_per_graph

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {
            out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(
                -1, keepdim=True
            )
        }

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


def src_dot_dst2(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] - edges.dst[dst_field])}

    return func


"""
    Single Attention Head
"""


class RelativePositionMessage(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self, out_dim):
        super(RelativePositionMessage, self).__init__()
        self.out_dim = out_dim

    def forward(self, edges):
        dist = -torch.sqrt((edges.src["G_h"] - edges.dst["G_h"]).pow(2).sum(-1) + 1e-6)
        distance = torch.exp((dist / np.sqrt(self.out_dim)).clamp(-5, 5))
        score = (edges.src["K_h"] * edges.dst["Q_h"]).sum(-1, keepdim=True)
        score_e = torch.exp((score / np.sqrt(self.out_dim)).clamp(-5, 5))
        print("checkling shapes", score_e.shape, distance.shape, edges.src["V_h"].shape)
        weight = torch.mul(score_e.view(-1, 1, 1), distance.view(-1, 1, 1))
        v_h = torch.mul(weight, edges.src["V_h"])

        return {"V1_h": v_h}


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_neigh, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.n_neigh = n_neigh
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.G = nn.Linear(in_dim, 3 * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.RelativePositionMessage = RelativePositionMessage(out_dim)
        # self.M1 = nn.Linear(1, out_dim, bias=False)
        # self.relu = nn.ReLU()
        # self.M2 = nn.Linear(out_dim, out_dim, bias=False)

    def propagate_attention(self, g):
        # Compute attention score
        # g.apply_edges(dist_calc("G_h", "G_h", "distance"))
        g.apply_edges(src_dot_dst("K_h", "Q_h", "score"))
        g.apply_edges(scaled_exp("score", np.sqrt(self.out_dim)))

        # g.apply_edges(scaled_exp("distance", np.sqrt(self.out_dim)))
        # g.apply_edges(score_times_dist("score_dis"))
        eids = g.edges()
        g.send_and_recv(eids, self.RelativePositionMessage, fn.sum("V1_h", "wV"))
        g.send_and_recv(eids, fn.copy_e("score", "score"), fn.sum("score", "z"))

    def forward(self, g, h):

        K_h = self.K(h)
        V_h = self.V(h)
        Q_h = self.Q(h)
        G_h = self.G(h)
        g.ndata["K_h"] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["Q_h"] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["G_h"] = G_h.view(-1, self.num_heads, 3)
        g.ndata["V_h"] = V_h.view(-1, self.num_heads, self.out_dim)
        s_l = g.ndata["G_h"]
        gu = knn_per_graph(g, s_l.view(-1, 3), self.n_neigh)
        gu.ndata["K_h"] = g.ndata["K_h"]
        gu.ndata["V_h"] = g.ndata["V_h"]
        gu.ndata["Q_h"] = g.ndata["Q_h"]
        gu.ndata["G_h"] = g.ndata["G_h"]
        self.propagate_attention(gu)
        # print(gu.ndata["z"].shape)
        gu.ndata["z"] = gu.ndata["z"].view(-1, 1, 1).tile((1, 1, self.out_dim))
        mask_empty = gu.ndata["z"] > 0
        head_out = gu.ndata["wV"]
        head_out[mask_empty] = head_out[mask_empty] / (gu.ndata["z"][mask_empty])
        gu.ndata["z"] = gu.ndata["z"][:, :, 0].view(
            gu.ndata["wV"].shape[0], self.num_heads, 1
        )

        return head_out


class GraphTransformerLayer(nn.Module):
    """
    Param:
    """

    def __init__(
        self,
        neigh,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=False,
        use_bias=False,
    ):
        super().__init__()
        self.d_shape = 32
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.neigh = neigh
        self.attention = MultiHeadAttentionLayer(
            self.neigh, self.d_shape, out_dim // num_heads, num_heads, use_bias
        )

        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.pre_gravnet = nn.Sequential(
            nn.Linear(self.in_channels, self.d_shape),  #! Dense 1
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 2
            nn.ELU(),
        )

    def forward(self, g, h):
        h_in1 = h  # for first residual connection
        h = self.pre_gravnet(h)
        # multi-head attention out
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.out_channels)
        # print("output of the attention ", h[0:2])
        # if torch.sum(torch.isnan(h)) > 0:
        #     print("output of the attention ALREADY NAN HERE")
        #     0 / 0
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O(h)

        if self.residual:
            h = h_in1 + h  # residual connection
        # print("output of residual ", h[0:2])
        # if torch.sum(torch.isnan(h)) > 0:
        #     print("output of the residual ALREADY NAN HERE")
        #     0 / 0
        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)
        # # print("output of batchnorm ", h[0:2])
        # if torch.sum(torch.isnan(h)) > 0:
        #     print("output of the batchnorm ALREADY NAN HERE")
        #     0 / 0
        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)
        # print("output of FFN_layer2 ", h[0:2])
        # if torch.sum(torch.isnan(h)) > 0:
        #     print("output of the FFN_layer2 ALREADY NAN HERE")
        #     0 / 0
        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)

        return h

    def __repr__(self):
        return "{}(in_channels={}, out_channels={}, heads={}, residual={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            self.residual,
        )

        # if torch.sum(torch.isnan(g.edata["vector"])) > 0:
        #     print("VECTOR ALREADY NAN HERE")
        #     0 / 0
        # e_data_m1 = self.M1(g.edata["vector"])
        # e_data_m1 = self.relu(e_data_m1)
        # e_data_m1 = self.M2(e_data_m1)
        # print("e_data_m1", e_data_m1[0:2])
        # g.edata["vector"] = e_data_m1
        # print("wV", g.ndata["wV"][0:2])
        # g.send_and_recv(eids, fn.copy_e("vector", "vector"), fn.sum("vector", "z"))
        # print("z", g.ndata["z"][0:2])
        # if torch.sum(torch.isnan(g.ndata["z"])) > 0:
        #     0 / 0


# class MultiHeadAttentionLayer2(nn.Module):
#     def __init__(self, n_neigh, in_dim, out_dim, num_heads, use_bias):
#         super().__init__()

#         self.out_dim = out_dim
#         self.num_heads = num_heads
#         self.n_neigh = n_neigh
#         if use_bias:
#             self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
#             self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
#             self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
#         else:
#             self.K = nn.Linear(in_dim, 3 * num_heads, bias=False)
#             self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
#             self.M1 = nn.Linear(3, out_dim, bias=False)
#             self.relu = nn.ReLU()
#             self.M2 = nn.Linear(out_dim, out_dim, bias=False)

#     def propagate_attention(self, g):
#         # Compute attention score
#         g.apply_edges(src_dot_dst2("K_h", "K_h", "vector"))  # , edges)
#         # if torch.sum(torch.isnan(g.edata["vector"])) > 0:
#         #     print("VECTOR ALREADY NAN HERE")
#         #     0 / 0
#         e_data_m1 = self.M1(g.edata["vector"])
#         e_data_m1 = self.relu(e_data_m1)
#         e_data_m1 = self.M2(e_data_m1)
#         g.edata["vector"] = e_data_m1
#         g.apply_edges(scaled_exp("vector", np.sqrt(self.out_dim)))
#         # if torch.sum(torch.isnan(g.edata["vector"])) > 0:
#         #     print(g.edata["vector"])
#         # Send weighted values to target nodes
#         eids = g.edges()
#         # vector attention to modulate individual channels
#         g.send_and_recv(eids, fn.u_mul_e("V_h", "vector", "V_h"), fn.sum("V_h", "wV"))
#         # print("wV", g.ndata["wV"][0:2])
#         g.send_and_recv(eids, fn.copy_e("vector", "vector"), fn.sum("vector", "z"))
#         # print("z", g.ndata["z"][0:2])
#         # if torch.sum(torch.isnan(g.ndata["z"])) > 0:
#         #     0 / 0

#     def forward(self, g, h):

#         K_h = self.K(h)
#         V_h = self.V(h)

#         g.ndata["K_h"] = K_h.view(-1, self.num_heads, 3)
#         g.ndata["V_h"] = V_h.view(-1, self.num_heads, self.out_dim)
#         # print("q_h", Q_h[0:2])
#         # print("K_h", K_h[0:2])
#         # print("V_h", V_h[0:2])
#         s_l = g.ndata["K_h"]
#         gu = knn_per_graph(g, s_l.view(-1, 3), self.n_neigh)
#         gu.ndata["K_h"] = g.ndata["K_h"]
#         gu.ndata["V_h"] = g.ndata["V_h"]
#         self.propagate_attention(gu)
#         # print(gu.ndata["z"].shape)
#         # gu.ndata["z"] = gu.ndata["z"].view(-1, 1, 1).tile((1, 1, self.out_dim))
#         mask_empty = gu.ndata["z"] > 0
#         head_out = gu.ndata["wV"]
#         # print(head_out.shape, gu.ndata["z"].shape)
#         head_out[mask_empty] = head_out[mask_empty] / (gu.ndata["z"][mask_empty])
#         # g.ndata["z"] = g.ndata["z"][:, :, 0].view(
#         #     g.ndata["wV"].shape[0], self.num_heads, 1
#         # )
#         # print("head_out", head_out[0:2])
#         # if torch.sum(torch.isnan(head_out)) > 0:
#         #     print("head_out ALREADY NAN HERE")
#         #     0 / 0
#         return head_out
