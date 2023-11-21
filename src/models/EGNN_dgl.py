import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import dgl
from copy import deepcopy
from src.logger.plotting_tools import PlotCoordinates
import torch_cmspepr
from torch_scatter import scatter_max, scatter_add, scatter_mean
from src.layers.object_cond import calc_LV_Lbeta
from src.layers.obj_cond_inf import calc_energy_loss
from src.models.gravnet_model import global_exchange, obtain_batch_numbers


def assert_no_nans(x):
    """
    Raises AssertionError if there is a nan in the tensor
    """
    if torch.isnan(x).any():
        print(x)
    assert not torch.isnan(x).any()


class EGNN(nn.Module):
    def __init__(
        self,
        args,
        dev,
        activation: str = ("relu",),
        concat_global_exchange: bool = False,
        separate_heads: bool = False,
    ):
        """
        :param concat_global_exchange: Whether to concat "global" features to the node features.
        """
        super().__init__()
        in_node_nf = 8
        hidden_nf = 128
        out_node_nf = 4
        self.args = args
        self.output_dim = out_node_nf
        self.special_beta_core = False
        self.clust_space_norm = "none"
        in_edge_nf = 0
        device = dev
        act_fn = nn.SiLU()
        n_layers = 7
        residual = True
        attention = False
        normalize = False
        tanh = False
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.concat_global_exchange = concat_global_exchange
        self.beta_weight = 1.0  # set this to zero for no beta loss (when it's frozen)
        self.beta_exp_weight = 0.0  # set this to also optimize using beta zeros loss
        self.attr_rep_weight = 1.0
        if self.concat_global_exchange:
            add_global_exchange = 3 * (in_node_nf + 3)  # also add coords
        else:
            add_global_exchange = 0
        self.embedding_in = nn.Linear(in_node_nf + add_global_exchange, self.hidden_nf)
        # self.embedding_in_coords = nn.Linear(6, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf + 3, out_node_nf)
        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            self.layers.append(
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                )
            )
        self.embedding_in_beta = deepcopy(self.embedding_in)
        self.layers_beta = deepcopy(self.layers)

        self.separate_heads = separate_heads

        self.post_pid_pool_module = nn.Sequential(  # to project pooled "particle type" embeddings to a common space
            nn.Linear(22, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 22),
            nn.Softmax(dim=-1),
        )
        if self.separate_heads:
            self.beta_head = nn.Sequential(
                nn.Linear(self.hidden_nf + 3, 32),
                nn.SELU(),
                nn.Linear(32, 32),
                nn.SELU(),
                nn.Linear(32, 1),
            )
            self.coords_head = nn.Sequential(
                nn.Linear(self.hidden_nf + 3, 32),
                nn.SELU(),
                nn.Linear(32, 32),
                nn.SELU(),
                nn.Linear(32, 3),
            )

    def forward(self, g, step_count):
        batch = obtain_batch_numbers(g.ndata["h"], g)
        h = g.ndata["h"][:, 3:]
        # g.ndata["x"] = self.embedding_in_coords(g.ndata["c"])  # NBx2
        g.ndata["x"] = g.ndata["h"][:, 0:3] / 3330
        g.ndata["original_coords"] = g.ndata["h"][:, 0:3] / 3330
        if step_count % 5:
            PlotCoordinates(
                g, path="input_coords", outdir=self.args.model_prefix, egnn=True
            )
        if self.concat_global_exchange:
            h = global_exchange(g.ndata["h"], batch)
            h = h[:, 3:]
        h = self.embedding_in(h)  # NBx80
        g.ndata["hh"] = h
        # if self.special_beta_core:
        #     h_beta = self.embedding_in_beta
        #     g_beta = g
        #     for conv in self.layers_beta:
        #         g_beta = conv(g_beta)
        #         g_beta = update_knn(g_beta)
        #     h_beta = torch.cat((g_beta.ndata["hh"], g_beta.ndata["x"]), dim=1)
        g = update_knn(g)  # add edges to graph ad it does not have them by default
        for conv in self.layers:
            g = conv(g)
            g = update_knn(g)
            # the second step could be to do the knn again for each graph with the new coordinates
        h = torch.cat((g.ndata["hh"], g.ndata["x"]), dim=1)
        if self.separate_heads:
            # if self.special_beta_core:
            #     bj_raw = self.beta_head(h_beta)
            # else:
            bj_raw = self.beta_head(h)
            xj_raw = self.coords_head(h)
            g.ndata["final_cluster"] = xj_raw
            g.ndata["beta"] = bj_raw.view(-1)
            if step_count % 5:
                PlotCoordinates(
                    g,
                    path="final_clustering",
                    outdir=self.args.model_prefix,
                    predict=self.args.predict,
                    egnn=True,
                )
            h = torch.cat((xj_raw, bj_raw), dim=1)
        else:
            h = self.embedding_out(h)
        return h

    def create_separate_beta_core(self):
        self.special_beta_core = True
        self.embedding_in_beta.load_state_dict(self.embedding_in.state_dict())
        self.layers_beta.load_state_dict(self.layers.state_dict())
        for param in self.embedding_in_beta.parameters():
            param.requires_grad = True
        for layer in self.layers_beta:
            for param in layer.parameters():
                param.requires_grad = True
        print("Created separate beta head.")

    def freeze(self, what="core", unfreeze=False):
        # what = ["core", "beta", "coords"]
        assert what in ["core", "beta", "coords"]
        if what == "core":
            for layer in self.layers:
                for param in layer.parameters():
                    param.requires_grad = unfreeze
            for param in self.embedding_in.parameters():
                param.requires_grad = unfreeze
            for param in self.embedding_out.parameters():
                param.requires_grad = unfreeze
        elif what == "beta":
            for param in self.beta_head.parameters():
                param.requires_grad = unfreeze
        elif what == "coords":
            for param in self.coords_head.parameters():
                param.requires_grad = unfreeze

    def object_condensation_loss2(
        self,
        batch,
        pred,
        y,
        return_resolution=False,
        clust_loss_only=True,
        add_energy_loss=False,
        calc_e_frac_loss=False,
        q_min=0.1,
        frac_clustering_loss=0.1,
        attr_weight=1.0,
        repul_weight=1.0,
        fill_loss_weight=1.0,
        use_average_cc_pos=0.0,
        hgcalloss=False,
        e_frac_loss_radius=0.7,
        e_frac_loss_return_particles=False,
    ):
        """
        :param batch:
        :param pred:
        :param y:
        :param return_resolution: If True, it will only output resolution data to plot for regression (only used for evaluation...)
        :param clust_loss_only: If True, it will only add the clustering terms to the loss
        :return:
        """
        clust_loss_only = True
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
            hgcal_implementation=hgcalloss,
            hit_energies=batch.ndata["h"][:, 3],
        )
        if return_resolution:
            return a
        if clust_loss_only:
            print(
                "BETA WEIGHT",
                self.beta_weight,
                "BETA EXP WEIGHT",
                self.beta_exp_weight,
                "ATTR./REP. WEIGHT",
                self.attr_rep_weight,
            )
            loss = (
                self.attr_rep_weight * a[0]
                + self.beta_weight * a[1]
                + self.beta_exp_weight * a[15]
            )
            # loss = a[10]       #  ONLY INTERCLUSTERING LOSS - TEMPORARY!
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
                loss_e_frac, loss_e_frac_true = calc_energy_loss(
                    batch,
                    xj,
                    bj,
                    qmin=q_min,
                    radius=e_frac_loss_radius,
                    y=y,
                    e_frac_loss_return_particles=e_frac_loss_return_particles,
                    select_centers_by_particle=True,
                )
                if e_frac_loss_return_particles:
                    loss_e_frac_nopart, loss_e_frac_true_nopart = calc_energy_loss(
                        batch,
                        xj,
                        bj,
                        qmin=q_min,
                        radius=e_frac_loss_radius,
                        y=y,
                        e_frac_loss_return_particles=e_frac_loss_return_particles,
                        select_centers_by_particle=False,
                    )
                    return (
                        loss,
                        a,
                        loss_e_frac,
                        loss_e_frac_true,
                        loss_e_frac_nopart,
                        loss_e_frac_true_nopart,
                    )
                return loss, a, loss_e_frac, loss_e_frac_true
            else:
                return loss, a, 0, 0
        return loss, a, 0, 0


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.message = RelativePositionCordMessage(
            input_nf,
            output_nf,
            hidden_nf,
            residual=residual,
            attention=attention,
            normalize=normalize,
            coords_agg=coords_agg,
            tanh=tanh,
        )

        self.agg = Aggregationlayer(input_nf, hidden_nf, output_nf, residual)

    def forward(self, g, edge_attr=None, node_attr=None):

        g.update_all(self.message, self.agg)

        return g


class RelativePositionCordMessage(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        residual,
        attention,
        normalize,
        coords_agg,
        tanh,
    ):
        super(RelativePositionCordMessage, self).__init__()
        self.temp = 1
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        edges_in_d = 0
        act_fn = nn.SiLU()
        self.epsilon = 1e-8
        edge_coords_nf = 3

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + 1 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

    def forward(self, edges):
        coord_diff0 = edges.src["x"] - edges.dst["x"]
        # coord_diff1 = edges.src["x"][:, 3:] - edges.dst["x"][:, 3:]
        coord_diff = edges.src["x"] - edges.dst["x"]
        # coord_diff0 = torch.atan2(
        #     torch.sin(edges.src["x"][:, 1] - edges.dst["x"][:, 1]),
        #     torch.cos(edges.src["x"][:, 1] - edges.dst["x"][:, 1]),
        # )
        # radial1 = torch.sqrt(torch.sum(coord_diff1**2, 1))
        # radial0 = torch.sqrt(torch.sum(coord_diff0**2, 1)).unsqueeze(1)
        radial0 = torch.sum(coord_diff0**2, 1).unsqueeze(1)
        # radial = torch.cat((radial0.unsqueeze(1), radial1.unsqueeze(1)), dim=1)

        edge_feature = torch.cat(
            (radial0, edges.src["hh"], edges.dst["hh"]), dim=1
        )  # E x (2+80*2)
        edge_feature = self.edge_mlp(edge_feature)  # E x 80
        print("edge_feature", edge_feature)
        if self.normalize:
            norm = torch.sqrt(radial0).detach() + self.epsilon
            coord_diff = coord_diff / norm

        trans = coord_diff * self.coord_mlp(edge_feature)  # E x 2
        print("trans", trans)
        return {"radial": radial0, "trans": trans, "edge_feature": edge_feature}


class Aggregationlayer(nn.Module):
    def __init__(self, input_nf, hidden_nf, output_nf, residual):
        super(Aggregationlayer, self).__init__()
        act_fn = nn.SiLU()
        self.residual = residual
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

    def forward(self, nodes):
        # shape = nodes.mailbox['agg_feat'].shape # 1x7x80
        nodes.data["x"] = torch.clip(
            nodes.data["x"], min=-1e3, max=1e3
        )  # to mitigate weird random NaN errors...
        nodes.mailbox["trans"] = torch.clip(nodes.mailbox["trans"], min=-1e3, max=1e3)
        trans = torch.mean(nodes.mailbox["trans"], dim=1)
        coord = nodes.data["x"] + trans

        edge_feature = torch.sum(nodes.mailbox["edge_feature"], dim=1)

        agg = torch.cat((nodes.data["hh"], edge_feature), dim=1)
        h = self.node_mlp(agg)
        if self.residual:
            h = nodes.data["hh"] + h
        assert_no_nans(coord)
        assert_no_nans(h)
        return {"x": coord, "hh": h}


def update_knn(batch):
    graphs_eval = dgl.unbatch(batch)
    number_graphs = len(graphs_eval)
    graphs = []
    node_counter = 0
    for index in range(0, number_graphs):
        g = graphs_eval[index]
        # g_temp = dgl.knn_graph(g.ndata["x"], 11, exclude_self=True)
        edge_index = torch_cmspepr.knn_graph(g.ndata["x"], k=11)
        g_temp = dgl.graph(
            (edge_index[0], edge_index[1]), num_nodes=g.ndata["x"].shape[0]
        )
        g_temp = dgl.remove_self_loop(g_temp)
        g_temp.ndata["x"] = g.ndata["x"]
        g_temp.ndata["hh"] = g.ndata["hh"]
        graphs.append(g_temp)
    bg = dgl.batch(graphs)
    return bg
