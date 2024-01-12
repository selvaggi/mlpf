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
from src.models.gravnet_model import (
    scatter_count,
    obtain_batch_numbers,
    global_exchange,
)

from src.models.GattedGCN_correction import GraphTransformerNet, GCNNet
import torch_cmspepr


class GravnetModel(nn.Module):
    def __init__(
        self,
        args,
        dev,
        input_dim: int = 9,
        output_dim: int = 4,
        n_postgn_dense_blocks: int = 3,
        n_gravnet_blocks: int = 4,
        clust_space_norm: str = "twonorm",
        k_gravnet: int = 7,
        activation: str = "elu",
        weird_batchnom=False,
    ):

        super(GravnetModel, self).__init__()
        self.args = args
        assert activation in ["relu", "tanh", "sigmoid", "elu"]
        acts = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
        }
        self.act = acts[activation]

        N_NEIGHBOURS = [16, 128, 16, 256]
        TOTAL_ITERATIONS = len(N_NEIGHBOURS)
        self.return_graphs = False
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = TOTAL_ITERATIONS
        self.n_postgn_dense_blocks = n_postgn_dense_blocks
        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_1 = WeirdBatchNorm(self.input_dim)
        else:
            self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.01)

        self.Dense_1 = nn.Linear(input_dim, 64, bias=False)
        self.Dense_1.weight.data.copy_(torch.eye(64, input_dim))
        print("clust_space_norm", clust_space_norm)
        assert clust_space_norm in ["twonorm", "tanh", "none"]
        self.clust_space_norm = clust_space_norm

        self.d_shape = 32
        self.gravnet_blocks = nn.ModuleList(
            [
                GravNetBlock(
                    64 if i == 0 else (self.d_shape * i + 64),
                    k=N_NEIGHBOURS[i],
                    weird_batchnom=weird_batchnom,
                )
                for i in range(self.n_gravnet_blocks)
            ]
        )

        # Post-GravNet dense layers
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend(
                [
                    nn.Linear(4 * self.d_shape + 64 if i == 0 else 64, 64),
                    self.act,  # ,
                ]
            )
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)

        # Output block
        self.output = nn.Sequential(
            nn.Linear(64, 64),
            self.act,
            nn.Linear(64, 64),
            self.act,
            nn.Linear(64, 64),
        )

        self.post_pid_pool_module = nn.Sequential(  # to project pooled "particle type" embeddings to a common space
            nn.Linear(22, 64),
            self.act,
            nn.Linear(64, 64),
            self.act,
            nn.Linear(64, 22),
            nn.Softmax(dim=-1),
        )
        self.clustering = nn.Linear(64, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(64, 1)

        init_weights_ = True
        if init_weights_:
            # init_weights(self.clustering)
            init_weights(self.beta)
            init_weights(self.postgn_dense)
            init_weights(self.output)

        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_2 = WeirdBatchNorm(64)
        else:
            self.ScaledGooeyBatchNorm2_2 = nn.BatchNorm1d(64, momentum=0.01)

        self.GatedGCNNet = GraphTransformerNet(dev)

    def forward(self, g, step_count, y, local_rank):
        x = g.ndata["h"]
        original_coords = x[:, 0:3]
        g.ndata["original_coords"] = original_coords
        device = x.device
        batch = obtain_batch_numbers(x, g)
        x = self.ScaledGooeyBatchNorm2_1(x)
        x = self.Dense_1(x)
        assert x.device == device

        allfeat = []  # To store intermediate outputs
        allfeat.append(x)
        graphs = []
        loss_regularizing_neig = 0.0
        loss_ll = 0
        if step_count % 10:
            PlotCoordinates(g, path="input_coords", outdir=self.args.model_prefix)
        for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
            #! first time dim x is 64
            #! second time is 64+d
            x, graph, loss_regularizing_neig_block, loss_ll_ = gravnet_block(
                g,
                x,
                batch,
                original_coords,
                step_count,
                self.args.model_prefix,
                num_layer,
            )

            allfeat.append(x)
            graphs.append(graph)
            loss_regularizing_neig = (
                loss_regularizing_neig_block + loss_regularizing_neig
            )
            loss_ll = loss_ll_ + loss_ll
            if len(allfeat) > 1:
                x = torch.concatenate(allfeat, dim=1)

        x = torch.cat(allfeat, dim=-1)
        assert x.device == device

        x = self.postgn_dense(x)
        x = self.ScaledGooeyBatchNorm2_2(x)
        x_cluster_coord = self.clustering(x)
        beta = self.beta(x)
        g.ndata["final_cluster"] = x_cluster_coord
        g.ndata["beta"] = beta.view(-1)
        if step_count % 5:
            PlotCoordinates(
                g,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        graphs_new, true_new, sum_e = obtain_clustering_for_matched_showers(
            g, x, y, local_rank
        )
        graphs_new.ndata["h"][:, 0:3] = graphs_new.ndata["h"][:, 0:3] / 3300
        e_correction = self.GatedGCNNet(graphs_new)
        print("e_correction", e_correction)
        e_correction = torch.ones_like(e_correction) + e_correction
        return x, e_correction, true_new, sum_e  # , loss_regularizing_neig, loss_ll


def object_condensation_loss2(
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
    hgcalloss=False,
    output_dim=4,
    clust_space_norm="none",
    tracking=False,
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
        clust_space_dim = output_dim - 1
    else:
        clust_space_dim = output_dim - 28

    # xj = torch.nn.functional.normalize(
    #     pred[:, 0:clust_space_dim], dim=1
    # )  # 0, 1, 2: cluster space coords

    bj = torch.sigmoid(torch.reshape(pred[:, clust_space_dim], [-1, 1]))  # 3: betas
    original_coords = batch.ndata["h"][:, 0:clust_space_dim]
    xj = pred[:, 0:clust_space_dim]  # xj: cluster space coords
    if clust_space_norm == "twonorm":
        xj = torch.nn.functional.normalize(xj, dim=1)  # 0, 1, 2: cluster space coords
    elif clust_space_norm == "tanh":
        xj = torch.tanh(xj)
    elif clust_space_norm == "none":
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
    clustering_index_l = batch.ndata["particle_number"] # this starts at 1 because index 0 is for noise cluster 

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
        post_pid_pool_module=None,
        clust_space_dim=clust_space_dim,
        frac_combinations=frac_clustering_loss,
        attr_weight=attr_weight,
        repul_weight=repul_weight,
        fill_loss_weight=fill_loss_weight,
        use_average_cc_pos=use_average_cc_pos,
        hgcal_implementation=hgcalloss,
        tracking=tracking,
    )
    if return_resolution:
        return a
    if clust_loss_only:
        loss = a[0] + a[1]  # + 5 * a[14]
        # if calc_e_frac_loss:
        #     loss_E_frac, loss_E_frac_true = calc_energy_loss(
        #         batch, xj, bj.view(-1), qmin=q_min
        #     )
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
            return loss, a, 0, 0
        else:
            return loss, a, 0, 0
    return loss, a, 0, 0


def object_condensation_inference(self, batch, pred):
    """
    Similar to object_condensation_loss, but made for inference
    """
    _, S = pred.shape
    xj = torch.nn.functional.normalize(
        pred[:, 0:3], dim=1
    )  # 0, 1, 2: cluster space coords
    bj = torch.sigmoid(torch.reshape(pred[:, 3], [-1, 1]))  # 3: betas
    distance_threshold = torch.reshape(
        pred[:, 4:7], [-1, 3]
    )  # 4, 5, 6: distance thresholds
    energy_correction = torch.nn.functional.relu(
        torch.reshape(pred[:, 7], [-1, 1])
    )  # 7: energy correction factor
    momentum = torch.nn.functional.relu(
        torch.reshape(pred[:, 30], [-1, 1])
    )  # momentum magnitude
    pid_predicted = pred[:, 8:30]  # 8:30: predicted particle PID
    clustering_index = get_clustering(bj, xj)
    dev = batch.device
    len_batch = len(batch.batch_num_nodes())
    batch_numbers = torch.repeat_interleave(
        torch.range(0, len_batch - 1).to(dev), batch.batch_num_nodes()
    ).to(dev)

    pred = calc_LV_Lbeta_inference(
        batch,
        distance_threshold,
        energy_correction,
        momentum=momentum,
        predicted_pid=pid_predicted,
        beta=bj.view(-1),
        cluster_space_coords=xj,  # Predicted by model
        cluster_index_per_event=clustering_index.view(
            -1
        ).long(),  # Predicted hit->cluster index, determined by the clustering
        batch=batch_numbers.long(),
        qmin=0.1,
        post_pid_pool_module=self.post_pid_pool_module,
    )
    return pred


class GravNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 96,
        space_dimensions: int = 3,
        propagate_dimensions: int = 22,
        k: int = 40,
        # batchnorm: bool = True
        weird_batchnom=False,
    ):
        super(GravNetBlock, self).__init__()
        self.d_shape = 32
        out_channels = self.d_shape
        if weird_batchnom:
            self.batchnorm_gravnet1 = WeirdBatchNorm(self.d_shape)
        else:
            self.batchnorm_gravnet1 = nn.BatchNorm1d(self.d_shape, momentum=0.01)
        propagate_dimensions = self.d_shape
        self.gravnet_layer = GravNetConv(
            self.d_shape,
            out_channels,
            space_dimensions,
            propagate_dimensions,
            k,
            weird_batchnom,
        ).jittable()

        self.post_gravnet = nn.Sequential(
            nn.Linear(
                out_channels + space_dimensions + self.d_shape, self.d_shape
            ),  #! Dense 3
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 4
            nn.ELU(),
        )
        self.pre_gravnet = nn.Sequential(
            nn.Linear(in_channels, self.d_shape),  #! Dense 1
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 2
            nn.ELU(),
        )
        self.output = nn.Sequential(nn.Linear(self.d_shape, self.d_shape), nn.ELU())

        init_weights(self.output)
        init_weights(self.post_gravnet)
        init_weights(self.pre_gravnet)

        if weird_batchnom:
            self.batchnorm_gravnet2 = WeirdBatchNorm(self.d_shape)
        else:
            self.batchnorm_gravnet2 = nn.BatchNorm1d(self.d_shape, momentum=0.01)

    def forward(
        self,
        g,
        x: Tensor,
        batch: Tensor,
        original_coords: Tensor,
        step_count,
        outdir,
        num_layer,
    ) -> Tensor:
        x = self.pre_gravnet(x)
        x = self.batchnorm_gravnet1(x)
        x_input = x
        xgn, graph, gncoords, loss_regularizing_neig, ll_r = self.gravnet_layer(
            g, x, original_coords, batch
        )
        g.ndata["gncoords"] = gncoords
        if step_count % 50:
            PlotCoordinates(
                g, path="gravnet_coord", outdir=outdir, num_layer=str(num_layer)
            )
        # gncoords = gncoords.detach()
        x = torch.cat((xgn, gncoords, x_input), dim=1)
        x = self.post_gravnet(x)
        x = self.batchnorm_gravnet2(x)  #! batchnorm 2
        # x = global_exchange(x, batch)
        # x = self.output(x)
        return x, graph, loss_regularizing_neig, ll_r


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)


def object_condensation_loss_tracking(
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
    hgcalloss=False,
    output_dim=4,
    clust_space_norm="none",
    tracking=False,
):

    _, S = pred.shape
    if clust_loss_only:
        clust_space_dim = output_dim - 1
    else:
        clust_space_dim = output_dim - 28

    bj = torch.sigmoid(torch.reshape(pred[:, clust_space_dim], [-1, 1]))  # 3: betas
    original_coords = batch.ndata["h"][:, 0:clust_space_dim]
    xj = pred[:, 0:clust_space_dim]  # xj: cluster space coords

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
        None,
        None,
        momentum=None,
        predicted_pid=None,
        beta=bj.view(-1),
        cluster_space_coords=xj,  # Predicted by model
        cluster_index_per_event=clustering_index_l.view(
            -1
        ).long(),  # Truth hit->cluster index
        batch=batch_numbers.long(),
        qmin=q_min,
        return_regression_resolution=return_resolution,
        post_pid_pool_module=None,
        clust_space_dim=clust_space_dim,
        frac_combinations=frac_clustering_loss,
        attr_weight=attr_weight,
        repul_weight=repul_weight,
        fill_loss_weight=fill_loss_weight,
        use_average_cc_pos=use_average_cc_pos,
        hgcal_implementation=hgcalloss,
        tracking=tracking,
    )

    loss = a[0] + a[1]  # + 5 * a[14]

    return loss, a


from sklearn.cluster import DBSCAN
from src.layers.inference_oc import match_showers


def obtain_clustering_for_matched_showers(batch_g, model_output, y, local_rank):
    graphs_showers_matched = []
    true_energy_showers = []
    reco_energy_showers = []
    batch_g.ndata["coords"] = model_output[:, 0:3]
    batch_g.ndata["beta"] = model_output[:, 3]
    graphs = dgl.unbatch(batch_g)
    batch_id = y[:, -1].view(-1)
    for i in range(0, len(graphs)):
        mask = batch_id == i
        dic = {}
        dic["graph"] = graphs[i]
        dic["part_true"] = y[mask]
        betas = torch.sigmoid(dic["graph"].ndata["beta"])
        X = dic["graph"].ndata["coords"]
        clustering_mode = "dbscan"
        if clustering_mode == "clustering_normal":
            clustering = get_clustering(betas, X)
        elif clustering_mode == "dbscan":
            distance_scale = (
                (
                    torch.min(
                        torch.abs(torch.min(X, dim=0)[0] - torch.max(X, dim=0)[0])
                    )
                    / 30
                )
                .view(-1)
                .detach()
                .cpu()
                .numpy()[0]
            )

            db = DBSCAN(eps=distance_scale, min_samples=15).fit(X.detach().cpu())
            labels = db.labels_ + 1
            labels = np.reshape(labels, (-1))
            labels = torch.Tensor(labels).long().to(model_output.device)

            particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
            shower_p_unique = torch.unique(labels)
            shower_p_unique, row_ind, col_ind, i_m_w = match_showers(
                labels, dic, particle_ids, model_output, local_rank, i, None
            )
            row_ind = torch.Tensor(row_ind).to(model_output.device).long()
            col_ind = torch.Tensor(col_ind).to(model_output.device).long()
            index_matches = col_ind + 1
            index_matches = index_matches.to(model_output.device).long()
            for unique_showers_label in shower_p_unique:
                if torch.sum(unique_showers_label == index_matches) == 1:
                    index_in_matched = torch.argmax(
                        (unique_showers_label == index_matches) * 1
                    )
                    mask = labels == unique_showers_label
                    # non_graph = torch.sum(mask)
                    sls_graph = graphs[i].ndata["h"][mask][:, 0:3]
                    k = 7
                    edge_index = torch_cmspepr.knn_graph(sls_graph, k=k)
                    g = dgl.graph(
                        (edge_index[0], edge_index[1]), num_nodes=sls_graph.shape[0]
                    )
                    g = dgl.remove_self_loop(g)
                    # g = dgl.DGLGraph().to(graphs[i].device)
                    # g.add_nodes(non_graph.detach().cpu())
                    g.ndata["h"] = torch.cat(
                        (
                            graphs[i].ndata["h"][mask],
                            graphs[i].ndata["beta"][mask].view(-1, 1),
                        ),
                        dim=1,
                    )
                    energy_t = dic["part_true"][:, 3].to(model_output.device)
                    true_energy_shower = energy_t[row_ind[index_in_matched]]
                    reco_energy_shower = torch.sum(graphs[i].ndata["e_hits"][mask])
                    graphs_showers_matched.append(g)
                    true_energy_showers.append(true_energy_shower.view(-1))
                    reco_energy_showers.append(reco_energy_shower.view(-1))
    graphs_showers_matched = dgl.batch(graphs_showers_matched)
    true_energy_showers = torch.cat(true_energy_showers, dim=0)
    reco_energy_showers = torch.cat(reco_energy_showers, dim=0)
    return graphs_showers_matched, true_energy_showers, reco_energy_showers
