import torch
import torch.nn as nn
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


def scatter_count(input: torch.Tensor):
    """
    Returns ordered counts over an index array
    Example:
    >>> scatter_count(torch.Tensor([0, 0, 0, 1, 1, 2, 2])) # input
    >>> [3, 2, 2]
    Index assumptions work like in torch_scatter, so:
    >>> scatter_count(torch.Tensor([1, 1, 1, 2, 2, 4, 4]))
    >>> tensor([0, 3, 2, 0, 2])
    """
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


def obtain_batch_numbers(x, g):
    dev = x.device
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes).to(dev))
        #num_nodes = gj.number_of_nodes()

    batch = torch.cat(batch_numbers, dim=0)
    return batch


def global_exchange(x, batch):
    """
    Adds columns for the means, mins, and maxs per feature, per batch.
    Assumes x: (n_hits x n_features), batch: (n_hits),
    and that the batches are sorted!
    """
    batch = batch.to(torch.int64)
    n_hits_per_event = scatter_count(batch)
    n_hits, n_features = x.size()
    batch_size = int(batch.max()) + 1

    # minmeanmax: (batch_size x 3*n_features)
    meanminmax = torch.cat(
        (
            scatter_mean(x, batch, dim=0),
            scatter_min(x, batch, dim=0)[0],
            scatter_max(x, batch, dim=0)[0],
        ),
        dim=1,
    )
    assert list(meanminmax.size()) == [batch_size, 3 * n_features]

    meanminmax = torch.repeat_interleave(meanminmax, n_hits_per_event, dim=0)
    assert list(meanminmax.size()) == [n_hits, 3 * n_features]

    out = torch.cat((meanminmax, x), dim=1)
    assert out.size() == (n_hits, 4 * n_features)
    assert out.device == x.device
    return out


# FROM https://link.springer.com/content/pdf/10.1140/epjc/s10052-019-7113-9.pdf:

# GravNet model: The model consists of four blocks. Each
# block starts with concatenating the mean of the vertex
# features to the vertex features, three dense layers with
# 64 nodes and tanh activation, and one GravNet layer
# with S = 4 coordinate dimensions, FLR = 22 features to
# propagate, and FOUT = 48 output nodes per vertex. For
# each vertex, 40 neighbours are considered. The output
# of each block is passed as input to the next block and
# added to a list containing the output of all blocks. This
# determines the full vector of vertex features passed to a
# final dense layer with 128 nodes and ReLU activation

# In all cases, each output vertex of these model building blocks
# is fed through one dense layer with ReLU activation and three
# nodes, followed by a dense layer with two output nodes and
# softmax activation. This last processing step deter- mines the
# energy fraction belonging to each shower. Batch normalisation
# is applied in all models to the input and after each block.


class GravNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 96,
        space_dimensions: int = 4,
        propagate_dimensions: int = 22,
        k: int = 40,
        # batchnorm: bool = True
    ):
        super(GravNetBlock, self).__init__()
        # self.batchnorm = batchnorm
        # Includes all layers up to the global_exchange
        self.gravnet_layer = GravNetConv(
            in_channels, out_channels, space_dimensions, propagate_dimensions, k
        ).jittable()
        self.post_gravnet = nn.Sequential(
            # nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, 128),
            nn.Tanh(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, 96),
            nn.Tanh(),
        )
        self.output = nn.Sequential(
            nn.Linear(4 * 96, 96), nn.Tanh()  # , nn.BatchNorm1d(96)
        )

    def forward(self, g, x: Tensor, batch: Tensor) -> Tensor:
        x, graph = self.gravnet_layer(g, x, batch)
        x = self.post_gravnet(x)
        assert x.size(1) == 96
        x = global_exchange(x, batch)
        x = self.output(x)
        assert x.size(1) == 96
        return x, graph


class GravnetModel(nn.Module):
    def __init__(
        self,
        dev,
        input_dim: int = 9,
        output_dim: int = 31,
        n_postgn_dense_blocks: int = 4,
        n_gravnet_blocks: int = 4,
        clust_space_norm: str = "twonorm",
    ):
        # if not batchnorm:
        #    print("!!!! no batchnorm !!!")
        super(GravnetModel, self).__init__()
        # input_dim: int = 8
        # output_dim: int = 8 + 22  # 3x cluster positions, 1x beta, 3x position correction factor, 1x energy correction factor, 22x one-hot encoded particles (0th is the "OTHER" category)
        # n_gravnet_blocks: int = 4
        # n_postgn_dense_blocks: int = 4
        k = 7  # changed this so that the graphs are not FC
        self.return_graphs = False
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = n_gravnet_blocks
        self.n_postgn_dense_blocks = n_postgn_dense_blocks
        # self.batchnorm = batchnorm
        # if self.batchnorm:
        self.batchnorm1 = nn.BatchNorm1d(self.input_dim)
        # else:
        #    self.batchnorm1 = nn.Identity()
        self.input = nn.Linear(4 * input_dim, 64)
        assert clust_space_norm in ["twonorm", "tanh"]
        self.clust_space_norm = clust_space_norm
        # if isinstance(k, int):
        #     k = n_gravnet_blocks * [k]

        # assert len(k) == n_gravnet_blocks

        # Note: out_channels of the internal gravnet layer
        # not clearly specified in paper
        self.gravnet_blocks = nn.ModuleList(
            [
                GravNetBlock(64 if i == 0 else 96, k=k)
                for i in range(self.n_gravnet_blocks)
            ]
        )

        # Post-GravNet dense layers
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend(
                [
                    nn.Linear(4 * 96 if i == 0 else 128, 128),
                    nn.ReLU()  # ,
                    # nn.BatchNorm1d(128),
                ]
            )
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)

        # Output block
        self.output = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
        )

        self.post_pid_pool_module = nn.Sequential(  # to project pooled "particle type" embeddings to a common space
            nn.Linear(22, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 22),
            nn.Softmax(dim=-1),
        )

    def forward(self, g):
        x = g.ndata["h"]
        device = x.device
        batch = obtain_batch_numbers(x, g)
        # print('forward called on device', device)
        # x = self.batchnorm1(x)
        x = global_exchange(x, batch)
        x = self.input(x)
        assert x.device == device

        x_gravnet_per_block = []  # To store intermediate outputs
        graphs = []
        for gravnet_block in self.gravnet_blocks:
            x, graph = gravnet_block(g, x, batch)
            x_gravnet_per_block.append(x)
            graphs.append(graph)
        x = torch.cat(x_gravnet_per_block, dim=-1)
        assert x.size() == (x.size(0), 4 * 96)
        assert x.device == device

        x = self.postgn_dense(x)
        x = self.output(x)
        assert x.device == device
        if self.return_graphs:
            return x, graphs
        else:
            return x

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
        frac_clustering_loss=0.1
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
            frac_combinations=frac_clustering_loss
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


# class NoiseFilterModel(nn.Module):
#     def __init__(
#         self,
#         input_dim: int = 5,
#         output_dim: int = 2,
#     ):
#         super(NoiseFilterModel, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.network = nn.Sequential(
#             nn.BatchNorm1d(self.input_dim),
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 2),
#             nn.LogSoftmax(),
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         return self.network(x)


# class GravnetModelWithNoiseFilter(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(GravnetModelWithNoiseFilter, self).__init__()
#         self.signal_threshold = kwargs.pop("signal_threshold", 0.5)
#         self.gravnet = GravnetModel(*args, **kwargs)
#         self.noise_filter = NoiseFilterModel(input_dim=self.gravnet.input_dim)

#     def forward(self, x: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
#         out_noise_filter = self.noise_filter(x)
#         pass_noise_filter = torch.exp(out_noise_filter[:, 1]) > self.signal_threshold
#         # Get the GravNet model output on only hits that pass the noise threshold
#         out_gravnet = self.gravnet(x[pass_noise_filter], batch[pass_noise_filter])
#         return out_noise_filter, pass_noise_filter, out_gravnet
