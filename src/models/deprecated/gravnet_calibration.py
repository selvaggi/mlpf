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
    # calc_LV_Lbeta_inference,
)
from src.layers.obj_cond_inf import calc_energy_loss


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

        self.pred_energy = nn.Sequential(
            nn.Linear(64, 1, bias=False),
            nn.ELU(),
        )

        init_weights_ = True
        if init_weights_:
            # init_weights(self.clustering)
            init_weights(self.beta)
            init_weights(self.postgn_dense)
            init_weights(self.output)
            init_weights(self.pred_energy)

        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_2 = WeirdBatchNorm(64)
        else:
            self.ScaledGooeyBatchNorm2_2 = nn.BatchNorm1d(64, momentum=0.01)

    def forward(self, g, step_count):
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
        pred_energy_corr = self.pred_energy(x)
        pred_energy_corr = torch.ones_like(pred_energy_corr) + pred_energy_corr
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
        assert x.device == device

        if self.return_graphs:
            return x, graphs
        else:
            return x, pred_energy_corr.view(-1, 1)  # , loss_regularizing_neig, loss_ll


def object_condensation_loss2(
    batch,
    pred,
    pred_2,
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
    loss_type="hgcalimplementation",
    output_dim=4,
    clust_space_norm="none",
    dis=False,
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
    # print("bj", bj)
    original_coords = batch.ndata["h"][:, 0:clust_space_dim]
    if dis:
        distance_threshold = torch.reshape(pred[:, -1], [-1, 1])
    else:
        distance_threshold = 0
    energy_correction = pred_2
    xj = pred[:, 0:clust_space_dim]  # xj: cluster space coords
    if clust_space_norm == "twonorm":
        xj = torch.nn.functional.normalize(xj, dim=1)  # 0, 1, 2: cluster space coords
    elif clust_space_norm == "tanh":
        xj = torch.tanh(xj)
    elif clust_space_norm == "none":
        pass
    else:
        raise NotImplementedError
  
    dev = batch.device
    clustering_index_l = batch.ndata["particle_number"]

    len_batch = len(batch.batch_num_nodes())
    batch_numbers = torch.repeat_interleave(
        torch.arange(0, len_batch).to(dev), batch.batch_num_nodes()
    ).to(dev)

    a = calc_LV_Lbeta(
        original_coords,
        batch,
        y,
        distance_threshold,
        energy_correction,
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
        loss_type=loss_type,
        dis=dis,
    )

   
    loss = 1 * a[0] + a[1]  
      
    return loss, a


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
        torch.arange(0, len_batch).to(dev), batch.batch_num_nodes()
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


def calc_energy_correction_factor_loss(
    self,
    t_energy,
    t_dep_energies,
    pred_energy,
    pred_uncertainty_low,
    pred_uncertainty_high,
    return_concat=False,
):
    """
    This loss uses a Bayesian approach to predict an energy uncertainty.
    * t_energy              -> Truth energy of shower
    * t_dep_energies        -> Sum of deposited energy IF clustered perfectly
    * pred_energy           -> Correction factor applied to energy
    * pred_uncertainty_low  -> predicted uncertainty
    * pred_uncertainty_high -> predicted uncertainty (should be equal to ...low)
    """

    t_energy = tf.clip_by_value(t_energy, 0.0, 1e12)
    t_dep_energies = tf.clip_by_value(t_dep_energies, 0.0, 1e12)
    t_dep_energies = tf.where(
        t_dep_energies / t_energy > 2.0, 2.0 * t_energy, t_dep_energies
    )
    t_dep_energies = tf.where(
        t_dep_energies / t_energy < 0.5, 0.5 * t_energy, t_dep_energies
    )

    epred = pred_energy * t_dep_energies
    sigma = pred_uncertainty_high * t_dep_energies + 1.0

    # Uncertainty 'sigma' must minimize this term:
    # ln(2*pi*sigma^2) + (E_true - E-pred)^2/sigma^2
    matching_loss = (pred_uncertainty_low - pred_uncertainty_high) ** 2
    # prediction_loss = tf.math.divide_no_nan((t_energy - epred)**2, sigma**2)
    prediction_loss = tf.math.divide_no_nan((t_energy - epred), sigma)
    prediction_loss = huber(prediction_loss, d=2)

    uncertainty_loss = tf.math.log(sigma**2)

    matching_loss = tf.debugging.check_numerics(matching_loss, "matching_loss")
    prediction_loss = tf.debugging.check_numerics(prediction_loss, "matching_loss")
    uncertainty_loss = tf.debugging.check_numerics(uncertainty_loss, "matching_loss")
    prediction_loss = tf.clip_by_value(prediction_loss, 0, 10)
    uncertainty_loss = tf.clip_by_value(uncertainty_loss, 0, 10)

    if return_concat:
        return tf.concat([prediction_loss, matching_loss + uncertainty_loss], axis=-1)
    else:
        return prediction_loss, uncertainty_loss + matching_loss


def obtain_batch_numbers(x, g):
    dev = x.device
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes).to(dev))
        # num_nodes = gj.number_of_nodes()

    batch = torch.cat(batch_numbers, dim=0)
    return batch
