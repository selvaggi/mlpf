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
from src.layers.inference_oc import create_and_store_graph_output
from src.models.gravnet_calibration import (
    object_condensation_loss2,
    obtain_batch_numbers,
)
import lightning as L
from src.utils.nn.tools import log_losses_wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_cmspepr
import wandb
import torch.nn.functional as F
import dgl
import dgl.function as fn
import os


class GravnetModel(L.LightningModule):
    def __init__(
        self,
        args,
        dev,
        input_dim: int = 8,
        output_dim: int = 4,
        n_postgn_dense_blocks: int = 3,
        n_gravnet_blocks: int = 4,
        clust_space_norm: str = "twonorm",
        k_gravnet: int = 7,
        activation: str = "elu",
        weird_batchnom=False,
    ):

        super(GravnetModel, self).__init__()
        self.loss_final = 0
        self.number_b = 0
        self.input_dim = 8
        self.output_dim = output_dim
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []
        self.args = args
        self.validation_step_outputs = []
        activation = "elu"
        acts = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
        }
        self.act = acts[activation]
        in_dim_node = 8
        num_heads = 8
        hidden_dim = 32
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True
        dropout = 0.05
        self.number_of_layers = 3
        self.num_classes = 13
        num_neigh = [16, 16, 16, 16, 16]
        n_layers = [2, 4, 4]
        # self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.embedding_h = nn.Sequential(
            nn.Linear(in_dim_node, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_1 = WeirdBatchNorm(self.input_dim)
        else:
            self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim)
        self.layers = nn.ModuleList(
            [
                Swin3D(
                    in_dim_node=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    layer_norm=self.layer_norm,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=dropout,
                    M=0.5,
                    k_in=num_neigh[ii],
                    n_layers=n_layers[ii],
                )
                for ii in range(self.number_of_layers)
            ]
        )
        # self.batch_norm1 = nn.BatchNorm1d(in_dim_node, momentum=0.01)
        hidden_dim = hidden_dim
        out_dim = hidden_dim * (self.number_of_layers + 1)
        self.n_postgn_dense_blocks = 3
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend(
                [
                    nn.Linear(out_dim if i == 0 else 64, 64),
                    self.act,  # ,
                ]
            )
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)
        self.ScaledGooeyBatchNorm2_2 = nn.BatchNorm1d(64, momentum=0.1)
        self.step_count = 0
        self.clustering = nn.Linear(64, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(64, 1)

    def forward(self, g, y, step_count):
        x = g.ndata["h"]
        original_coords = x[:, 0:3]
        g.ndata["original_coords"] = original_coords
        g.ndata["c"] = original_coords
        #! this is to have the GT for the loss
        x = g.ndata["h"]
        c = g.ndata["c"]
        x = self.ScaledGooeyBatchNorm2_1(x)
        h = self.embedding_h(x)

        g1 = g

        if self.trainer.is_global_zero and (step_count % 100 == 0):
            PlotCoordinates(g, path="input_coords", outdir=self.args.model_prefix)
        full_res_features = []
        losses = 0
        depth_label = 0
        full_up_points = []
        ij_pairs = []
        latest_depth_rep = []
        for l, swin3 in enumerate(self.layers):
            features, up_points, g, i, j, s_l, loss_ud = swin3(g, h, c)
            if l == 0:
                full_res_features.append(features)
            c = s_l
            up_points = up_points.view(-1)
            if l == 0:
                g1.ndata["up_points"] = up_points + 1
            ij_pairs.append([i, j])
            full_up_points.append(up_points)
            h = features[up_points]
            c = c[up_points]
            depth_label = depth_label + 1
            losses = losses + loss_ud
            features_down = features
            for it in range(0, depth_label):
                h_up_down = self.push_info_down(features_down, i, j)
                try:
                    latest_depth_rep[l - it] = h_up_down
                except:
                    latest_depth_rep.append(h_up_down)
                if depth_label > 1 and (l - it - 1) >= 0:
                    # print(l, it)
                    h_up_down_previous = latest_depth_rep[l - it - 1]
                    up_points_down = full_up_points[l - it - 1]
                    h_up_down_previous[up_points_down] = h_up_down
                    features_down = h_up_down_previous
                    i, j = ij_pairs[l - it - 1]
            full_res_features.append(h_up_down)

        all_resolutions = torch.concat(full_res_features, dim=1)
        x = self.postgn_dense(all_resolutions)
        x = self.ScaledGooeyBatchNorm2_2(x)
        x_cluster_coord = self.clustering(x)
        beta = self.beta(x)
        if self.args.tracks:
            mask = g1.ndata["hit_type"] == 1
            beta[mask] = 9
        g1.ndata["final_cluster"] = x_cluster_coord
        g1.ndata["beta"] = beta.view(-1)
        if self.trainer.is_global_zero and (step_count % 100 == 0):
            PlotCoordinates(
                g1,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        pred_energy_corr = torch.ones_like(beta.view(-1, 1))

        return x, pred_energy_corr, losses

    def push_info_down(self, features, i, j):
        # feed information back down averaging the information of the upcoming uppoints
        g_connected_down = dgl.graph((j, i), num_nodes=features.shape[0])
        g_connected_down.ndata["features"] = features
        g_connected_down.update_all(
            fn.copy_u("features", "m"), fn.max("m", "h")
        )  #! full resolution graph
        h_up_down = g_connected_down.ndata["h"]
        # g connected down is the highest resolution graph with mean features of the up nodes
        return h_up_down

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]
        # if self.trainer.is_global_zero and self.current_epoch == 0:
        #     self.stat_dict = obtain_statistics_graph(self.stat_dict, y, batch_g)
        if self.trainer.is_global_zero:
            if self.args.correction:
                (
                    model_output,
                    e_cor1,
                    true_e,
                    sum_e,
                    new_graphs,
                    batch_idx,
                    graph_level_features,
                ) = self(batch_g, y, batch_idx)
                e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
                loss_ll = 0
            else:
                model_output, e_cor, loss_ll = self(batch_g, y, batch_idx)
        else:
            if self.args.correction:
                (
                    model_output,
                    e_cor1,
                    true_e,
                    sum_e,
                    new_graphs,
                    batch_idx,
                    graph_level_features,
                ) = self(batch_g, y, 1)
                e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
                loss_ll = 0
            else:
                model_output, e_cor, loss_ll = self(batch_g, y, 1)
                e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        if self.args.correction:
            print(model_output.shape, e_cor.shape, e_cor1.shape)
            e_cor1 += 1.0  # We regress the number around zero!!! # TODO: uncomment this if needed
        """energies_sums_features = new_graphs.ndata["h"][:, 15]
        energies_sums = [sum_e[i] for i in batch_idx]
        energies_sums = torch.tensor(energies_sums).to(energies_sums_features.device).flatten()
        print(energies_sums[energies_sums != energies_sums_features])
        print(energies_sums_features[energies_sums != energies_sums_features])
        assert (torch.abs(energies_sums - energies_sums_features) < 0.001).all()"""
        # print(model_output.shape, e_cor.shape, e_cor1.shape)
        # e_cor1 += 1.  # We regress the number around zero!!! # TODO: uncomment this if needed
        (loss, losses, loss_E, loss_E_frac_true,) = object_condensation_loss2(
            batch_g,
            model_output,
            e_cor,
            y,
            clust_loss_only=True,
            add_energy_loss=False,
            calc_e_frac_loss=False,
            q_min=self.args.qmin,
            frac_clustering_loss=self.args.frac_cluster_loss,
            attr_weight=self.args.L_attractive_weight,
            repul_weight=self.args.L_repulsive_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            hgcalloss=self.args.hgcalloss,
        )
        if self.args.correction:
            debug_regress_e_sum = False

            def corr(array):
                return torch.clamp(array, min=0.000001)

            if debug_regress_e_sum:
                # This section is just for debugging!
                loss, loss_abs, loss_abs_nocali = loss_reco_sum_absolute(
                    e_cor1, torch.log10(true_e), torch.log10(sum_e)
                )
                data = [[x, y] for (x, y) in zip(torch.log10(sum_e), e_cor1)]
                table = wandb.Table(
                    data=data, columns=["sum E (GT)", "sum E (regressed)"]
                )
                wandb.log(
                    {
                        "energy_corr": wandb.plot.scatter(
                            table,
                            "sum E (GT)",
                            "sum E (regressed)",
                            title="energy correction correlation",
                        )
                    }
                )
                print("debug_regress_e_sum == True !")
                loss, loss_abs, loss_abs_nocali = loss_reco_sum_absolute(
                    torch.log10(corr(e_cor1)),
                    torch.log10(corr(true_e)),
                    torch.log10(corr(sum_e)),
                )

                data = [
                    [x, y]
                    for (x, y) in zip(
                        torch.log10(corr(sum_e)), torch.log10(corr(e_cor1))
                    )
                ]
                table = wandb.Table(
                    data=data, columns=["log10 sum E (GT)", "log10 sum E (regressed)"]
                )
                wandb.log(
                    {
                        "energy_corr": wandb.plot.scatter(
                            table,
                            "log10 sum E (GT)",
                            "log10 sum E (regressed)",
                            title="energy correction correlation",
                        )
                    }
                )
                print("Logged!")
            else:
                loss, loss_abs, loss_abs_nocali = loss_reco_true(e_cor1, true_e, sum_e)
                true_e_corr = (true_e / sum_e) - 1
                model_e_corr = e_cor1
                data = [[x, y] for (x, y) in zip(true_e_corr, model_e_corr)]

                table = wandb.Table(
                    data=data, columns=["true E corr factor", "regressed E corr factor"]
                )
                wandb.log(
                    {
                        "energy_corr": wandb.plot.scatter(
                            table,
                            "true E corr factor",
                            "regressed E corr factor",
                            title="energy correction correlation",
                        )
                    }
                )
                table = wandb.Table(
                    data=data, columns=["true E corr factor", "regressed E corr factor"]
                )
                wandb.log(
                    {
                        "energy_corr": wandb.plot.scatter(
                            table,
                            "true E corr factor",
                            "regressed E corr factor",
                            title="energy correction correlation",
                        )
                    }
                )
                # if self.args.graph_level_features:
                #     # also plot graph level features correlations
                #     data = [[x, y] for (x, y) in zip(true_e_corr, sum_e)]
                #     table = wandb.Table(
                #         data=data, columns=["true E corr factor", "sum of E of hits"]
                #     )
                #     # wandb.log({"energy_corr_vs_E_hits": wandb.plot.scatter(table, "true E corr factor", "sum of E of hits",
                #     #                                                       title="energy correction vs. sum of E hits")})
                #     #  save graph-level features temporarily, to view later...
                #     cluster_features_path = os.path.join(
                #         self.args.model_prefix, "cluster_features"
                #     )
                #     if not os.path.exists(cluster_features_path):
                #         os.makedirs(cluster_features_path)
                #     save_features(
                #         cluster_features_path,
                #         {
                #             "x": graph_level_features.detach().cpu(),
                #             "e_true": true_e.detach().cpu(),
                #             "e_reco": model_e_corr.detach().cpu(),
                #             "true_e_corr": true_e_corr.detach().cpu(),
                #         },
                #     )
                #     print("!!!temporarily saving features in an external file!!!!")
                # print("Logged!")
        else:
            loss1 = (
                loss + loss_ll
            )  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
        if self.trainer.is_global_zero:
            log_losses_wandb(True, batch_idx, 0, losses, loss, loss_ll)

        self.loss_final = loss + self.loss_final
        self.number_b = self.number_b + 1
        return loss1

    def validation_step(self, batch, batch_idx):
        cluster_features_path = os.path.join(self.args.model_prefix, "cluster_features")
        show_df_eval_path = os.path.join(
            self.args.model_prefix, "showers_df_evaluation"
        )
        if not os.path.exists(show_df_eval_path):
            os.makedirs(show_df_eval_path)
        if not os.path.exists(cluster_features_path):
            os.makedirs(cluster_features_path)
        self.validation_step_outputs = []
        y = batch[1]
        batch_g = batch[0]
        if self.args.correction:
            (
                model_output,
                e_cor1,
                true_e,
                sum_e,
                new_graphs,
                batch_id,
                graph_level_features,
            ) = self(batch_g, y, 1)
            loss_ll = 0
            e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        else:
            model_output, e_cor1, loss_ll = self(batch_g, y, 1)
            loss_ll = 0
            e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        preds = model_output.squeeze()
        (loss, losses, loss_E, loss_E_frac_true,) = object_condensation_loss2(
            batch_g,
            model_output,
            e_cor,
            y,
            clust_loss_only=True,
            add_energy_loss=False,
            calc_e_frac_loss=False,
            q_min=self.args.qmin,
            frac_clustering_loss=self.args.frac_cluster_loss,
            attr_weight=self.args.L_attractive_weight,
            repul_weight=self.args.L_repulsive_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            hgcalloss=self.args.hgcalloss,
        )
        loss_ec = 0
        if self.args.correction:
            loss, loss_abs, loss_abs_nocali = loss_reco_true(e_cor1, true_e, sum_e)
            loss_ec = 0
        # else:
        #     print("Doing both correction and OC loss!!")
        #     loss_ec, loss_abs, loss_abs_nocali = loss_reco_true(e_cor1, true_e, sum_e)
        #     loss = loss + 0.05 * loss_ec
        print("starting validation step", batch_idx, loss)
        if self.trainer.is_global_zero:
            log_losses_wandb(
                True, batch_idx, 0, losses, loss, loss_ll, loss_ec, val=True
            )
        self.validation_step_outputs.append([model_output, e_cor, batch_g, y])
        if self.args.predict:
            model_output1 = torch.cat((model_output, e_cor.view(-1, 1)), dim=1)
            if self.args.correction:
                e_corr = e_cor1
            else:
                e_corr = None
            (df_batch_pandora, df_batch1, df_batch) = create_and_store_graph_output(
                batch_g,
                model_output1,
                y,
                0,
                batch_idx,
                0,
                path_save=show_df_eval_path,
                store=True,
                predict=True,
                e_corr=e_corr,
                tracks=self.args.tracks,
            )
            self.df_showers.append(df_batch)
            self.df_showers_pandora.append(df_batch_pandora)
            self.df_showes_db.append(df_batch1)

    def on_train_epoch_end(self):
        # if self.current_epoch == 0 and self.trainer.is_global_zero:
        #     save_stat_dict(
        #         self.stat_dict,
        #         os.path.join(self.args.model_prefix, "showers_df_evaluation"),
        #     )
        #     plot_distributions(
        #         self.stat_dict,
        #         os.path.join(self.args.model_prefix, "showers_df_evaluation"),
        #         pf=True,
        #     )
        #     self.stat_dict = {}
        self.log("train_loss_epoch", self.loss_final / self.number_b)

    def on_train_epoch_start(self):
        # if self.current_epoch == 0 and self.trainer.is_global_zero:
        #     stats_dict = create_stats_dict(self.beta.weight.device)
        #     self.stat_dict = stats_dict
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            self.ScaledGooeyBatchNorm2_1.momentum = 0
            # self.ScaledGooeyBatchNorm2_2.momentum = 0
            # for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
            #     gravnet_block.batchnorm_gravnet1.momentum = 0
            #     gravnet_block.batchnorm_gravnet2.momentum = 0

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            if self.args.predict:
                from src.layers.inference_oc import store_at_batch_end
                import pandas as pd

                self.df_showers = pd.concat(self.df_showers)
                self.df_showers_pandora = pd.concat(self.df_showers_pandora)
                self.df_showes_db = pd.concat(self.df_showes_db)
                store_at_batch_end(
                    path_save=os.path.join(
                        self.args.model_prefix, "showers_df_evaluation"
                    ),
                    df_batch=self.df_showers,
                    df_batch_pandora=self.df_showers_pandora,
                    df_batch1=self.df_showes_db,
                    step=0,
                    predict=True,
                    store=True,
                )
            else:
                model_output = self.validation_step_outputs[0][0]
                e_corr = self.validation_step_outputs[0][1]
                batch_g = self.validation_step_outputs[0][2]
                y = self.validation_step_outputs[0][3]
                model_output1 = torch.cat((model_output, e_corr.view(-1, 1)), dim=1)
                create_and_store_graph_output(
                    batch_g,
                    model_output1,
                    y,
                    0,
                    0,
                    0,
                    path_save=os.path.join(
                        self.args.model_prefix, "showers_df_evaluation"
                    ),
                    store=True,
                    predict=False,
                    tracks=self.args.tracks,
                )
        self.validation_step_outputs = []

    def configure_optimizers(self):
        print("Configuring optimizer!")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3
        )
        print("Optimizer params:", filter(lambda p: p.requires_grad, self.parameters()))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=3),
                "interval": "epoch",
                "monitor": "train_loss_epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


class Swin3D(nn.Module):
    """MAIN block
    1) Find coordinates and score for the graph
    2) Do knn down graph
    3) Message passing on the down graph SWIN3D_Blocks
    4) Downsample:
            - find up points
            - find neigh of from down to up
    """

    def __init__(
        self,
        in_dim_node,
        hidden_dim,
        num_heads,
        layer_norm,
        batch_norm,
        residual,
        dropout,
        M,
        k_in,
        n_layers,
    ):
        super().__init__()
        self.k = k_in
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual

        self.send_scores = SendScoresMessage()
        self.find_up = FindUpPoints()
        self.sigmoid_scores = nn.Sigmoid()
        self.funky_coordinate_space = True
        if self.funky_coordinate_space:
            self.embedding_coordinates = nn.Linear(
                in_dim_node, 3
            )  # node feat is an integer
        self.M = M  # number of points up to connect to
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim - 3)

        self.score_ = nn.Linear(hidden_dim, 1)
        self.SWIN3D_Blocks = SWIN3D_Blocks(
            n_layers,
            hidden_dim,
            num_heads,
            dropout,
            self.layer_norm,
            self.batch_norm,
            self.residual,
            possible_empty=True,
        )
        self.Downsample = Downsample_maxpull(hidden_dim, M)

    def forward(self, g, h, c):
        object = g.ndata["particle_number"]
        # 1) Find coordinates and score for the graph
        # embedding to calculate the coordinates in the embedding space #! this could also be kept to the original coordinates
        if self.funky_coordinate_space:
            s_l = self.embedding_coordinates(h)
        else:
            s_l = c
        h = self.embedding_h(h)
        # scores = torch.rand(h.shape[0]).to(h.device)

        # 2) Do knn down graph
        g.ndata["s_l"] = s_l
        g = knn_per_graph(g, s_l, 7)
        g.ndata["h"] = torch.cat((h, s_l), dim=1)

        # 3) Message passing on the down graph SWIN3D_Blocks
        h = self.SWIN3D_Blocks(g)
        scores = torch.sigmoid(self.score_(h))
        g.ndata["scores"] = scores
        g.ndata["particle_number"] = object
        g.ndata["s_l"] = s_l
        g.ndata["h"] = h

        # calculate loss of score
        g.update_all(self.send_scores, self.find_up)
        loss_ud = self.find_up.loss_ud

        # 4) Downsample:
        features, up_points, new_graphs_up, i, j = self.Downsample(g)
        return features, up_points, new_graphs_up, i, j, s_l, loss_ud


class SendScoresMessage(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self):
        super(SendScoresMessage, self).__init__()

    def forward(self, edges):
        score_neigh = edges.src["scores"]
        same_object = edges.dst["particle_number"] == edges.src["particle_number"]
        return {"score_neigh": score_neigh.view(-1), "same_object": same_object}


class FindUpPoints(nn.Module):
    """
    Feature aggregation in a DGL graph
    """

    def __init__(self):
        super(FindUpPoints, self).__init__()
        self.loss_ud = 0

    def forward(self, nodes):
        same_object = nodes.mailbox["same_object"]
        scores_neigh = nodes.mailbox["score_neigh"]
        # loss per neighbourhood of same object as src node
        values_max, index = torch.max(scores_neigh * same_object, dim=1)
        number_points_same_object = torch.sum(same_object, dim=1)
        # print("number_points_same_object", number_points_same_object)
        # print("values_max", values_max)
        loss_u = 1 - values_max
        # loss_d = (
        #     1 / number_points_same_object * torch.sum(scores_neigh * same_object, dim=1)
        # )
        sum_same_object = torch.sum(scores_neigh * same_object, dim=1) - values_max
        # print("sum_same_object", sum_same_object)
        mask_ = number_points_same_object > 0
        if torch.sum(mask_) > 0:
            loss_d = 1 / number_points_same_object[mask_] * sum_same_object[mask_]
            # per neigh measure
            # print("loss_u", loss_u)
            # print("loss_d", torch.mean(loss_d))
            loss_total = loss_u.clone()
            # this takes into account some points not having neigh of the same class
            loss_total[mask_] = loss_u[mask_] + loss_d
            total_loss_ud = torch.mean(loss_total)
            # print("loss ud normal", total_loss_ud)
        else:
            total_loss_ud = torch.mean(loss_u)
            # print("loss ud no neigh", total_loss_ud)
        # print("total_loss_ud", total_loss_ud)
        self.loss_ud = total_loss_ud
        fake_feature = torch.sum(scores_neigh, dim=1)
        return {"new_feat": fake_feature}


class SWIN3D_Blocks(nn.Module):
    """Point 3)
    Just multiple blocks of sparse attention over the down graph
    """

    def __init__(
        self,
        n_layers,
        hidden_dim,
        num_heads,
        layer_norm,
        batch_norm,
        residual,
        dropout,
        possible_empty=True,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.layers_message_passing = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                    self.layer_norm,
                    self.batch_norm,
                    self.residual,
                    possible_empty=True,
                )
                for zz in range(n_layers)
            ]
        )

    def forward(self, g):
        h = g.ndata["h"]
        for ii, conv in enumerate(self.layers_message_passing):
            h = conv(g, h)

        return h


class Downsample_maxpull(nn.Module):
    """Point 4)
    - find up points
    - find neigh of from down to up
    """

    def __init__(self, hidden_dim, M):
        super().__init__()
        self.M = M
        self.embedding_features_to_att = nn.Linear(hidden_dim + 4, hidden_dim)
        self.MLP_difs = MLP_difs_maxpool(hidden_dim, hidden_dim)

    def forward(self, g):
        features = g.ndata["h"]
        list_graphs = dgl.unbatch(g)
        s_l = g.ndata["s_l"]
        graphs_UD = []
        graphs_U = []
        up_points = []
        for i in range(0, len(list_graphs)):
            graph_i = list_graphs[i]
            number_nodes_graph = graph_i.number_of_nodes()

            # find up nodes
            s_l_i = graph_i.ndata["s_l"]
            scores_i = graph_i.ndata["scores"].view(-1)
            device = scores_i.device
            number_up = np.floor(number_nodes_graph * 0.25).astype(int)
            up_points_i_index = torch.flip(torch.sort(scores_i, dim=0)[1], [0])[
                0:number_up
            ]
            up_points_i = torch.zeros_like(scores_i)
            up_points_i[up_points_i_index.long()] = 1
            up_points_i = up_points_i.bool()

            up_points.append(up_points_i)

            # connect down to up
            number_up_points_i = torch.sum(up_points_i)
            if number_up_points_i > 5:
                M_i = 5
            else:
                M_i = number_up_points_i
            nodes = torch.range(start=0, end=number_nodes_graph - 1, step=1).to(device)
            nodes_up = nodes[up_points_i]
            nodes_down = nodes[~up_points_i]

            neigh_indices, neigh_dist_sq = torch_cmspepr.select_knn_directional(
                s_l_i[~up_points_i], s_l_i[up_points_i], M_i
            )
            j = nodes_up[neigh_indices]
            j = j.view(-1)
            i = torch.tile(nodes_down.view(-1, 1), (1, M_i)).reshape(-1)

            g_i = dgl.graph((i.long(), j.long()), num_nodes=number_nodes_graph).to(
                device
            )
            g_i.ndata["h"] = graph_i.ndata["h"]
            g_i.ndata["s_l"] = graph_i.ndata["s_l"]
            g_i.ndata["particle_number"] = graph_i.ndata["particle_number"]
            # find index in original numbering
            graphs_UD.append(g_i)
            # use this way if no message passing between nodes
            # edge_index = torch_cmspepr.knn_graph(s_l_i[up_points_i], k=7)
            # graph_up = dgl.graph(
            #     (edge_index[0], edge_index[1]), num_nodes=len(nodes_up)
            # ).to(device)
            graph_up = dgl.DGLGraph().to(device)
            graph_up.add_nodes(len(nodes_up))
            graph_up.ndata["particle_number"] = g_i.ndata["particle_number"][
                up_points_i
            ]
            graph_up.ndata["s_l"] = g_i.ndata["s_l"][up_points_i]
            graphs_U.append(graph_up)

        graphs_UD = dgl.batch(graphs_UD)
        i, j = graphs_UD.edges()
        graphs_U = dgl.batch(graphs_U)
        # naive way of giving the coordinates gradients
        features = torch.cat((features, s_l, g.ndata["scores"].view(-1, 1)), dim=1)
        features = self.embedding_features_to_att(features)

        # do attention in g connected to up, this features have only been updated for points that have neighbourgs pointing to them: up-points
        features = self.MLP_difs(graphs_UD, features)

        up_points = torch.concat(up_points, dim=0).view(-1)

        return features, up_points, graphs_U, i, j


class MLP_difs_maxpool(nn.Module):
    """
    Param:
    """

    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.edgedistancespassing = EdgePassing(self.in_channels, self.out_channels)
        self.meanmaxaggregation = Max_aggregation(self.out_channels)
        # self.batch_norm = nn.BatchNorm1d(out_dim)
        # self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        # self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)
        # self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h_in = h
        g.ndata["features"] = h
        g.update_all(self.edgedistancespassing, self.meanmaxaggregation)
        h = g.ndata["h_updated"]
        # h = h_in + h
        # h = self.batch_norm(h)
        # h_in2 = h
        # h = self.FFN_layer1(h)
        # h = F.relu(h)
        # h = self.FFN_layer2(h)
        # h = h_in2 + h  # residual connection
        # h = self.batch_norm2(h)
        return h


class EdgePassing(nn.Module):
    """
    Compute the input feature from neighbors
    """

    def __init__(self, in_dim, out_dim):
        super(EdgePassing, self).__init__()
        # self.MLP = nn.Sequential(
        #     nn.Linear(in_dim, out_dim),  #! Dense 3
        #     nn.ReLU(),
        #     nn.Linear(out_dim, 1),  #! Dense 4
        #     nn.ReLU(),
        # )

    def forward(self, edges):
        dif = edges.src["features"]
        # att_weight = self.MLP(dif)
        # att_weight = torch.sigmoid(att_weight)  #! try sigmoid
        # feature = att_weight * edges.src["features"]
        return {"feature_n": dif}


class Max_aggregation(nn.Module):
    """
    Feature aggregation in a DGL graph
    """

    def __init__(self, out_dim):
        super(Max_aggregation, self).__init__()

    def forward(self, nodes):
        max_agg = torch.max(nodes.mailbox["feature_n"], dim=1)[0]

        return {"h_updated": max_agg}


def knn_per_graph(g, sl, k):
    """Build knn for each graph in the batch

    Args:
        g (_type_): original batch of dgl graphs
        sl (_type_): coordinates
        k (_type_): number of neighbours

    Returns:
        _type_: updates batch of dgl graphs with edges
    """
    graphs_list = dgl.unbatch(g)
    node_counter = 0
    new_graphs = []
    for graph in graphs_list:
        non = graph.number_of_nodes()
        sls_graph = sl[node_counter : node_counter + non]
        edge_index = torch_cmspepr.knn_graph(sls_graph, k=k)
        new_graph = dgl.graph(
            (edge_index[0], edge_index[1]), num_nodes=sls_graph.shape[0]
        )
        new_graphs.append(new_graph)
        node_counter = node_counter + non
    return dgl.batch(new_graphs)


def loss_reco_true(e_cor, true_e, sum_e):
    # m = nn.ELU()
    # e_cor = m(e_cor)
    print("corection", e_cor[0:5])
    print("sum_e", sum_e[0:5])
    print("true_e", true_e[0:5])
    # true_e = -1 * sum_e  # Temporarily, to debug - so the model would have to learn corr. factor of -1 for each particle...
    loss = torch.square(((e_cor) * sum_e - true_e) / true_e)
    loss_abs = torch.mean(torch.abs(e_cor * sum_e - true_e) / true_e)
    loss_abs_nocali = torch.mean(torch.abs(sum_e - true_e) / true_e)
    loss = torch.mean(loss)
    return loss, loss_abs, loss_abs_nocali


def loss_reco_sum_absolute(e_cor, true_e, sum_e):
    # implementation of a loss that regresses the sum of the hits instead of the corr. factor ( just for debugging )
    # m = nn.ELU()
    # e_cor = m(e_cor)
    print("corection", e_cor[0:5])
    print("sum_e", sum_e[0:5])
    print("true_e", true_e[0:5])
    # true_e = -1 * sum_e  # Temporarily, to debug - so the model would have to learn corr. factor of -1 for each particle...
    loss = torch.square(e_cor - sum_e)
    loss_abs = torch.mean(torch.abs(e_cor * sum_e - true_e) / true_e)
    loss_abs_nocali = torch.mean(torch.abs(sum_e - true_e) / true_e)
    loss = torch.mean(loss)
    return loss, loss_abs, loss_abs_nocali


class GraphTransformerLayer(nn.Module):
    """
    Param:
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        use_bias=False,
        possible_empty=False,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.possible_empty = possible_empty

        self.attention = MultiHeadAttentionLayer(
            in_dim, out_dim // num_heads, num_heads, use_bias, possible_empty
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

    def forward(self, g, h):
        h_in1 = h  # for first residual connection
        # print("h in attention", h)
        # multi-head attention out
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.out_channels)
        # print("h attention", h)
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O(h)
        # print("h attention 1", h)
        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)
        # print("h attention final", h)
        return h


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias, possible_empty):
        super().__init__()
        self.possible_empty = possible_empty
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
        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(
            eids,
            fn.u_mul_e("V_h", "score", "V_h"),
            fn.sum("V_h", "wV"),  # deprecated in dgl 1.0.1
        )
        # print(g.edata["score"].shape)
        g.send_and_recv(
            eids, fn.copy_e("score", "score"), fn.sum("score", "z")
        )  # copy_e deprecated in dgl 1.0.1
        # print("wV ", g.ndata["wV"])

    def forward(self, g, h):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        # print("Q_h", Q_h)
        # print("K_h", Q_h)
        # print("V_h", Q_h)
        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata["Q_h"] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["K_h"] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["V_h"] = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(g)
        if self.possible_empty:
            # print(g.ndata["wV"].shape, g.ndata["z"].shape, g.ndata["z"].device)
            g.ndata["z"] = g.ndata["z"].tile((1, 1, self.out_dim))
            mask_empty = g.ndata["z"] > 0
            head_out = g.ndata["wV"]
            head_out[mask_empty] = head_out[mask_empty] / (g.ndata["z"][mask_empty])
            g.ndata["z"] = g.ndata["z"][:, :, 0].view(
                g.ndata["wV"].shape[0], self.num_heads, 1
            )
        else:
            head_out = g.ndata["wV"] / g.ndata["z"]
        return head_out


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
