from os import path
import sys

# sys.path.append(
#     path.abspath("/afs/cern.ch/work/m/mgarciam/private/geometric-algebra-transformer/")
# )
# sys.path.append(path.abspath("/mnt/proj3/dd-23-91/cern/geometric-algebra-transformer/"))
from time import time
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar, extract_point, embed_scalar
from torch_scatter import scatter_add, scatter_mean
import torch
import torch.nn as nn
from src.utils.save_features import save_features
from src.logger.plotting_tools import PlotCoordinates
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.layers.obj_cond_inf import calc_energy_loss
from src.models.gravnet_calibration import (
    object_condensation_loss2,
    obtain_batch_numbers,
)
from src.models.gravnet_3_L import obtain_clustering_for_matched_showers
from src.utils.post_clustering_features import (
    get_post_clustering_features,
    calculate_eta,
    calculate_phi,
)
from src.models.energy_correction_NN import (
    ECNetWrapper,
    ECNetWrapperGNN,
    ECNetWrapperGNNGlobalFeaturesSeparate,
    PickPAtDCA,
    AverageHitsP
)
from src.layers.inference_oc import create_and_store_graph_output
import lightning as L
from src.utils.nn.tools import log_losses_wandb_tracking
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from src.layers.inference_oc_tracks import (
    evaluate_efficiency_tracks,
    store_at_batch_end,
)
from src.models.gravnet_3_L_tracking import object_condensation_loss_tracking
from xformers.ops.fmha import BlockDiagonalMask
import os
import wandb

# from src.layers.obtain_statistics import (
#     obtain_statistics_graph_tracking,
#     create_stats_dict,
#     save_stat_dict,
#     plot_distributions,
# )
from src.utils.nn.tools import log_losses_wandb
import torch.nn.functional as F


def criterion(ypred, ytrue, step):
    if True or step < 5000:  # Always use the L1 loss!!
        #### ! using L1 loss for this training only!
        return F.l1_loss(ypred, ytrue)
    else:
        losses = F.l1_loss(ypred, ytrue, reduction="none") / ytrue.abs()
        if len(losses.shape) > 0:
            if int(losses.size(0) * 0.05) > 1:
                top_percentile = torch.kthvalue(
                    losses, int(losses.size(0) * 0.95)
                ).values
                mask = losses > top_percentile
                losses[mask] = 0.0
        return losses.mean()


class ExampleWrapper(L.LightningModule):
    """Example wrapper around a GATr model.

    Expects input data that consists of a point cloud: one 3D point for each item in the data.
    Returns outputs that consists of one scalar number for the whole dataset.

    Parameters
    ----------
    blocks : int
        Number of transformer blocks
    hidden_mv_channels : int
        Number of hidden multivector channels
    hidden_s_channels : int
        Number of hidden scalar channels
    """

    def __init__(
        self,
        args,
        dev,
        input_dim: int = 5,
        output_dim: int = 4,
        n_postgn_dense_blocks: int = 3,
        n_gravnet_blocks: int = 4,
        clust_space_norm: str = "twonorm",
        k_gravnet: int = 7,
        activation: str = "elu",
        weird_batchnom=False,
        blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=64,
    ):
        super().__init__()
        self.strict_loading = False
        self.input_dim = 3
        self.output_dim = 4
        self.loss_final = 0
        self.number_b = 0
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []
        self.args = args
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=2,
            out_s_channels=1,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        )
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        self.clustering = nn.Linear(3, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(2, 1)
        # Load the energy correction module
        if self.args.correction:
            ckpt_neutral = self.args.ckpt_neutral
            ckpt_charged = self.args.ckpt_charged
            if self.args.regress_pos and self.args.ec_model != "dnn":
                print(
                    "Regressing position as well, changing the hardcoded models to sth else"
                )
                print(
                    "Regressing position as well, changing the hardcoded models to sth else"
                )
            if self.args.ec_model == "gat":
                in_features = 17
                if self.args.add_track_chis:
                    in_features += 1
                num_global_features = 14
                self.ec_model_wrapper_charged = ECNetWrapperGNN(
                    device=dev, in_features=in_features + num_global_features
                )
                self.ec_model_wrapper_neutral = ECNetWrapperGNN(
                    device=dev, in_features=in_features + num_global_features
                )
            elif self.args.ec_model == "gat-concat":
                in_features = 17
                if self.args.add_track_chis:
                    in_features += 1
                num_global_features = 14
                self.ec_model_wrapper_charged = ECNetWrapperGNNGlobalFeaturesSeparate(
                    device=dev,
                    in_features_global=num_global_features,
                    in_features_gnn=in_features,
                    ckpt_file=ckpt_charged,
                    gnn=True,
                    pos_regression=self.args.regress_pos,
                )
                self.ec_model_wrapper_neutral = ECNetWrapperGNNGlobalFeaturesSeparate(
                    device=dev,
                    in_features_global=num_global_features,
                    in_features_gnn=in_features,
                    ckpt_file=ckpt_neutral,
                    gnn=True,
                    pos_regression=self.args.regress_pos,
                )
            elif self.args.ec_model == "dnn-neutrals":
                assert self.args.add_track_chis
                num_global_features = 14
                self.ec_model_wrapper_charged = PickPAtDCA()  #  #  #
                #self.ec_model_wrapper_neutral = ECNetWrapperGNNGlobalFeaturesSeparate(
                #    device=dev,
                #    in_features_global=num_global_features,
                #    in_features_gnn=18,
                #    ckpt_file=ckpt_neutral,
                #    gnn=False,
                #    pos_regression=self.args.regress_pos,
                #)
                self.ec_model_wrapper_neutral = AverageHitsP()
            elif self.args.ec_model == "gatr-neutrals":
                assert self.args.add_track_chis
                num_global_features = 14
                print("this is the model for charged")
                if len(self.args.classify_pid_charged):
                    self.pids_charged = [int(x) for x in self.args.classify_pid_charged.split(",")] + [0]
                else:
                    self.pids_charged = []
                if len(self.args.classify_pid_neutral):
                    self.pids_neutral = [int(x) for x in self.args.classify_pid_neutral.split(",")] + [0]
                else:
                    self.pids_neutral = []
                if len(self.pids_charged):
                    print("Also running classification for charged particles", self.pids_charged)
                if len(self.pids_neutral):
                    print("Also running classification for neutral particles", self.pids_neutral)
                if self.args.PID_4_class:
                    self.pids_charged = [0, 1, 2, 3] # electron, CH, NH, gamma
                    self.pids_neutral = [0, 1, 2, 3] # electron, CH, NH, gamma
                    self.pid_conversion_dict = {11: 0, -11: 0, 211: 1, -211: 1, 130: 2, -130: 2, 2112: 2, -2112: 2, 22: 3}
                out_f = 1
                if self.args.regress_pos:
                    out_f += 3
                self.ec_model_wrapper_charged = ECNetWrapperGNNGlobalFeaturesSeparate(
                    device=dev,
                    in_features_global=num_global_features,
                    in_features_gnn=20,
                    ckpt_file=ckpt_charged,
                    gnn=True,
                    gatr=True,
                    pos_regression=self.args.regress_pos,
                    charged=True,
                    pid_channels=len(self.pids_charged),
                    unit_p=self.args.regress_unit_p,
                    out_f=1
                )
                self.ec_model_wrapper_neutral = ECNetWrapperGNNGlobalFeaturesSeparate(
                    device=dev,
                    in_features_global=num_global_features,
                    in_features_gnn=20,
                    ckpt_file=ckpt_neutral,
                    gnn=True,
                    gatr=True,
                    pos_regression=self.args.regress_pos,
                    pid_channels=len(self.pids_neutral),
                    unit_p=self.args.regress_unit_p,
                    out_f=out_f,
                    neutral_avg=False
                )
            else:  # DNN
                # only a DNN for energy correction
                if not self.args.add_track_chis:
                    # self.ec_model_wrapper_charged = NetWrapper(
                    #    "/eos/user/g/gkrzmanc/2024/models/charged22000.pkl", dev
                    # )
                    # self.ec_model_wrapper_neutral = NetWrapper(
                    #    "/eos/user/g/gkrzmanc/2024/models/neutral22000.pkl", dev
                    # )
                    self.ec_model_wrapper_charged = ECNetWrapper(
                        ckpt_file=None, device=dev, in_features=13
                    )
                    self.ec_model_wrapper_neutral = ECNetWrapper(
                        ckpt_file=None, device=dev, in_features=13
                    )
                else:
                    print("using the dnn model")
                    # this is if it's dnn
                    num_global_features = 14
                    self.ec_model_wrapper_charged = (
                        ECNetWrapperGNNGlobalFeaturesSeparate(
                            device=dev,
                            in_features_global=num_global_features,
                            in_features_gnn=18,
                            ckpt_file=ckpt_charged,
                            gnn=False,
                            pos_regression=self.args.regress_pos,
                            charged=True,
                        )
                    )
                    # self.pos_dca = PickPAtDCA()
                    self.ec_model_wrapper_neutral = (
                        ECNetWrapperGNNGlobalFeaturesSeparate(
                            device=dev,
                            in_features_global=num_global_features,
                            in_features_gnn=18,
                            ckpt_file=ckpt_neutral,
                            gnn=False,
                            pos_regression=self.args.regress_pos,
                        )
                    )

        # freeze these models completely
        # for param in self.ec_model_wrapper_charged.model.parameters():
        #    param.requires_grad = False
        # for param in self.ec_model_wrapper_neutral.model.parameters():
        #    param.requires_grad = False

    def forward(self, g, y, step_count, eval="", return_train=False):
        inputs = g.ndata["pos_hits_xyz"]
        # if self.trainer.is_global_zero and step_count % 500 == 0:
        #     g.ndata["original_coords"] = g.ndata["pos_hits_xyz"]
        #     PlotCoordinates(
        #         g,
        #         path="input_coords",
        #         outdir=self.args.model_prefix,
        #         # features_type="ones",
        #         predict=self.args.predict,
        #         epoch=str(self.current_epoch) + eval,
        #         step_count=step_count,
        #     )
        inputs_scalar = g.ndata["hit_type"].view(-1, 1)
        inputs = self.ScaledGooeyBatchNorm2_1(inputs)
        # inputs = inputs.unsqueeze(0)
        embedded_inputs = embed_point(inputs) + embed_scalar(inputs_scalar)
        embedded_inputs = embedded_inputs.unsqueeze(
            -2
        )  # (batch_size*num_points, 1, 16)
        mask = self.build_attention_mask(g)
        scalars = torch.zeros((inputs.shape[0], 1))
        scalars = g.ndata["h"][:, -2:]  # this corresponds to e,p
        # Pass data through GATr
        forward_time_start = time()
        embedded_outputs, scalar_outputs = self.gatr(
            embedded_inputs, scalars=scalars, attention_mask=mask
        )  # (..., num_points, 1, 16)
        forward_time_end = time()
        # wandb.log({"time_gatr_pass": forward_time_end - forward_time_start})
        points = extract_point(embedded_outputs[:, 0, :])

        # Extract scalar and aggregate outputs from point cloud
        nodewise_outputs = extract_scalar(embedded_outputs)  # (..., num_points, 1, 1)
        x_point = points
        x_scalar = torch.cat(
            (nodewise_outputs.view(-1, 1), scalar_outputs.view(-1, 1)), dim=1
        )
        x_cluster_coord = self.clustering(x_point)
        beta = self.beta(x_scalar)
        if self.args.tracks:
            mask = g.ndata["hit_type"] == 1
            beta[mask] = 9
        g.ndata["final_cluster"] = x_cluster_coord
        g.ndata["beta"] = beta.view(-1)
        # if self.trainer.is_global_zero and step_count % 500 == 0:
        #     PlotCoordinates(
        #         g,
        #         path="final_clustering",
        #         outdir=self.args.model_prefix,
        #         predict=self.args.predict,
        #         epoch=str(self.current_epoch) + eval,
        #         step_count=step_count,
        #     )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        pred_energy_corr = torch.ones_like(beta.view(-1, 1)).flatten()
        if self.args.correction:
            # self.args.regress_pos = True
            result = self.forward_correction(g, x, y, return_train)
            # loop through params and print the ones without grad
            #for name, param in self.named_parameters():
            #    if not param.requires_grad:
            #        print("doesn't have grad", name)
            return result
        else:
            pred_energy_corr = torch.ones_like(beta.view(-1, 1))
            return x, pred_energy_corr, 0, 0

    def forward_correction(self, g, x, y, return_train):
        time_matching_start = time()

        (
            graphs_new,
            graphs_high_level_features,
            charged_idx,
            neutral_idx,
            features_neutral_no_nan,
            sum_e,
            pred_pos,
            true_new,
            true_pid,
            true_coords,
            batch_idx,
            e_true_corr_daughters,
            pred_energy_corr,
            pred_pid
        ) = self.clustering_and_global_features(g, x, y)
        # print("   -----  Charged idx:", charged_idx, " Neutral idx:", neutral_idx)
        charged_energies = self.charged_prediction(
            graphs_new, charged_idx, graphs_high_level_features
        )
        neutral_energies = self.neutral_prediction(
            graphs_new, neutral_idx, features_neutral_no_nan
        )
        if self.args.regress_pos:
            if len(self.pids_charged):
                charged_energies, charged_positions, charged_PID_pred = charged_energies
            else:
                charged_energies, charged_positions, _ = charged_energies
            if len(self.pids_neutral):
                neutral_energies, neutral_positions, neutral_PID_pred = neutral_energies
            else:
                neutral_energies, neutral_positions, _ = neutral_energies
        if self.args.explain_ec:
            assert not self.args.regress_pos, "not implemented"
            (
                charged_energies,
                charged_energies_shap_vals,
                charged_energies_ec_x,
            ) = charged_energies
            (
                neutral_energies,
                neutral_energies_shap_vals,
                neutral_energies_ec_x,
            ) = neutral_energies
            shap_vals = (
                torch.ones(
                    graphs_high_level_features.shape[0],
                    charged_energies_shap_vals[0].shape[1],
                )
                .to(graphs_new.ndata["h"].device)
                .detach()
                .cpu()
                .numpy()
            )
            ec_x = torch.zeros(
                graphs_high_level_features.shape[0],
                charged_energies_ec_x.shape[1],
            )
            shap_vals[charged_idx.detach().cpu().numpy()] = charged_energies_shap_vals[
                0
            ]
            shap_vals[neutral_idx.detach().cpu().numpy()] = neutral_energies_shap_vals[
                0
            ]
            ec_x[charged_idx.detach().cpu().numpy()] = charged_energies_ec_x[0]
            ec_x[neutral_idx.detach().cpu().numpy()] = neutral_energies_ec_x[0]
        # dummy loss to make it work without complaining about not using params in loss
        pred_energy_corr[charged_idx.flatten()] = (
            charged_energies / sum_e.flatten()[charged_idx.flatten()]
        )
        pred_energy_corr[neutral_idx.flatten()] = (
            neutral_energies / sum_e.flatten()[neutral_idx.flatten()]
        )
        if len(self.pids_charged):
            if len(charged_idx):
                charged_PID_pred1 = np.array(self.pids_charged)[np.argmax(charged_PID_pred.cpu().detach(), axis=1)]
            else:
                charged_PID_pred1 = []
            pred_pid[charged_idx.flatten()] = torch.tensor(charged_PID_pred1).long().to(charged_idx.device)
        if len(self.pids_neutral):
            if len(neutral_idx):
                neutral_PID_pred1 = np.array(self.pids_neutral)[np.argmax(neutral_PID_pred.cpu().detach(), axis=1)]
            else:
                neutral_PID_pred1 = []
            pred_pid[neutral_idx.flatten()] = torch.tensor(neutral_PID_pred1).long().to(neutral_idx.device)
        pred_energy_corr[pred_energy_corr < 0] = 0.0
        if self.args.regress_pos:
            if len(charged_idx):
                pred_pos[charged_idx.flatten()] = charged_positions
            if len(neutral_idx):
                pred_pos[neutral_idx.flatten()] = neutral_positions
            pred_energy_corr = {
                "pred_energy_corr": pred_energy_corr,
                "pred_pos": pred_pos,
                "neutrals_idx": neutral_idx.flatten(),
                "charged_idx": charged_idx.flatten()
            }
            if len(self.pids_charged) or len(self.pids_neutral):
                pred_energy_corr["pred_PID"] = pred_pid
                pred_energy_corr["charged_PID_pred"] = charged_PID_pred
                pred_energy_corr["neutral_PID_pred"] = neutral_PID_pred
                #if self.args.PID_4_class:
                #    true_pid = np.array([self.pid_conversion_dict.get(x, 3) for x in true_pid])
            #"charged_PID_pred": charged_PID_pred,
            #"neutral_PID_pred": neutral_PID_pred,
            #if len(self.pids_charged):
            #    pred_energy_corr["charged_PID_pred"] = charged_PID_pred
            #if len(self.pids_neutral):
            #    pred_energy_corr["neutral_PID_pred"] = neutral_PID_pred

        if return_train:
            return (
                x,
                pred_energy_corr,
                true_new,
                sum_e,
                true_pid,
                true_new,
                true_coords,
            )
        else:
            if self.args.explain_ec:
                return (
                    x,
                    pred_energy_corr,
                    true_new,
                    sum_e,
                    graphs_new,
                    batch_idx,
                    graphs_high_level_features,
                    true_pid,
                    e_true_corr_daughters,
                    shap_vals,
                    ec_x,
                )
            return (
                x,
                pred_energy_corr,
                true_new,
                sum_e,
                graphs_new,
                batch_idx,
                graphs_high_level_features,
                true_pid,
                e_true_corr_daughters,
                true_coords,
            )

    def charged_prediction(self, graphs_new, charged_idx, graphs_high_level_features):
        # Prediction for chardged particles
        unbatched = dgl.unbatch(graphs_new)
        if len(charged_idx) > 0:
            charged_graphs = dgl.batch([unbatched[i] for i in charged_idx])
            charged_energies = (self.ec_model_wrapper_charged
            .predict(
                graphs_high_level_features[charged_idx],
                charged_graphs,
                explain=self.args.explain_ec,
            ))
        else:
            if not self.args.regress_pos:
                charged_energies = torch.tensor([]).to(graphs_new.ndata["h"].device)
            else:
                charged_energies = [
                    torch.tensor([]).to(graphs_new.ndata["h"].device),
                    torch.tensor([]).to(graphs_new.ndata["h"].device),
                ]
            if len(self.pids_charged):
                charged_energies += [torch.tensor([]).to(graphs_new.ndata["h"].device)]
        return charged_energies

    def neutral_prediction(self, graphs_new, neutral_idx, features_neutral_no_nan):
        unbatched = dgl.unbatch(graphs_new)
        if len(neutral_idx) > 0:
            neutral_graphs = dgl.batch([unbatched[i] for i in neutral_idx])
            neutral_energies = self.ec_model_wrapper_neutral.predict(
                features_neutral_no_nan,
                neutral_graphs,
                explain=self.args.explain_ec,
            )
        else:
            if not self.args.regress_pos:
                neutral_energies = torch.tensor([]).to(graphs_new.ndata["h"].device)
            else:
                neutral_energies = [
                    torch.tensor([]).to(graphs_new.ndata["h"].device),
                    torch.tensor([]).to(graphs_new.ndata["h"].device),
                ]
            if len(self.pids_neutral):
                neutral_energies += [ torch.tensor([]).to(graphs_new.ndata["h"].device) ]
        return neutral_energies

    def clustering_and_global_features(self, g, x, y):
        time_matching_start = time()
        # Match graphs
        (
            graphs_new,
            true_new,
            sum_e,
            true_pid,
            e_true_corr_daughters,
            true_coords,
        ) = obtain_clustering_for_matched_showers(
            g,
            x,
            y,
            self.trainer.global_rank,
            use_gt_clusters=self.args.use_gt_clusters,
        )
        time_matching_end = time()
        # wandb.log({"time_clustering_matching": time_matching_end - time_matching_start})
        batch_num_nodes = graphs_new.batch_num_nodes()
        batch_idx = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
        batch_idx = torch.tensor(batch_idx).to(self.device)
        graphs_new.ndata["h"][:, 0:3] = graphs_new.ndata["h"][:, 0:3] / 3300
        # TODO: add global features to each node here
        graphs_sum_features = scatter_add(graphs_new.ndata["h"], batch_idx, dim=0)
        # now multiply graphs_sum_features so the shapes match
        graphs_sum_features = graphs_sum_features[batch_idx]
        # append the new features to "h" (graphs_sum_features)
        shape0 = graphs_new.ndata["h"].shape
        graphs_new.ndata["h"] = torch.cat(
            (graphs_new.ndata["h"], graphs_sum_features), dim=1
        )
        assert shape0[1] * 2 == graphs_new.ndata["h"].shape[1]
        # print("Also computing graph-level features")
        graphs_high_level_features = get_post_clustering_features(
            graphs_new, sum_e, add_hit_chis=self.args.add_track_chis
        )
        pred_energy_corr = torch.ones(graphs_high_level_features.shape[0]).to(
            graphs_new.ndata["h"].device
        )
        if self.args.regress_pos:
            pred_pos = torch.ones((graphs_high_level_features.shape[0], 3)).to(
                graphs_new.ndata["h"].device
            )
            pred_pid = torch.ones((graphs_high_level_features.shape[0])).to(
                graphs_new.ndata["h"].device
            ).long()
        else:
            pred_pos = None
        node_features_avg = scatter_mean(graphs_new.ndata["h"], batch_idx, dim=0)[
            :, 0:3
        ]
        # energy-weighted node_features_avg
        # node_features_avg = scatter_sum(
        #    graphs_new.ndata["h"][:, 0:3] * graphs_new.ndata["h"][:, 3].view(-1, 1),
        #    batch_idx,
        #    dim=0,
        # )
        # node_features_avg = node_features_avg[:, 0:3]
        weights = graphs_new.ndata["h"][:, 7].view(-1, 1)  # Energies as the weights
        normalizations = scatter_add(weights, batch_idx, dim=0)
        # normalizations1 = torch.ones_like(weights)
        normalizations1 = normalizations[batch_idx]
        weights = weights / normalizations1
        # node_features_avg = scatter_add(
        #    graphs_new.ndata["h"]*weights , batch_idx, dim=0
        # )[: , 0:3]
        # node_features_avg = node_features_avg / normalizations
        eta, phi = calculate_eta(
            node_features_avg[:, 0],
            node_features_avg[:, 1],
            node_features_avg[:, 2],
        ), calculate_phi(node_features_avg[:, 0], node_features_avg[:, 1])
        graphs_high_level_features = torch.cat(
            (graphs_high_level_features, node_features_avg), dim=1
        )
        graphs_high_level_features = torch.cat(
            (graphs_high_level_features, eta.view(-1, 1)), dim=1
        )
        graphs_high_level_features = torch.cat(
            (graphs_high_level_features, phi.view(-1, 1)), dim=1
        )
        # print("Computed graph-level features")
        # print("Shape", graphs_high_level_features.shape)
        # pred_energy_corr = self.GatedGCNNet(graphs_high_level_features)
        num_tracks = graphs_high_level_features[:, 7]
        charged_idx = torch.where(num_tracks >= 1)[0]
        neutral_idx = torch.where(num_tracks < 1)[0]
        # assert their union is the whole set
        assert len(charged_idx) + len(neutral_idx) == len(num_tracks)
        # assert (num_tracks > 1).sum() == 0
        # if (num_tracks > 1).sum() > 0:
        #    print("! Particles with more than one track !")
        #    print((num_tracks > 1).sum().item(), "out of", len(num_tracks))
        assert (
            graphs_high_level_features.shape[0] == graphs_new.batch_num_nodes().shape[0]
        )
        features_neutral_no_nan = graphs_high_level_features[neutral_idx]
        features_neutral_no_nan[features_neutral_no_nan != features_neutral_no_nan] = 0
        # if self.args.ec_model == "gat" or self.args.ec_model == "gat-concat":
        return (
            graphs_new,
            graphs_high_level_features,
            charged_idx,
            neutral_idx,
            features_neutral_no_nan,
            sum_e,
            pred_pos,
            true_new,
            true_pid,
            true_coords,
            batch_idx,
            e_true_corr_daughters,
            pred_energy_corr,
            pred_pid
        )

    def build_attention_mask(self, g):
        """Construct attention mask from pytorch geometric batch.

        Parameters
        ----------
        inputs : torch_geometric.data.Batch
            Data batch.

        Returns
        -------
        attention_mask : xformers.ops.fmha.BlockDiagonalMask
            Block-diagonal attention mask: within each sample, each token can attend to each other
            token.
        """
        batch_numbers = obtain_batch_numbers(g)
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]
        initial_time = time()
        if self.trainer.is_global_zero:
            result = self(batch_g, y, batch_idx)
        else:
            result = self(batch_g, y, 1)
        if self.args.correction:
            (
                model_output,
                e_cor,
                e_true,
                e_sum_hits,
                new_graphs,
                batch_id,
                graph_level_features,
                pid_true_matched,
                e_true_corr_daughters,
                part_coords_matched,
            ) = result
        else:
            (model_output, e_cor, _, _) = result
        if self.args.regress_pos:
            dic = e_cor
            e_cor, pred_pos, neutral_idx, charged_idx = (
                e_cor["pred_energy_corr"],
                e_cor["pred_pos"],
                e_cor["neutrals_idx"],
                e_cor["charged_idx"],
            )
            if len(self.pids_charged):
                charged_PID_pred = dic["charged_PID_pred"]
                charged_PID_true = np.array(pid_true_matched)[dic["charged_idx"].cpu().tolist()]
                # one-hot encoded
                charged_PID_true_onehot = torch.zeros(
                    len(charged_PID_true), len(self.pids_charged)
                )
                if not self.args.PID_4_class:
                    for i in range(len(charged_PID_true)):
                        if charged_PID_true[i] in self.pids_charged:
                            charged_PID_true_onehot[i, self.pids_charged.index(charged_PID_true[i])] = 1
                        else:
                            charged_PID_true_onehot[i, -1] = 1
                else:
                    for i in range(len(charged_PID_true)):
                        charged_PID_true_onehot[i, self.pid_conversion_dict.get(charged_PID_true[i], 3)] = 1
                charged_PID_true_onehot = charged_PID_true_onehot.clone().to(dic["charged_idx"].device)
            if len(self.pids_neutral):
                neutral_PID_pred = dic["neutral_PID_pred"]
                neutral_PID_true = np.array(pid_true_matched)[neutral_idx.cpu()]
                # one-hot encoded
                neutral_PID_true_onehot = torch.zeros(
                    len(neutral_PID_true), len(self.pids_neutral)
                )
                if not self.args.PID_4_class:
                    for i in range(len(neutral_PID_true)):
                        if neutral_PID_true[i] in self.pids_neutral:
                            neutral_PID_true_onehot[i, self.pids_neutral.index(neutral_PID_true[i])] = 1
                        else:
                            neutral_PID_true_onehot[i, -1] = 1
                else:
                    for i in range(len(neutral_PID_true)):
                        neutral_PID_true_onehot[i, self.pid_conversion_dict.get(neutral_PID_true[i], 3)] = 1
                neutral_PID_true_onehot = neutral_PID_true_onehot.to(neutral_idx.device)
        loss_time_start = time()
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
            loss_type=self.args.losstype,
        )
        loss_time_end = time()
        # wandb.log({"loss_comp_time_inside_training": loss_time_end - loss_time_start})
        if self.args.correction:
            # loss_EC = torch.nn.L1Loss()(e_cor * e_sum_hits, e_true)
            step = self.trainer.global_step
            loss_EC = criterion(e_cor * e_sum_hits, e_true_corr_daughters, step)
            if self.args.regress_pos:
                true_pos = torch.tensor(part_coords_matched).to(pred_pos.device)
                if self.args.regress_unit_p:
                    true_pos = (true_pos / torch.norm(true_pos, dim=1).view(-1, 1)).clone()
                loss_pos = torch.nn.L1Loss()(pred_pos, true_pos)
                charged_idx = np.array(sorted(list(set(range(len(e_cor))) - set(neutral_idx))))
                #loss_pos_charged = torch.nn.L1Loss()(pred_pos[charged_idx], true_pos[charged_idx])
                #loss_pos_neutrals = torch.nn.L1Loss()(pred_pos[neutral_idx], true_pos[neutral_idx])
                loss_EC_neutrals = torch.nn.L1Loss()(
                    e_cor[neutral_idx], e_true[neutral_idx]
                )
                # charged idx is e_cor indices minus neutral idx
                charged_idx = np.array(sorted(list(set(range(len(e_cor))) - set(neutral_idx))))
                loss_pos_neutrals = torch.nn.L1Loss()(
                    pred_pos[neutral_idx], true_pos[neutral_idx]
                )
                loss_charged = torch.nn.L1Loss()(
                    pred_pos[charged_idx], true_pos[charged_idx]
                ) # just for logging
                # wandb.log(
                #     {"loss_pxyz": loss_pos, "loss_pxyz_neutrals": loss_pos_neutrals}
                # )
                wandb.log({"loss_EC_neutrals": loss_EC_neutrals, "loss_EC_charged": loss_charged, "loss_p_neutrals": loss_pos_neutrals, "loss_p_charged": loss_charged})
                # print("Loss pxyz neutrals", loss_pos_neutrals)
                loss = loss + loss_pos
                if len(self.pids_charged):
                    loss_charged_pid = torch.nn.CrossEntropyLoss()(
                        charged_PID_pred, charged_PID_true_onehot
                    )
                    loss = loss + loss_charged_pid
                    wandb.log({"loss_charged_pid": loss_charged_pid})
                if len(self.pids_neutral):
                    loss_neutral_pid = torch.nn.CrossEntropyLoss()(
                        neutral_PID_pred, neutral_PID_true_onehot
                    )
                    loss = loss + loss_neutral_pid
                    wandb.log({"loss_neutral_pid": loss_neutral_pid})
            # loss_EC=torch.nn.L1Loss()(e_cor * e_sum_hits, e_true_corr_daughters)
            # wandb.log({"loss_EC": loss_EC})
            loss = loss + loss_EC
            # loss = loss_EC
            if self.args.save_features:
                cluster_features_path = os.path.join(
                    self.args.model_prefix, "cluster_features"
                )
                if not os.path.exists(cluster_features_path):
                    os.makedirs(cluster_features_path)
                save_features(
                    cluster_features_path,
                    {
                        "x": graph_level_features.detach().cpu(),
                        # """ "xyz_covariance_matrix": covariances.cpu(),"""
                        "e_true": e_true.detach().cpu(),
                        "e_reco": e_cor.detach().cpu(),
                        "true_e_corr": (e_true / e_sum_hits - 1).detach().cpu(),
                        "e_true_corrected_daughters": e_true_corr_daughters.detach().cpu(),
                        # "node_features_avg": scatter_mean(
                        #    batch_g.ndata["h"], batch_idx, dim=0
                        # ),  # graph-averaged node features
                        "coords_y": part_coords_matched,
                        "pid_y": pid_true_matched,
                    },
                )

        misc_time_start = time()
        if self.trainer.is_global_zero:
            log_losses_wandb(True, batch_idx, 0, losses, loss, 0)
        self.loss_final = loss.item() + self.loss_final
        self.number_b = self.number_b + 1
        del model_output
        del e_cor
        del losses
        final_time = time()
        # wandb.log({"misc_time_inside_training": final_time - misc_time_start})
        # wandb.log({"training_step_time": final_time - initial_time})
        return loss

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
        shap_vals, ec_x = None, None
        if self.args.correction:
            result = self(batch_g, y, 1)
            if self.args.explain_ec:
                (
                    model_output,
                    e_cor,
                    e_true,
                    e_sum_hits,
                    new_graphs,
                    batch_id,
                    graph_level_features,
                    pid_true_matched,
                    e_true_corr_daughters,
                    shap_vals,
                    ec_x,
                ) = result
            else:
                (
                    model_output,
                    e_cor,
                    e_true,
                    e_sum_hits,
                    new_graphs,
                    batch_id,
                    graph_level_features,
                    pid_true_matched,
                    e_true_corr_daughters,
                    coords_true,
                ) = result
            if self.args.regress_pos:
                if len(self.pids_charged):
                    charged_PID_pred = e_cor["charged_PID_pred"]
                    #charged_PID_pred =  np.array(self.pids_charged + [0])[np.argmax(charged_PID_pred.cpu(), axis=1)]
                    charged_idx = e_cor["charged_idx"]
                    #charged_PID_true = np.array(pid_true_matched)[charged_idx.cpu()]
                    #if self.args.PID_4_class:
                    #    charged_PID_true = np.array([self.pid_conversion_dict.get(x, 3) for x in charged_PID_true])
                    #pid_list[charged_idx] = charged_PID_pred
                if len(self.pids_neutral):
                    neutral_idx = e_cor["neutrals_idx"]
                    #neutral_PID_pred = e_cor["neutral_PID_pred"]
                    #neutral_PID_pred = np.array(self.pids_neutral + [0])[np.argmax(neutral_PID_pred.cpu(), axis=1)]
                    #neutral_PID_true = np.array(pid_true_matched)[neutral_idx.cpu()]
                    #if self.args.PID_4_class:
                    #    neutral_PID_true = np.array([self.pid_conversion_dict.get(x, 3) for x in neutral_PID_true])
                pred_pid = e_cor["pred_PID"]
                e_cor, pred_pos = e_cor["pred_energy_corr"], e_cor["pred_pos"]
                #pid_list = np.zeros_like(e_cor)
            else:
                pred_pos = None
            loss_ll = 0
            e_cor1 = torch.ones_like(model_output[:, 0].view(-1, 1))
        else:
            model_output, e_cor1, loss_ll, _ = self(batch_g, y, 1)
            loss_ll = 0
            e_cor1 = torch.ones_like(model_output[:, 0].view(-1, 1))
            e_cor = e_cor1
            pred_pos = None
        preds = model_output.squeeze()
        # if self.global_step < 200:
        #     self.args.losstype = "hgcalimplementation"
        # else:
        #     self.args.losstype = "vrepweighted"
        (loss, losses, loss_E, loss_E_frac_true,) = object_condensation_loss2(
            batch_g,
            model_output,
            e_cor1,
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
            loss_type=self.args.losstype,
        )
        loss_ec = 0

        print("starting validation step", batch_idx, loss)
        if self.trainer.is_global_zero:
            log_losses_wandb(
                True, batch_idx, 0, losses, loss, loss_ll, loss_ec, val=True
            )
        if self.args.explain_ec:
            self.validation_step_outputs.append(
                [model_output, e_cor, batch_g, y, shap_vals, ec_x]
            )
        else:
            self.validation_step_outputs.append([model_output, e_cor, batch_g, y])
        if self.args.predict:
            if self.args.correction:
                model_output1 = model_output
                e_corr = e_cor
            else:
                model_output1 = torch.cat((model_output, e_cor.view(-1, 1)), dim=1)
                e_corr = None

            (
                df_batch_pandora,
                df_batch1,
                self.total_number_events,
            ) = create_and_store_graph_output(
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
                shap_vals=shap_vals,
                ec_x=ec_x,
                total_number_events=self.total_number_events,
                pred_pos=pred_pos,
                pred_pid=pred_pid,
                use_gt_clusters=self.args.use_gt_clusters,
                pids_neutral=self.pids_neutral,
                pids_charged=self.pids_charged,
            )
            # self.df_showers.append(df_batch)
            self.df_showers_pandora.append(df_batch_pandora)
            print("Appending another batch", len(df_batch1))
            self.df_showes_db.append(df_batch1)

        del losses
        del loss
        del model_output

    def on_train_epoch_end(self):
        self.log("train_loss_epoch", self.loss_final / self.number_b)

    def on_train_epoch_start(self):
        # if self.trainer.is_global_zero and self.current_epoch == 0:
        #     self.stat_dict = {}
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.total_number_events = 0
        self.make_mom_zero()
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            print("making momentum 0")
            self.ScaledGooeyBatchNorm2_1.momentum = 0

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            if self.args.predict:
                from src.layers.inference_oc import store_at_batch_end
                import pandas as pd
                if self.args.explain_ec:
                    shap_vals = self.validation_step_outputs[0][4]
                    path_shap_vals = os.path.join(
                        self.args.model_prefix, "shap_vals.pkl"
                    )
                    torch.save(shap_vals, path_shap_vals)
                    print("SHAP values saved!")
                # self.df_showers = pd.concat(self.df_showers)
                self.df_showers_pandora = pd.concat(self.df_showers_pandora)
                self.df_showes_db = pd.concat(self.df_showes_db)
                print(self.df_showes_db.keys())
                store_at_batch_end(
                    path_save=os.path.join(
                        self.args.model_prefix, "showers_df_evaluation"
                    ),
                    # df_batch=self.df_showers,
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
                shap_vals = None
                ec_x = None
                if self.args.explain_ec:
                    shap_vals = self.validation_step_outputs[0][4]
                    ec_x = self.validation_step_outputs[0][5]
                if self.args.correction:
                    model_output1 = model_output
                    e_corr = e_corr
                else:
                    model_output1 = torch.cat((model_output, e_corr.view(-1, 1)), dim=1)
                    e_corr = None
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
                    e_corr=e_corr,
                    tracks=self.args.tracks,
                    shap_vals=shap_vals,
                    ec_x=ec_x,
                    use_gt_clusters=self.args.use_gt_clusters,
                )
                del model_output1
                del batch_g
        self.validation_step_outputs = []
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def configure_optimizers(self):
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


def obtain_batch_numbers(g):
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes))
        num_nodes = gj.number_of_nodes()
    batch = torch.cat(batch_numbers, dim=0)
    return batch
