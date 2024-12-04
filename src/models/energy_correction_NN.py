"""
    PID predict energy correction
    The model taken from notebooks/train_energy_correction_head.py
    At first the model is fixed and the weights are loaded from earlier training
"""
import wandb
from xformers.ops.fmha import BlockDiagonalMask
from gatr.interface import (
    embed_point,
    extract_point,
    extract_translation,
    embed_scalar,
    extract_scalar
)
from src.layers.utils_training import obtain_batch_numbers, obtain_clustering_for_matched_showers
from torch_scatter import scatter_add, scatter_mean
from src.utils.post_clustering_features import (
    get_post_clustering_features,
    get_extra_features,
    calculate_eta,
    calculate_phi,
)
from src.utils.save_features import save_features

import os
from time import time
import numpy as np
from gatr import GATr, SelfAttentionConfig, MLPConfig
import pickle
from copy import deepcopy
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.models.thrust_axis import Thrust, hits_xyz_to_momenta, LR, weighted_least_squares_line
from src.utils.pid_conversion import pid_conversion_dict
from torch_geometric.nn.models import GAT, GraphSAGE
from torch_scatter import scatter_mean, scatter_sum
from gatr import GATr
import dgl


class Net(nn.Module):
    def __init__(self, in_features=13, out_features=1, return_raw=True):
        super(Net, self).__init__()
        self.out_features = out_features
        self.return_raw = return_raw
        self.model = nn.ModuleList(
            [
                # nn.BatchNorm1d(13),
                nn.Linear(in_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                # nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, out_features),
            ]
        )
        self.explainer_mode = False

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        for layer in self.model:
            x = layer(x)
        if self.out_features > 1 and not self.return_raw:
            return x[:, 0], x[:, 1:]
        if self.explainer_mode:
            return x.numpy()
        return x

    def freeze_batchnorm(self):
        for layer in self.model:
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()
                print("Frozen batchnorm in 1st layer only - ", layer)
                break

import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)

class EnergyCorrectionWrapper(torch.nn.Module):
    def __init__(
        self,
        device,
        in_features_global=13,
        in_features_gnn=13,
        out_features_gnn=16,
        ckpt_file=None,
        gnn=True,
        pos_regression=False,
        gatr=False,
        charged=False,
        unit_p=False,
        pid_channels=0,  # PID: list of possible PID values to classify using an additional head. If empty, don't do PID.
        out_f=1,
        ignore_global_features_for_p=True,  # Whether to ignore the high-level features for the momentum regression and just use the GATr outputs
        neutral_avg=False,
        neutral_PCA=False,
        neutral_thrust_axis=True,
        simple_p_GNN=False,
        predict=True,
        args=None
    ):
        super(EnergyCorrectionWrapper, self).__init__()
        if not charged:
            self.ec_model_wrapper_neutral_avg = ECNetWrapperAvg()
        self.charged = charged
        self.args = args
        self.predict_arg = predict
        self.simple_p_GNN = simple_p_GNN
        self.neutral_avg = neutral_avg
        self.pos_regression = pos_regression
        self.unit_p = unit_p
        self.neutral_PCA = neutral_PCA
        self.neutral_thrust_axis = neutral_thrust_axis
        self.use_gatr = gatr
        self.separate_pid_gatr = args.separate_PID_GATr
        self.n_layers_pid_head = args.n_layers_PID_head
        print("pos_regression", self.pos_regression)
        # if pos_regression:
        #     out_f += 3
        self.ignore_global_features_for_p = ignore_global_features_for_p
        if self.charged:
            self.ignore_global_features_for_p = False
        if not self.charged:
            self.model = Net(
                in_features=out_features_gnn + in_features_global, out_features=out_f,
                return_raw=True
            )
            self.model.explainer_mode = False
        # use a GAT
        if gnn:
            if self.use_gatr:
                self.gatr = GATr(
                    in_mv_channels=1,
                    out_mv_channels=1,
                    hidden_mv_channels=4,
                    in_s_channels=3,
                    out_s_channels=None,
                    hidden_s_channels=4,
                    num_blocks=3,
                    attention=SelfAttentionConfig(),  # Use default parameters for attention
                    mlp=MLPConfig(),  # Use default parameters for MLP
                )
                # self.lin_e = nn.Linear(4, 1)
                self.gnn = "gatr"
                # self.lin_final_exp = nn.Linear(16, 4) # e, pxpypz
                if self.separate_pid_gatr and not self.charged:
                    print("Separate PID GATr")
                    self.gatr_pid = GATr(
                        in_mv_channels=1,
                        out_mv_channels=1,
                        hidden_mv_channels=4,
                        in_s_channels=3,
                        out_s_channels=None,
                        hidden_s_channels=4,
                        num_blocks=3,
                        attention=SelfAttentionConfig(),  # Use default parameters for attention
                        mlp=MLPConfig(),  # Use default parameters for MLP
                    )
            else:
                self.gnn = GAT(
                    in_features_gnn,
                    out_channels=out_features_gnn,
                    heads=4,
                    concat=True,
                    hidden_channels=64,
                    num_layers=3,
                )
            # self.gnn = GraphSAGE(in_channels=in_features_gnn, out_channels=out_features_gnn, hidden_channels=64, num_layers=3)
        else:
            self.gnn = None
        self.pid_channels = pid_channels
        if pid_channels > 1: # 1 is just the 'other' category
            n_layers = self.n_layers_pid_head
            if n_layers == 1:
                self.PID_head = nn.Linear(out_features_gnn + in_features_global, pid_channels)   # Additional head for PID classification
            else:
                self.PID_head = nn.ModuleList()
                self.PID_head.append(nn.Linear(out_features_gnn + in_features_global, 64))
                for i in range(n_layers - 1):
                    self.PID_head.append(nn.Linear(64, 64))
                    self.PID_head.append(nn.ReLU())
                self.PID_head.append(nn.Linear(64, pid_channels))
                self.PID_head = nn.Sequential(*self.PID_head)
            self.PID_head.to(device)
        if ckpt_file is not None and ckpt_file != "" and not self.charged:
            # self.model.model = pickle.load(open(ckpt_file, 'rb'))
            with open(ckpt_file.strip(), "rb") as f:
                self.model.model = CPU_Unpickler(f).load()
                # if self.use_gatr:
                #     self.gatr = CPU_Unpickler(f).load()
            print("Loaded energy correction model weights from ECNetWrapperGNNGlobalFeaturesSeparate", ckpt_file)

        else:
            print("Not loading energy correction model weights")
        if not self.charged:
            self.model.to(device)
        self.PickPAtDCA = PickPAtDCA()
        self.AvgHits = AverageHitsP()
        self.NeutralPCA = NeutralPCA()
        self.ThrustAxis = ThrustAxis()

    def charged_prediction(self, graphs_new, charged_idx, graphs_high_level_features):
        # Prediction for charged particles
        unbatched = dgl.unbatch(graphs_new)
        if len(charged_idx) > 0:
            charged_graphs = dgl.batch([unbatched[i] for i in charged_idx])
            charged_energies = self.predict(
                graphs_high_level_features,
                charged_graphs,
                explain=self.args.explain_ec,
            )
        else:
            if not self.args.regress_pos:
                charged_energies = torch.tensor([]).to(graphs_new.ndata["h"].device)
            else:
                charged_energies = [
                    torch.tensor([]).to(graphs_new.ndata["h"].device),
                    torch.tensor([]).to(graphs_new.ndata["h"].device),
                    torch.tensor([]).to(graphs_new.ndata["h"].device),
                ]
            if self.pid_channels:
                charged_energies += [torch.tensor([]).to(graphs_new.ndata["h"].device)]
        return charged_energies

    def neutral_prediction(self, graphs_new, neutral_idx, features_neutral_no_nan):
        unbatched = dgl.unbatch(graphs_new)
        if len(neutral_idx) > 0:
            neutral_graphs = dgl.batch([unbatched[i] for i in neutral_idx])
            neutral_energies = self.predict(
                features_neutral_no_nan,
                neutral_graphs,
                explain=self.args.explain_ec,
            )
            neutral_pxyz_avg = self.ec_model_wrapper_neutral_avg.predict(
                features_neutral_no_nan,
                neutral_graphs,
                explain=self.args.explain_ec,
            )[1]
        else:
            if not self.args.regress_pos:
                neutral_energies = torch.tensor([]).to(graphs_new.ndata["h"].device)
            else:
                neutral_energies = [
                    torch.tensor([]).to(graphs_new.ndata["h"].device),
                    torch.tensor([]).to(graphs_new.ndata["h"].device),
                    torch.tensor([]).to(graphs_new.ndata["h"].device),
                        ]
            if self.pid_channels:
                neutral_energies += [ torch.tensor([]).to(graphs_new.ndata["h"].device) ]
        return neutral_energies, neutral_pxyz_avg
    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        if graphs_new is not None and self.gnn is not None:
            batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
            batch_idx = []
            batch_bounds = []
            for i, n in enumerate(batch_num_nodes):
                batch_idx.extend([i] * n)
                batch_bounds.append(n)
            batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
            x = graphs_new.ndata["h"]
            edge_index = torch.stack(graphs_new.edges())
            hits_points = graphs_new.ndata["h"][:, 0:3]
            hit_type = graphs_new.ndata["h"][:, 3:7].argmax(dim=1)
            betas = graphs_new.ndata["h"][:, 9]
            p = graphs_new.ndata["h"][:, 8]
            e = graphs_new.ndata["h"][:, 7]
            embedded_inputs = embed_point(hits_points) + embed_scalar(
                hit_type.view(-1, 1)
            )
            extra_scalars = torch.cat(
                [betas.unsqueeze(1), p.unsqueeze(1), e.unsqueeze(1)], dim=1
            )
            mask = self.build_attention_mask(graphs_new)
            embedded_inputs = embedded_inputs.unsqueeze(-2)
            embedded_outputs, _ = self.gatr(
                embedded_inputs, scalars=extra_scalars, attention_mask=mask
            )
            p_vectors = extract_translation(embedded_outputs)
            p_vectors = p_vectors[:, 0, :]
            p_vectors_per_batch = scatter_mean(p_vectors, batch_idx, dim=0)
            embedded_outputs_per_batch = scatter_sum(
                embedded_outputs[:, 0, :], batch_idx, dim=0
            )
            model_x = torch.cat(
                [x_global_features, embedded_outputs_per_batch], dim=1
            )
            if self.separate_pid_gatr and not self.charged:
                embedded_outputs, _ = self.gatr_pid(
                    embedded_inputs, scalars=extra_scalars, attention_mask=mask
                )
                p_vectors = extract_translation(embedded_outputs)
                p_vectors = p_vectors[:, 0, :]
                p_vectors_per_batch = scatter_mean(p_vectors, batch_idx, dim=0)
                embedded_outputs_per_batch1 = scatter_sum(
                    embedded_outputs[:, 0, :], batch_idx, dim=0
                )
                model_x_pid = torch.cat(
                    [x_global_features, embedded_outputs_per_batch1], dim=1
                )
            else:
                model_x_pid = model_x
        else:
            # not using GATr features
            gnn_output = torch.randn(x_global_features.shape[0], 32).to(
                x_global_features.device
            )
        #if not self.use_gatr:
        #    model_x = torch.cat([x_global_features, gnn_output], dim=1).to(
        #        self.model.model[0].weight.device
        #    )
        if not self.charged:
            # Predict energy for neutrals using the neural network
            res = self.model(model_x)
        if self.pid_channels > 1:
            pid_pred = self.PID_head(model_x_pid)
        else:
            pid_pred = None
        if self.pos_regression:
            if self.charged:
                p_tracks, pos, ref_pt_pred = self.PickPAtDCA.predict(x_global_features, graphs_new)
                E = torch.norm(pos, dim=1)
                if self.unit_p:
                    pos = (pos / torch.norm(pos, dim=1).unsqueeze(1)).clone()
                return E, pos, pid_pred, ref_pt_pred
            else:
                E_pred = res[:, 0]
                E_pred = torch.clamp(E_pred, min=0, max=None)
                _, _, ref_pt_pred = self.AvgHits.predict(x_global_features, graphs_new)
                if self.neutral_avg:
                    _, p_pred, ref_pt_pred = self.AvgHits.predict(x_global_features, graphs_new)
                elif self.neutral_PCA:
                    _, p_pred, ref_pt_pred = self.NeutralPCA.predict(x_global_features, graphs_new)
                elif self.neutral_thrust_axis:
                    _, p_pred, ref_pt_pred = self.ThrustAxis.predict(x_global_features, graphs_new)
                else:
                    p_pred = res[:, 1:4]
                    raise NotImplementedError
                if self.unit_p:
                    p_pred = (p_pred / torch.norm(p_pred, dim=1).unsqueeze(1)).clone()
                return E_pred, p_pred, pid_pred, ref_pt_pred
        else:
            # normalize the vectors
            # E = torch.clamp(res[0].flatten(), min=0, max=None)
            # p = res[1]  # / torch.norm(res[1], dim=1).unsqueeze(1)
            # if self.use_gatr and not use_full_mv:
            #     p = p_vectors_per_batch
            # return E, p
            return torch.clamp(res[:, 0], min=0, max=None)
    @staticmethod
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

    def build_attention_mask(self, g):
        batch_numbers = self.obtain_batch_numbers(g)
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )

class ECNetWrapperAvg(torch.nn.Module):
    # use the GNN+NN model for energy correction
    # This one concatenates GNN features to the global features
    def __init__(self):
        super(ECNetWrapperAvg, self).__init__()
        self.AvgHits = AverageHitsP()

    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        _, p_pred, _ = self.AvgHits.predict(x_global_features, graphs_new)
        p_pred = (p_pred / torch.norm(p_pred, dim=1).unsqueeze(1)).clone()
        return None, p_pred, None, None
    @staticmethod
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

    def build_attention_mask(self, g):
        batch_numbers = self.obtain_batch_numbers(g)
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )

class PickPAtDCA(torch.nn.Module):
    # Same layout of the module as the GNN one, but just picks the track
    def __init__(self):
        super(PickPAtDCA, self).__init__()

    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        assert graphs_new is not None
        batch_num_nodes = graphs_new.batch_num_nodes()  # Num. hits in each graph
        batch_idx = []
        batch_bounds = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
        # ht = graphs_new.ndata["hit_type"]
        ht = graphs_new.ndata["h"][:, 3:7].argmax(dim=1)
        filt = ht == 1  # track
        filt_hits = ((ht == 2) + (ht == 3)).bool()
        # if "pos_pxpypz_at_vertex" in graphs_new.ndata.keys():
        #    key = "pos_pxpypz_at_vertex"
        # else:
        #    key = "pos_pxpypz"
        # p_direction = scatter_mean(
        #     graphs_new.ndata["pos_pxpypz_at_vertex"][filt], batch_idx[filt], dim=0
        # )
        # take the min chi squared track if there are multiple
        p_direction, p_xyz = pick_lowest_chi_squared(
            graphs_new.ndata["pos_pxpypz_at_vertex"][filt],
            graphs_new.ndata["chi_squared_tracks"][filt],
            batch_idx[filt],
            graphs_new.ndata["h"][filt, :3]
        )
        # Barycenters of clusters of hits
        xyz_hits = graphs_new.ndata["h"][:, :3]
        E_hits = graphs_new.ndata["h"][:, 8]
        weighted_avg_hits = scatter_sum(xyz_hits * E_hits.unsqueeze(1), batch_idx, dim=0)
        E_total = scatter_sum(E_hits, batch_idx, dim=0)
        barycenters = weighted_avg_hits / E_total.unsqueeze(1)
        p_tracks = torch.norm(p_direction, dim=1)
        p_direction = p_direction  # / torch.norm(p_direction, dim=1).unsqueeze(1)
        return p_tracks, p_direction, barycenters - p_xyz   # torch.concat([barycenters, p_xyz], dim =1) # Reference point

class AverageHitsP(torch.nn.Module):
    # Same layout of the module as the GNN one, but just computes the average of the hits. Try to compare this + ML clustering with Pandora
    def __init__(self):
        super(AverageHitsP, self).__init__()
    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        assert graphs_new is not None
        batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
        batch_idx = []
        batch_bounds = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
        xyz_hits = graphs_new.ndata["h"][:, :3]
        E_hits = graphs_new.ndata["h"][:, 8]
        weighted_avg_hits = scatter_sum(xyz_hits * E_hits.unsqueeze(1), batch_idx, dim=0)
        E_total = scatter_sum(E_hits, batch_idx, dim=0)
        p_direction = weighted_avg_hits / E_total.unsqueeze(1)
        p_tracks = torch.norm(p_direction, dim=1)
        p_direction = p_direction / torch.norm(p_direction, dim=1).unsqueeze(1)
        # if self.pos_regression:
        return p_tracks, p_direction,  weighted_avg_hits / E_total.unsqueeze(1) * 3300 # Reference point
        # return p_tracks

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from wpca import PCA, WPCA, EMPCA  # The sklearn PCA doesn't support weights so we're using Weighted PCA here

class NeutralPCA(torch.nn.Module):
    # Same layout of the module as the GNN one, but just computes the direction of the shower.
    def __init__(self):
        super(NeutralPCA, self).__init__()

    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        assert graphs_new is not None
        batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
        batch_idx = []
        batch_bounds = []
        p_directions = []
        barycenters = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        for i in np.unique(batch_idx):
            #w = WPCA(n_components=1)
            w = PCA(n_components=1) # Try unweighted PCA for debugging
            weights = graphs_new.ndata["h"][np.array(batch_idx) == i, 7].detach().cpu().reshape(-1, 1)
            weights = weights / torch.sum(weights)
            # repeat weights 3 times
            weights = np.repeat(weights, 3, axis=1)
            hits_xyz = graphs_new.ndata["h"][np.array(batch_idx) == i, :3].detach().cpu()
            w.fit(hits_xyz)   #, weights=weights)
            k = torch.tensor(w.components_[0])
            # mask = dist_from_first_pca < 50
            # only keep the 90% closest hits
            mean = torch.tensor(w.mean_)
            #norm1 =  torch.norm(mean + k)
            #norm2 = torch.norm(mean - k)
            #if norm1 < norm2:
            #    k *= -1
            a = hits_xyz - mean
            dist_from_first_pca = np.sqrt(np.linalg.norm(a, axis=1) ** 2 - np.dot(a, k) ** 2)
            mask = dist_from_first_pca < np.quantile(dist_from_first_pca, 0.9)
            if mask.sum() == 0:
                #mask = dist_from_first_pca < np.quantile(dist_from_first_pca, 0.95)
                mask = np.ones_like(mask)
            hits_filtered = hits_xyz[mask]
            hits_E_filtered = graphs_new.ndata["h"][np.array(batch_idx) == i, 7][mask].detach().cpu().numpy()
            k = weighted_least_squares_line(hits_filtered, hits_E_filtered)[1]
            k = torch.tensor(k)
            k /= torch.norm(k)
            if np.dot(k, mean) < 0:
                k *= -1
            # Figure out the direction
            p_directions.append(k)
            barycenters.append(mean)
            #print(graphs_new.ndata["h"][batch_idx == i, :3])
            #print(w.components_)
            #print("-------------------")
        p_direction = torch.stack(p_directions)
        #batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
        #xyz_hits = graphs_new.ndata["h"][:, :3]
        #E_hits = graphs_new.ndata["h"][:, 7]
        #weighted_avg_hits = scatter_sum(xyz_hits * E_hits.unsqueeze(1), batch_idx, dim=0)
        # get the principal axis
        #E_total = scatter_sum(E_hits, batch_idx, dim=0)
        #p_direction = weighted_avg_hits / E_total.unsqueeze(1)
        p_tracks = torch.norm(p_direction, dim=1)
        p_direction = p_direction  / torch.norm(p_direction, dim=1).unsqueeze(1)
        # if self.pos_regression:
        return p_tracks, p_direction, torch.stack(barycenters)*3300# reference point
        # return p_tracks

class ThrustAxis(torch.nn.Module):
    #  Same layout of the module as the GNN one, but just computes the direction of the shower by finding the Thrust Axis.
    def __init__(self):
        super(ThrustAxis, self).__init__()
    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        assert graphs_new is not None
        batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
        batch_idx = []
        batch_bounds = []
        p_directions = []
        barycenters = []
        p_dir_avg = []  # For debugging
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        for i in np.unique(batch_idx):
            hits_xyz = graphs_new.ndata["h"][np.array(batch_idx) == i, :3].detach().cpu().numpy()
            hits_E = graphs_new.ndata["h"][np.array(batch_idx) == i, 7].detach().cpu().numpy()
            momenta = hits_xyz_to_momenta(hits_xyz, hits_E)
            #thrust_axis = Thrust.calculate_thrust(momenta)
            #thrust_axis = LR.calculate_thrust(hits_xyz, hits_E)
            thrust_axis = weighted_least_squares_line(hits_xyz, np.ones_like(hits_E))[1]
            thrust_axis /= np.linalg.norm(thrust_axis)
            barycenter = np.average(hits_xyz, weights=hits_E, axis=0)
            dot_prod = np.dot(thrust_axis, barycenter)
            if dot_prod < 0:
                thrust_axis = -thrust_axis
            p_directions.append(torch.tensor(thrust_axis))
            barycenters.append(torch.tensor(barycenter)*3300)
        p_direction = torch.stack(p_directions)
        p_tracks = torch.norm(p_direction, dim=1)
        p_direction = p_direction / torch.norm(p_direction, dim=1).unsqueeze(1)
        barycenters = torch.stack(barycenters)
        return p_tracks, p_direction, barycenters # ref pt

def pick_lowest_chi_squared(pxpypz, chi_s, batch_idx, xyz_nodes):
    unique_batch = torch.unique(batch_idx)
    p_direction = []
    track_xyz = []
    for i in range(0, len(unique_batch)):
        mask = batch_idx == unique_batch[i]
        if torch.sum(mask) > 1:
            chis = chi_s[mask]
            ind_min = torch.argmin(chis)
            p_direction.append(pxpypz[mask][ind_min].view(-1, 3))
            track_xyz.append(xyz_nodes[mask][ind_min].view(-1, 3))

        else:
            p_direction.append(pxpypz[mask].view(-1, 3))
            track_xyz.append(xyz_nodes[mask].view(-1, 3))
    return torch.concat(p_direction, dim=0), torch.stack(track_xyz)[:, 0]


class EnergyCorrection():
    def __init__(self, main_model):
        #super(EnergyCorrection, self).__init__()
        self.args = main_model.args
        self.get_PID_categories(main_model)
        self.get_energy_correction(main_model)
        self.pid_conversion_dict = pid_conversion_dict
        self.main_model = main_model
    def get_PID_categories(self, main_model):
        assert main_model.args.add_track_chis
        if len(main_model.args.classify_pid_charged):
            pids_charged = [int(x) for x in self.args.classify_pid_charged.split(",")]
        else:
            pids_charged = []
        if len(self.args.classify_pid_neutral):
            pids_neutral = [int(x) for x in self.args.classify_pid_neutral.split(",")]
        else:
            pids_neutral = []
        if len(pids_charged):
            print("Also running classification for charged particles", self.pids_charged)
        if len(pids_neutral):
            print("Also running classification for neutral particles", self.pids_neutral)
        pids_charged = [0, 1, 2, 3]  # electron, CH, NH, gamma, muon
        pids_neutral = [0, 1, 2, 3]  # electron, CH, NH, gamma, muon (not implemented yet)
        if self.args.restrict_PID_charge:
            print("Restricting PID classification to match charge")
            pids_charged = [0, 1]
            pids_neutral = [2, 3]
        if self.args.is_muons:
            pids_charged += [4]
            if not self.args.restrict_PID_charge:
                pids_neutral += [4]
        self.pids_charged = pids_charged
        self.pids_neutral = pids_neutral

    def get_energy_correction(self, main_model):
        # To be called by the model to initialize the energy correction modules
        ckpt_neutral = main_model.args.ckpt_neutral
        ckpt_charged = main_model.args.ckpt_charged
        dev = main_model.dev
        num_global_features = 14
        if main_model.args.is_muons:
            num_global_features += 2 # for the muon calorimeter hits and the number of muon hits
        self.model_charged = EnergyCorrectionWrapper(
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
            out_f=1,
            neutral_avg=False,
            neutral_PCA=False,
            neutral_thrust_axis=False,
            args=self.args
        )
        self.model_neutral = EnergyCorrectionWrapper(
            device=dev,
            in_features_global=num_global_features,
            in_features_gnn=20,
            ckpt_file=ckpt_neutral,
            gnn=True,
            gatr=True,
            pos_regression=self.args.regress_pos,
            pid_channels=len(self.pids_neutral),
            unit_p=self.args.regress_unit_p,
            out_f=1,  # To change to 1 for new models!!!!
            neutral_avg=True,
            neutral_PCA=False,
            neutral_thrust_axis=False,
            predict=self.args.predict,
            args=self.args
        )

    def clustering_and_global_features(self, g, x, y, add_fakes=True):
        time_matching_start = time()
        # Match graphs
        (
            graphs_new, # Contains both fakes and true showers
            true_new, # FOR THE MATCHED SHOWERS
            sum_e, # FOR THE MATCHED + FAKE SHOWERS
            true_pid, # FOR THE MATCHED SHOWERS
            e_true_corr_daughters, # FOR THE MATCHED SHOWERS
            true_coords, # FOR THE MATCHED SHOWERS
            number_of_fakes,
            fakes_idx
        ) = obtain_clustering_for_matched_showers(
            g,
            x,
            y,
            self.main_model.trainer.global_rank,
            use_gt_clusters=self.args.use_gt_clusters,
            add_fakes=add_fakes,
        )
        time_matching_end = time()
        # wandb.log({"time_clustering_matching": time_matching_end - time_matching_start})
        batch_num_nodes = graphs_new.batch_num_nodes()
        batch_idx = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
        batch_idx = torch.tensor(batch_idx).to(self.main_model.device)
        graphs_new.ndata["h"][:, 0:3] = graphs_new.ndata["h"][:, 0:3] / 3300
        # TODO: add global features to each node here
        graphs_sum_features = scatter_add(graphs_new.ndata["h"], batch_idx, dim=0)
        # now multiply graphs_sum_features so the shapes match
        graphs_sum_features = graphs_sum_features[batch_idx]
        # append the new features to "h" (graphs_sum_features)
        shape0 = graphs_new.ndata["h"].shape
        betas = torch.sigmoid(graphs_new.ndata["h"][:, -1])
        graphs_new.ndata["h"] = torch.cat(
            (graphs_new.ndata["h"], graphs_sum_features), dim=1
        )
        assert shape0[1] * 2 == graphs_new.ndata["h"].shape[1]
        # print("Also computing graph-level features")
        graphs_high_level_features = get_post_clustering_features(
            graphs_new, sum_e, is_muons=self.main_model.args.is_muons, add_hit_chis=self.args.add_track_chis
        )
        extra_features = get_extra_features(graphs_new, betas)
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
            pred_pid = torch.ones((graphs_high_level_features.shape[0])).to(
                graphs_new.ndata["h"].device
            ).long()
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
        #    graphs_new.ndata["h"]*weights, batch_idx, dim=0
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
        features_charged_no_nan = graphs_high_level_features[charged_idx]
        features_charged_no_nan[features_charged_no_nan != features_charged_no_nan] = 0
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
            pred_pid,
            features_charged_no_nan,
            number_of_fakes,
            extra_features,
            fakes_idx
        )

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
            pred_pid,
            features_charged_no_nan,
            number_of_fakes,
            extra_features,
            fakes_idx
        ) = self.clustering_and_global_features(g, x, y, add_fakes=self.args.predict)
        charged_energies = self.model_charged.charged_prediction(
            graphs_new, charged_idx, features_charged_no_nan
        )
        neutral_energies, neutral_pxyz_avg = self.model_neutral.neutral_prediction(
            graphs_new, neutral_idx, features_neutral_no_nan
        )
        if self.args.regress_pos:
            if len(self.pids_charged):
                charged_energies, charged_positions, charged_PID_pred, charged_ref_pt_pred = charged_energies # charged_pxyz_pred: we are also storing the xyz of the track, to see the effect of the weirdly fitted tracks on the results
            else:
                charged_energies, charged_positions, _ = charged_energies
            if len(self.pids_neutral):
                neutral_energies, neutral_positions, neutral_PID_pred, neutral_ref_pt_pred = neutral_energies
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
            charged_energies #/ sum_e.flatten()[charged_idx.flatten()]
        )
        pred_energy_corr[neutral_idx.flatten()] = (
            neutral_energies #/ sum_e.flatten()[neutral_idx.flatten()]
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
            pred_ref_pt = torch.ones_like(pred_pos)
            if len(charged_idx):
                pred_ref_pt[charged_idx.flatten()] = charged_ref_pt_pred.to(pred_ref_pt.device)
                pred_pos[charged_idx.flatten()] = charged_positions.float().to(pred_pos.device)
            if len(neutral_idx):
                pred_ref_pt[neutral_idx.flatten()] = neutral_ref_pt_pred.to(neutral_idx.device)
                pred_pos[neutral_idx.flatten()] = neutral_positions.to(neutral_idx.device).float()
            pred_energy_corr = {
                "pred_energy_corr": pred_energy_corr,
                "pred_pos": pred_pos,
                "neutrals_idx": neutral_idx.flatten(),
                "charged_idx": charged_idx.flatten(),
                "pred_ref_pt": pred_ref_pt,
                "extra_features": extra_features,
                "fakes_labels": fakes_idx
            }
            if len(self.pids_charged) or len(self.pids_neutral):
                pred_energy_corr["pred_PID"] = pred_pid
                pred_energy_corr["charged_PID_pred"] = charged_PID_pred
                pred_energy_corr["neutral_PID_pred"] = neutral_PID_pred

        if return_train:
            return (
                x,
                pred_energy_corr,
                true_new,
                sum_e,
                true_pid,
                true_new,
                true_coords,
                number_of_fakes
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
                    number_of_fakes
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
                number_of_fakes
            )
    @staticmethod
    def criterion(ypred, ytrue, step):
        return F.l1_loss(ypred, ytrue)

    def get_loss(self, batch_g, y, result):
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
            num_fakes
        ) = result
        if self.args.regress_pos:
            dic = e_cor
            e_cor, pred_pos, neutral_idx, charged_idx, pred_ref_pt, extra_features = (
                e_cor["pred_energy_corr"],
                e_cor["pred_pos"],
                e_cor["neutrals_idx"],
                e_cor["charged_idx"],
                e_cor["pred_ref_pt"],
                e_cor["extra_features"]
                #e_cor["pred_pos_avg"],
            )
            if len(self.pids_charged):
                charged_PID_pred = dic["charged_PID_pred"]
                charged_PID_true = np.array(pid_true_matched)[dic["charged_idx"].cpu().tolist()]
                # one-hot encoded
                charged_PID_true_onehot = torch.zeros(
                    len(charged_PID_true), len(self.pids_charged)
                )
                mask_charged = torch.ones(len(charged_PID_true))
                if not self.args.PID_4_class:
                    for i in range(len(charged_PID_true)):
                        if charged_PID_true[i] in self.pids_charged:
                            charged_PID_true_onehot[i, self.pids_charged.index(charged_PID_true[i])] = 1
                        else:
                            charged_PID_true_onehot[i, -1] = 1
                else:
                    for i in range(len(charged_PID_true)):
                        true_idx = self.pid_conversion_dict.get(charged_PID_true[i], 3)
                        if true_idx not in self.pids_charged:
                            # Nonsense example - don't train on this one
                            mask_charged[i] = 0
                        else:
                            charged_PID_true_onehot[i, self.pids_charged.index(true_idx)] = 1
                        if charged_PID_true[i] not in self.pid_conversion_dict:
                            print("Unknown PID", charged_PID_true[i])
                charged_PID_true_onehot = charged_PID_true_onehot.clone().to(dic["charged_idx"].device)
            if len(self.pids_neutral):
                neutral_PID_pred = dic["neutral_PID_pred"]
                neutral_PID_true = np.array(pid_true_matched)[neutral_idx.cpu()]
                if type(neutral_PID_true) == np.float64:
                    neutral_PID_true = [neutral_PID_true]
                # One-hot encoded
                #print("NeutralPIDTrue", neutral_PID_true, "PidsNeutral", self.pids_neutral, "NeutralIdx", neutral_idx)
                neutral_PID_true_onehot = torch.zeros(
                    len(neutral_PID_true), len(self.pids_neutral)
                )
                mask_neutral = torch.ones(len(neutral_PID_true))
                if not self.args.PID_4_class:
                    for i in range(len(neutral_PID_true)):
                        if neutral_PID_true[i] in self.pids_neutral:
                            neutral_PID_true_onehot[i, self.pids_neutral.index(neutral_PID_true[i])] = 1
                        else:
                            neutral_PID_true_onehot[i, -1] = 1
                else:
                    for i in range(len(neutral_PID_true)):
                        true_idx = self.pid_conversion_dict.get(neutral_PID_true[i], 3)
                        if true_idx not in self.pids_neutral:
                            mask_neutral[i] = 0
                        else:
                            neutral_PID_true_onehot[i, self.pids_neutral.index(true_idx)] = 1
                        if neutral_PID_true[i] not in self.pid_conversion_dict:
                            print("Unknown PID", neutral_PID_true[i])
                neutral_PID_true_onehot = neutral_PID_true_onehot.to(neutral_idx.device)
        if self.args.correction:
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
                    num_fakes
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
                    num_fakes,
                ) = result
            if self.args.regress_pos:
                if len(self.pids_charged):
                    charged_PID_pred = e_cor["charged_PID_pred"]
                    # charged_PID_pred =  np.array(self.pids_charged + [0])[np.argmax(charged_PID_pred.cpu(), axis=1)]
                    charged_idx = e_cor["charged_idx"]
                    # charged_PID_true = np.array(pid_true_matched)[charged_idx.cpu()]
                    # if self.args.PID_4_class:
                    #    charged_PID_true = np.array([self.pid_conversion_dict.get(x, 3) for x in charged_PID_true])
                    # pid_list[charged_idx] = charged_PID_pred
                if len(self.pids_neutral):
                    neutral_idx = e_cor["neutrals_idx"]
                    # neutral_PID_pred = e_cor["neutral_PID_pred"]
                    # neutral_PID_pred = np.array(self.pids_neutral + [0])[np.argmax(neutral_PID_pred.cpu(), axis=1)]
                    # neutral_PID_true = np.array(pid_true_matched)[neutral_idx.cpu()]
                    # if self.args.PID_4_class:
                    #    neutral_PID_true = np.array([self.pid_conversion_dict.get(x, 3) for x in neutral_PID_true])
                pred_pid = e_cor["pred_PID"]
                e_cor, pred_pos, pred_ref_pt, extra_features = e_cor["pred_energy_corr"], e_cor["pred_pos"], e_cor[
                    "pred_ref_pt"], e_cor["extra_features"]
                # pid_list = np.zeros_like(e_cor)
            else:
                pred_pos = None
                pred_ref_pt = None
                e_cor = None
                pred_pid = None
            loss_ll = 0
            e_cor1 = torch.ones_like(model_output[:, 0].view(-1, 1))
        else:
            model_output, e_cor1, loss_ll = result[0], result[1], result[2]
            loss_ll = 0
            e_cor1 = torch.ones_like(model_output[:, 0].view(-1, 1))
            e_cor = e_cor1
            pred_pos = None
            pred_pid = None
            pred_ref_pt = None
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
                    num_fakes
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
                    num_fakes,
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
                e_cor, pred_pos, pred_ref_pt, extra_features = e_cor["pred_energy_corr"], e_cor["pred_pos"], e_cor["pred_ref_pt"], e_cor["extra_features"]
                #pid_list = np.zeros_like(e_cor)
            else:
                pred_pos = None
                pred_ref_pt = None
                e_cor = None
                pred_pid=None
            loss_ll = 0
            e_cor1 = torch.ones_like(model_output[:, 0].view(-1, 1))
        step = self.main_model.trainer.global_step
        loss_EC = self.criterion(e_cor, e_true_corr_daughters, step)
        if self.args.regress_pos:
            true_pos = torch.tensor(part_coords_matched).to(pred_pos.device)
            if self.args.regress_unit_p:
                true_pos = (true_pos / torch.norm(true_pos, dim=1).view(-1, 1)).clone()
                pred_pos = (pred_pos / torch.norm(pred_pos, dim=1).view(-1, 1)).clone()
            # loss_pos = torch.nn.L1Loss()(pred_pos, true_pos)
            loss_pos = 1 - ((torch.nn.CosineSimilarity()(pred_pos, true_pos)).mean())
            charged_idx = np.array(sorted(list(set(range(len(e_cor))) - set(neutral_idx))))
            # loss_pos_charged = torch.nn.L1Loss()(pred_pos[charged_idx], true_pos[charged_idx])
            # loss_pos_neutrals = torch.nn.L1Loss()(pred_pos[neutral_idx], true_pos[neutral_idx])
            loss_EC_neutrals = torch.nn.L1Loss()(
                e_cor[neutral_idx], e_true[neutral_idx]
            )
            filt_neutrons = (e_true[neutral_idx] < 5).cpu() & (torch.tensor(pid_true_matched)[neutral_idx.cpu()] == 2112)
            loss_EC_neutrons = torch.nn.L1Loss()(
                e_cor[neutral_idx][filt_neutrons].detach().cpu(), e_true[neutral_idx][filt_neutrons].detach().cpu()
            )
            filt_KL = (e_true[neutral_idx] < 5).cpu() & (torch.tensor(pid_true_matched)[neutral_idx.cpu()] == 130)
            loss_EC_KL = torch.nn.L1Loss()(
                e_cor[neutral_idx][filt_KL].detach().cpu(), e_true[neutral_idx][filt_KL].detach().cpu()
            )
            # charged idx is e_cor indices minus neutral idx
            charged_idx = np.array(sorted(list(set(range(len(e_cor))) - set(neutral_idx))))
            loss_pos_neutrals = torch.nn.L1Loss()(
                pred_pos[neutral_idx], true_pos[neutral_idx]
            )
            loss_charged = torch.nn.L1Loss()(
                pred_pos[charged_idx], true_pos[charged_idx]
            )  # just for logging
            # wandb.log(
            #     {"loss_pxyz": loss_pos, "loss_pxyz_neutrals": loss_pos_neutrals}
            # )
            wandb.log({
                "loss_EC_neutrals": loss_EC_neutrals, "loss_EC_charged": loss_charged,
                "loss_p_neutrals": loss_pos_neutrals, "loss_p_charged": loss_charged,
                "loss_EC_KL": loss_EC_KL, "loss_EC_neutrons": loss_EC_neutrons
            })
            # print("Loss pxyz neutrals", loss_pos_neutrals)
            if len(self.pids_charged):
                if self.args.balance_pid_classes and charged_PID_true_onehot.shape[0] > 20:
                    # Batch size must be big enough
                    weights = charged_PID_true_onehot.sum(dim=0)
                    weights[weights == 0] = 1  # to avoid issues
                    weights = 1 / weights  # maybe choose something else?
                    print("Charged class weights:", weights)
                else:
                    weights = torch.ones(len(self.pids_charged)).to(charged_PID_pred.device)
                if len(charged_PID_pred):
                    mask_charged = mask_charged.bool()
                    loss_charged_pid = torch.nn.CrossEntropyLoss(weight=weights)(
                        charged_PID_pred[mask_charged], charged_PID_true_onehot[mask_charged]
                    )
                else:
                    loss_charged_pid = 0
                wandb.log({"loss_charged_pid": loss_charged_pid})
            if len(self.pids_neutral):
                if self.args.balance_pid_classes and neutral_PID_true_onehot.shape[0] > 20:
                    # Batch size must be big enough
                    weights = neutral_PID_true_onehot.sum(dim=0)
                    weights[weights == 0] = 1  # To avoid issues
                    weights = 1 / weights  # Maybe choose something else?
                    print("Neutral class weights:", weights)
                else:
                    weights = torch.ones(len(self.pids_neutral)).to(charged_PID_pred.device)
                if len(neutral_PID_pred):
                    mask_neutral = mask_neutral.bool()
                    loss_neutral_pid = torch.nn.CrossEntropyLoss(weight=weights)(
                        neutral_PID_pred[mask_neutral], neutral_PID_true_onehot[mask_neutral]
                    )
                    print("Neutral PID pred:\n", neutral_PID_pred[mask_neutral][:4])
                    print("Neutral PID true:\n", neutral_PID_true_onehot[mask_neutral][:4])
                else:
                    loss_neutral_pid = 0
                wandb.log({"loss_neutral_pid": loss_neutral_pid})
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
        return loss_EC, loss_pos, loss_neutral_pid, loss_charged_pid

    def get_validation_step_outputs(self, batch_g, y, result):
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
                num_fakes
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
                num_fakes,
            ) = result
        if self.args.regress_pos:
            if len(self.pids_charged):
                charged_PID_pred = e_cor["charged_PID_pred"]
                charged_idx = e_cor["charged_idx"]
            if len(self.pids_neutral):
                neutral_idx = e_cor["neutrals_idx"]
            pred_pid = e_cor["pred_PID"]
            e_cor, pred_pos, pred_ref_pt, extra_features, fakes_labels = e_cor["pred_energy_corr"], e_cor["pred_pos"], e_cor[
                "pred_ref_pt"], e_cor["extra_features"], e_cor["fakes_labels"]
        else:
            pred_pos = None
            pred_ref_pt = None
            e_cor = None
            pred_pid = None
            extra_features = None
            fakes_labels = None
        return e_cor, pred_pos, pred_ref_pt, pred_pid, num_fakes, extra_features, fakes_labels
