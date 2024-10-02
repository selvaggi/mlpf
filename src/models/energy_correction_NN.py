"""
    PID predict energy correction
    The model taken from notebooks/13_NNs.py
    At first the model is fixed and the weights are loaded from earlier training
"""
from xformers.ops.fmha import BlockDiagonalMask
from gatr.interface import (
    embed_point,
    extract_point,
    extract_translation,
    embed_scalar,
    extract_scalar
)
import numpy as np

from gatr import GATr, SelfAttentionConfig, MLPConfig
import pickle

from copy import deepcopy
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.models.gravnet_3_L import GravnetModel
from src.models.thrust_axis import Thrust, hits_xyz_to_momenta, LR, weighted_least_squares_line
from torch_geometric.nn.models import GAT, GraphSAGE
from torch_scatter import scatter_mean, scatter_sum
from gatr import GATr
import dgl

class Net(nn.Module):
    def __init__(self, in_features=13, out_features=1, return_raw=False):
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


class ECNetWrapper(torch.nn.Module):
    def __init__(self, ckpt_file, device, in_features=13, pid_predict_channels=4):
        super(ECNetWrapper, self).__init__()
        self.model = Net(in_features=in_features, out_features=1 + pid_predict_channels)
        # load weights from pickle
        if ckpt_file is not None:
            self.model.model = pickle.load(open(ckpt_file, "rb"))
            print("Loaded energy correction model weights from", ckpt_file)
        # print("Temporarily not loading the model weights")
        self.model.to(device)

    def predict(self, x):
        # if isinstance(pred, tuple):
        #    return (pred[0].flatten(), pred[1])
        # return self.model(x).flatten(), None
        return self.model(x)


class ECNetWrapperGNN(torch.nn.Module):
    # use the GNN+NN model for energy correction
    def __init__(self, device, in_features=13, arch="vanilla"):
        super(ECNetWrapperGNN, self).__init__()
        gnn_features = 64
        self.model = Net(in_features=gnn_features, out_features=1)
        # use a GAT
        if arch == "GAT":
            self.gnn = GAT(
                in_features,
                out_channels=gnn_features,
                heads=4,
                concat=True,
                hidden_channels=64,
                num_layers=3,
            )
        elif arch == "vanilla":
            self.gnn = GraphSAGE(
                in_features, gnn_features, hidden_channels=64, num_layers=3
            )
        # elif arch == "GATr":
        #    self.gnn = GATr(in_features,
        else:
            raise NotImplementedError
        self.model.to(device)

    def predict(self, x_global_features, graphs_new, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        assert explain is False, "Explain not implemented for this GNN"
        batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
        batch_idx = []
        batch_bounds = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
        node_global_features = x_global_features[batch_idx]
        x = torch.cat([graphs_new.ndata["h"], node_global_features], dim=1)
        edge_index = torch.stack(graphs_new.edges())
        gnn_output = self.gnn(x, edge_index)
        gnn_output = scatter_mean(gnn_output, batch_idx, dim=0)
        return self.model(gnn_output).flatten()

import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


class ECNetWrapperGNNGlobalFeaturesSeparate(torch.nn.Module):
    # use the GNN+NN model for energy correction
    # This one concatenates GNN features to the global features
    def __init__(
        self,
        device,
        in_features_global=13,
        in_features_gnn=13,
        out_features_gnn=32,
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
        simple_p_GNN=False
    ):
        super(ECNetWrapperGNNGlobalFeaturesSeparate, self).__init__()
        self.charged = charged
        self.simple_p_GNN = simple_p_GNN
        self.neutral_avg = neutral_avg
        self.pos_regression = pos_regression
        self.unit_p = unit_p
        self.neutral_PCA = neutral_PCA
        self.neutral_thrust_axis = neutral_thrust_axis
        self.use_gatr = gatr
        print("pos_regression", self.pos_regression)
        # if pos_regression:
        #     out_f += 3
        self.ignore_global_features_for_p = ignore_global_features_for_p
        if self.charged:
            self.ignore_global_features_for_p = False
        if self.ignore_global_features_for_p:
            if not self.simple_p_GNN:
                self.gatr_p = GATr(
                    in_mv_channels=1,
                    out_mv_channels=1,
                    hidden_mv_channels=4,
                    in_s_channels=3,
                    out_s_channels=None,
                    hidden_s_channels=4,
                    num_blocks=3,
                    attention=SelfAttentionConfig(),  # Use default parameters for attention...
                    mlp=MLPConfig(),  # Use default parameters for MLP
                )
            else:
                self.gnn_p = GAT(
                    in_features_gnn,
                    out_channels=16,
                    heads=4,
                    concat=True,
                    hidden_channels=64,
                    num_layers=3)
            #if not self.neutral_avg:
            self.model_p = Net(16, 3, return_raw=True)
        self.model = Net(
            in_features=out_features_gnn + in_features_global, out_features=out_f
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
            self.PID_head = nn.Linear(out_features_gnn + in_features_global, pid_channels)   # Additional head for PID classification
            self.PID_head.to(device)
        if ckpt_file is not None and ckpt_file != "":
            # self.model.model = pickle.load(open(ckpt_file, 'rb'))
            with open(ckpt_file.strip(), "rb") as f:
                self.model.model = CPU_Unpickler(f).load()
            print("Loaded energy correction model weights from", ckpt_file)
        else:
            print("Not loading energy correction model weights")
        self.model.to(device)
        self.PickPAtDCA = PickPAtDCA()
        self.AvgHits = AverageHitsP()
        self.NeutralPCA = NeutralPCA()
        self.ThrustAxis = ThrustAxis()

    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        use_full_mv = True  # Whether to use the full multivector to regress E and p or just sth else
        if graphs_new is not None and self.gnn is not None:
            batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
            batch_idx = []
            batch_bounds = []
            for i, n in enumerate(batch_num_nodes):
                batch_idx.extend([i] * n)
                batch_bounds.append(n)
            batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
            node_global_features = x_global_features
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
            # energy = torch.clamp(extract_scalar(embedded_outputs), min=0).flatten()
            # if self.pos_regression:
            #    return energy, p_vectors
            # return energy
            if use_full_mv:
                padding = torch.randn(x_global_features.shape[0], 16).to(
                    p_vectors_per_batch.device
                )
                model_x = torch.cat(
                    [x_global_features, embedded_outputs_per_batch, padding], dim=1
                ).to(self.model.model[0].weight.device)
                if self.ignore_global_features_for_p:
                    if self.simple_p_GNN:
                        output = self.gnn_p(x, edge_index)
                        output = scatter_mean(output, batch_idx, dim=0)
                        res_pxyz = self.model_p(output)
                    else:
                        embedded_outputs_p, _ = self.gatr_p(
                            embedded_inputs, scalars=extra_scalars, attention_mask=mask
                        )
                        #p_vectors_p = extract_translation(embedded_outputs_p)
                        embedded_outputs_per_batch_p = scatter_sum(
                            embedded_outputs_p[:, 0, :], batch_idx, dim=0
                        )
                        #res_pxyz = scatter_sum(
                        #    p_vectors_p[:, 0, :], batch_idx, dim=0
                        #)
                        res_pxyz = self.model_p(embedded_outputs_per_batch_p)
            else:
                padding = torch.randn(x_global_features.shape[0], 32).to(
                    p_vectors_per_batch.device
                )
                model_x = torch.cat([x_global_features, padding], dim=1).to(
                    self.model.model[0].weight.device
                )
        else:
            # not using GATr features
            gnn_output = torch.randn(x_global_features.shape[0], 32).to(
                x_global_features.device
            )
        if not self.use_gatr:
            model_x = torch.cat([x_global_features, gnn_output], dim=1).to(
                self.model.model[0].weight.device
            )
            ''' if explain:
                assert not self.use_gatr
                print("explain")
                # take a selection of 10% or 50 samples to get typical feature values
                print(model_x.shape)
                n_samples = min(50, int(0.2 * model_x.shape[0]))
                model_exp = deepcopy(self.model)
                model_exp.to("cpu")
                model_exp.explainer_mode = True
                with torch.no_grad():
                    for parameter in model_exp.model.parameters():
                        parameter.requires_grad = False
                    explainer = shap.KernelExplainer(
                        model_exp, model_x[:n_samples].detach().cpu().numpy()
                    )
                    shap_vals = explainer.shap_values(
                        model_x.detach().cpu().numpy(), nsamples=200
            )
            return self.model(model_x).flatten(), shap_vals, model_x.detach().cpu()'''
        res = self.model(model_x)
        if self.pid_channels > 1:
            pid_pred = self.PID_head(model_x)
        else:
            pid_pred = None
        if self.pos_regression:
            if self.charged:
                p_tracks, pos, ref_pt_pred = self.PickPAtDCA.predict(x_global_features, graphs_new)
                if self.unit_p:
                    pos = (pos / torch.norm(pos, dim=1).unsqueeze(1)).clone()
                return torch.clamp(res.flatten(), min=0, max=None), pos, pid_pred, ref_pt_pred
            else:
                E_pred, p_pred = res[0], res[1]
                E_pred = torch.clamp(E_pred, min=0, max=None)
                _, _, ref_pt_pred = self.AvgHits.predict(x_global_features, graphs_new)
                if self.neutral_avg:
                    _, p_pred, ref_pt_pred = self.AvgHits.predict(x_global_features, graphs_new)
                    #E_pred = x_global_features[:, 6] # For the photons, just take the energy
                elif self.neutral_PCA:
                    _, p_pred, ref_pt_pred = self.NeutralPCA.predict(x_global_features, graphs_new)
                elif self.neutral_thrust_axis:
                    _, p_pred, ref_pt_pred = self.ThrustAxis.predict(x_global_features, graphs_new)
                else:
                    if self.ignore_global_features_for_p:
                        p_pred = res_pxyz  # Temporarily discard the pxyz output of the E prediction head
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
            return torch.clamp(res.flatten(), min=0, max=None)
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
        # if "pos_pxpypz_at_vertex" in graphs_new.ndata.keys():
        #    key = "pos_pxpypz_at_vertex"
        # else:
        #    key = "pos_pxpypz"
        # p_direction = scatter_mean(
        #     graphs_new.ndata["pos_pxpypz_at_vertex"][filt], batch_idx[filt], dim=0
        # )
        # take the min chi squared track if there are multiple
        p_direction = pick_lowest_chi_squared(
            graphs_new.ndata["pos_pxpypz_at_vertex"][filt],
            graphs_new.ndata["chi_squared_tracks"][filt],
            batch_idx[filt],
        )
        p_tracks = torch.norm(p_direction, dim=1)
        p_direction = p_direction  # / torch.norm(p_direction, dim=1).unsqueeze(1)
        return p_tracks, p_direction, torch.zeros_like(p_direction) # reference point
        # return p_tracks

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

def pick_lowest_chi_squared(pxpypz, chi_s, batch_idx):
    unique_batch = torch.unique(batch_idx)
    p_direction = []
    for i in range(0, len(unique_batch)):
        mask = batch_idx == unique_batch[i]
        if torch.sum(mask) > 1:
            chis = chi_s[mask]
            ind_min = torch.argmin(chis)
            p_direction.append(pxpypz[mask][ind_min].view(-1, 3))
        else:
            p_direction.append(pxpypz[mask].view(-1, 3))
    return torch.concat(p_direction, dim=0)
