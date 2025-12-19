"""
    PID predict energy correction
    The model taken from notebooks/train_energy_correction_head.py
    At first the model is fixed and the weights are loaded from earlier training
"""
import wandb
from xformers.ops.fmha import BlockDiagonalMask
from gatr.interface  import (
    embed_point,
    embed_scalar,
)
from src.layers.utils_training import obtain_clustering_for_matched_showers
from torch_scatter import scatter_add, scatter_mean
from src.utils.post_clustering_features import (
    get_post_clustering_features,
    get_extra_features,
    calculate_eta,
    calculate_phi,
)
from time import time
import numpy as np
from gatr import GATr, SelfAttentionConfig, MLPConfig
import pickle
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.utils.pid_conversion import pid_conversion_dict
# from torch_geometric.nn.models import GAT, GraphSAGE
from torch_scatter import scatter_mean, scatter_sum
from gatr import GATr
import dgl
from src.layers.tools_for_regression import ECNetWrapperAvg,PickPAtDCA, AverageHitsP,NeutralPCA,ThrustAxis
from src.models.GATr.E_correction_module import Net, CPU_Unpickler
from src.layers.regression.loss_regression import loss_position, loss_score_func, obtain_PID_charged, obtain_PID_neutral
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
        neutral_thrust_axis=False,
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
                    in_s_channels=2,
                    out_s_channels=None,
                    hidden_s_channels=4,
                    num_blocks=3,
                    attention=SelfAttentionConfig(),  # Use default parameters for attention
                    mlp=MLPConfig(),  # Use default parameters for MLP
                )
                self.gnn = "gatr"
                if self.separate_pid_gatr and not self.charged:
                    print("Separate PID GATr")
                    self.gatr_pid = GATr(
                        in_mv_channels=1,
                        out_mv_channels=1,
                        hidden_mv_channels=4,
                        in_s_channels=2,
                        out_s_channels=None,
                        hidden_s_channels=4,
                        num_blocks=3,
                        attention=SelfAttentionConfig(),  # Use default parameters for attention
                        mlp=MLPConfig(),  # Use default parameters for MLP
                    )
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
        self.fake_score_network = False 
        if self.fake_score_network:
            n_layers = 3
            self.fakes_head = nn.ModuleList()
            self.fakes_head.append(nn.Linear(out_features_gnn + in_features_global, 64))
            for i in range(n_layers - 1):
                self.fakes_head.append(nn.Linear(64, 64))
                self.fakes_head.append(nn.ReLU())
            self.fakes_head.append(nn.Linear(64, 1))
            self.fakes_head = nn.Sequential(*self.fakes_head)
            self.fakes_head.to(device)
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
        self.AvgHits = AverageHitsP(ecal_only=True)
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
            hit_type = graphs_new.ndata["h"][:, 4:8].argmax(dim=1)
            betas = graphs_new.ndata["h"][:, 10]
            p = graphs_new.ndata["h"][:, 9]
            e = graphs_new.ndata["h"][:, 8]
            embedded_inputs = embed_point(hits_points) + embed_scalar(
                hit_type.view(-1, 1)
            )
            extra_scalars = torch.cat(
                [ p.unsqueeze(1), e.unsqueeze(1)], dim=1
            )
            mask = self.build_attention_mask(graphs_new)
            embedded_inputs = embedded_inputs.unsqueeze(-2)
            embedded_outputs, _ = self.gatr(
                embedded_inputs, scalars=extra_scalars, attention_mask=mask
            )
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
               
                embedded_outputs_per_batch1 = scatter_sum(
                    embedded_outputs[:, 0, :], batch_idx, dim=0
                )
                model_x_pid = torch.cat(
                    [x_global_features, embedded_outputs_per_batch1], dim=1
                )
            else:
                model_x_pid = model_x

        if not self.charged:
            # Predict energy for neutrals using the neural network
            res = self.model(model_x)
        if self.pid_channels > 1:
            pid_pred = self.PID_head(model_x_pid)
        else:
            pid_pred = None
        if self.fake_score_network:
            score_pred = self.fakes_head(model_x_pid)
        else:
            score_pred = None
        if self.pos_regression:
            if self.charged:
                p_tracks, pos, ref_pt_pred = self.PickPAtDCA.predict(x_global_features, graphs_new)
                
                E = torch.norm(pos, dim=1)
                if self.unit_p:
                    pos = (pos / torch.norm(pos, dim=1).unsqueeze(1)).clone()
                return E, pos, pid_pred, ref_pt_pred, score_pred
            else:
                E_pred = res[:, 0]
                if torch.sum(torch.isnan(E_pred))>0:
                    print("FOUND NAANANANNANANNA!!!!!!")
                    print("nans in betas", torch.sum(torch.isnan(betas)))
                    print("nans in x_global_features", torch.sum(torch.isnan(x_global_features)))   
                    print(x_global_features) 
                # E_pred = torch.clamp(E_pred, min=0, max=None)
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
                return E_pred, p_pred, pid_pred, ref_pt_pred, score_pred
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



class EnergyCorrection():
    def __init__(self, main_model):
        #super(EnergyCorrection, self).__init__()
        self.args = main_model.args
        self.get_PID_categories(main_model)
        self.get_energy_correction(main_model)
        self.pid_conversion_dict = pid_conversion_dict
        self.main_model = main_model
        self.fake_score_network = False
        self.global_step = 0
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
        pids_charged = [0, 1, 2, 3]  # electron, CH, NH, gamma
        pids_neutral = [0, 1, 2, 3]  # electron, CH, NH, gamma
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
            truth_tracks=self.args.truth_tracking
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
        # normalizations1 = torch.ones_like(weights)
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
        num_hits = graphs_high_level_features[:, 2]
        charged_idx = torch.where((num_tracks >= 1))[0]
        neutral_idx = torch.where((num_tracks < 1))[0]
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
                charged_energies, charged_positions, charged_PID_pred, charged_ref_pt_pred, charged_score_pred= charged_energies # charged_pxyz_pred: we are also storing the xyz of the track, to see the effect of the weirdly fitted tracks on the results
            else:
                charged_energies, charged_positions, _ = charged_energies
            if len(self.pids_neutral):
                neutral_energies, neutral_positions, neutral_PID_pred, neutral_ref_pt_pred, neutral_score_pred= neutral_energies
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
        if self.fake_score_network:
            score_object = pred_pid.clone()
            if len(charged_idx):
                score_object[charged_idx.flatten()] = charged_score_pred
            if len(neutral_idx):
                score_object[neutral_idx.flatten()] = neutral_score_pred
        

        if len(self.pids_charged):
            if len(charged_idx):
                charged_PID_pred1 = np.array(self.pids_charged)[np.argmax(charged_PID_pred.cpu().detach(), axis=1)]  #0,1,2
            else:
                charged_PID_pred1 = []
            pred_pid[charged_idx.flatten()] = torch.tensor(charged_PID_pred1).long().to(charged_idx.device)

        if len(self.pids_neutral):
            if len(neutral_idx):
                neutral_PID_pred1 = np.array(self.pids_neutral)[np.argmax(neutral_PID_pred.cpu().detach(), axis=1)] #0,1
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
                if self.fake_score_network:
                    pred_energy_corr["score_object"]=score_object
                    pred_energy_corr["fakes_idx"] = fakes_idx
    
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

    def get_loss(self, batch_g, y, result, stats, fixed):
        (
            model_output,
            dic_e_cor,
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
        e_cor = dic_e_cor["pred_energy_corr"]

        ############ loss EC of neutral only  ###########
        
        mask_neutral_for_loss = correct_mask_neutral(torch.tensor(pid_true_matched), dic_e_cor["neutrals_idx"])

        e_true_neutrals = e_true[mask_neutral_for_loss]
        e_pred_neutrals = e_cor[mask_neutral_for_loss]
        e_reco_neutrals = e_sum_hits[mask_neutral_for_loss]
        in_distribution = (torch.abs(e_true_neutrals-e_reco_neutrals)/e_true_neutrals)<0.6
        # print("distribution", torch.abs(e_true_neutrals-e_reco_neutrals)/e_true_neutrals)
        # loss_EC_neutrals = torch.nn.L1Loss()(
        #     e_pred_neutrals[in_distribution], e_true_neutrals[in_distribution]
        # )
        # print(e_pred_neutrals.shape)
        ypred = e_pred_neutrals[in_distribution]
        ybatch = e_true_neutrals[in_distribution]
        if len(ypred)>0:
            pid_neutrals = torch.tensor(pid_true_matched)[mask_neutral_for_loss.cpu()].to(ypred.device)
            # print(pid_neutrals)
            # print(torch.sum(in_distribution))
            # print(in_distribution)
            # print(pid_neutrals[in_distribution])
            loss_EC_neutrals, stats = criterion_E_cor(ypred.flatten(), ybatch.flatten(), self.global_step, torch.abs(pid_neutrals[in_distribution]), stats, frozen=fixed)
        else:
            loss_EC_neutrals = 0
        filt_neutrons = (e_true[dic_e_cor["neutrals_idx"]] < 5).cpu() & (torch.tensor(pid_true_matched)[dic_e_cor["neutrals_idx"].cpu()] == 2112)
        loss_EC_neutrons = torch.nn.L1Loss()(
            torch.exp(e_cor[dic_e_cor["neutrals_idx"]][filt_neutrons].detach().cpu()), e_true[dic_e_cor["neutrals_idx"]][filt_neutrons].detach().cpu()
        )
        filt_KL = (e_true[dic_e_cor["neutrals_idx"]] < 5).cpu() & (torch.tensor(pid_true_matched)[dic_e_cor["neutrals_idx"].cpu()] == 130)
        loss_EC_KL = torch.nn.L1Loss()(
            e_cor[dic_e_cor["neutrals_idx"]][filt_KL].detach().cpu(), e_true[dic_e_cor["neutrals_idx"]][filt_KL].detach().cpu()
        )
        loss_pos, loss_pos_neutrals, loss_pos_charged = loss_position(part_coords_matched, dic_e_cor["pred_pos"], self.args, dic_e_cor["neutrals_idx"], dic_e_cor["charged_idx"])
    
        wandb.log({
            "loss_EC_neutrals": loss_EC_neutrals, 
            "loss_p_neutrals": loss_pos_neutrals, "loss_p_charged": loss_pos_charged,
            "loss_EC_KL": loss_EC_KL, "loss_EC_neutrons": loss_EC_neutrons
        })

        ########### loss PID ###########
        # correct assignation of PIDs without track and go from PID montecarlo number to int 
        if len(self.pids_charged):
            charged_PID_pred, charged_PID_true_onehot, mask_charged = obtain_PID_charged(dic_e_cor,pid_true_matched, self.pids_charged, self.args, self.pid_conversion_dict)
            
        if len(self.pids_neutral):
            neutral_PID_pred, neutral_PID_true_onehot, mask_neutral = obtain_PID_neutral(dic_e_cor,pid_true_matched, self.pids_neutral, self.args, self.pid_conversion_dict)
        
        if len(self.pids_charged):
            loss_charged_pid,acc_charged, stats= pid_loss_weighted(charged_PID_pred, charged_PID_true_onehot,e_true[dic_e_cor["charged_idx"]], mask_charged, stats, fixed, "charged")        
            wandb.log({"loss_charged_pid": loss_charged_pid})
       
        if len(self.pids_neutral):
            loss_neutral_pid,acc_neutral, stats = pid_loss_weighted(neutral_PID_pred, neutral_PID_true_onehot,e_true, mask_neutral, stats, fixed, "neutral")
            wandb.log({"loss_neutral_pid": loss_neutral_pid})
        ########### loss score ###########
        if self.fake_score_network:
            loss_score = loss_score_func(dic_e_cor)
        else:
            loss_score = 0
        return loss_EC_neutrals, loss_pos, loss_neutral_pid, loss_charged_pid, loss_score, stats

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
                charged_idx = e_cor["charged_idx"]
            if len(self.pids_neutral):
                neutral_idx = e_cor["neutrals_idx"]
            pred_pid = e_cor["pred_PID"]
            e_cor, pred_pos, pred_ref_pt, extra_features, fakes_labels, charged_PID_pred, neutral_PID_pred = e_cor["pred_energy_corr"], e_cor["pred_pos"], e_cor[
                "pred_ref_pt"], e_cor["extra_features"], e_cor["fakes_labels"], e_cor["charged_PID_pred"], e_cor["neutral_PID_pred"]
            max_len = max(len(self.pids_charged), len(self.pids_neutral))
            if self.args.restrict_PID_charge:
                PID_logits = torch.zeros(len(e_cor), len(self.pids_charged)+ len(self.pids_neutral)).float()
                PID_logits = PID_logits.clone()
                PID_logits[charged_idx.cpu(),0] = charged_PID_pred.detach().cpu()[:,0]
                PID_logits[charged_idx.cpu(),1] = charged_PID_pred.detach().cpu()[:,1]
                PID_logits[charged_idx.cpu(),4] = charged_PID_pred.detach().cpu()[:,2]
                PID_logits[neutral_idx.cpu(),2] = neutral_PID_pred.detach().cpu()[:,0]
                PID_logits[neutral_idx.cpu(),3] = neutral_PID_pred.detach().cpu()[:,1]
            else:
                PID_logits = torch.zeros(len(e_cor), max_len).float()
                PID_logits[charged_idx.cpu()] = charged_PID_pred.detach().cpu()
                PID_logits[neutral_idx.cpu()] = neutral_PID_pred.detach().cpu()

            extra_features = extra_features.detach().cpu()
            extra_features = torch.cat((extra_features, PID_logits), dim=1).numpy()

        else:
            pred_pos = None
            pred_ref_pt = None
            e_cor = None
            pred_pid = None
            extra_features = None
            fakes_labels = None
        
        return e_cor, pred_pos, pred_ref_pt, pred_pid, num_fakes, extra_features, fakes_labels





def criterion_E_cor(ypred, ytrue, step, pid_neutrals, stats, frozen=False):
    # count occurrences of each PID
    # Initialize stats container if first call
    # ==============================================================
    # 1) Update stats only if not frozen
    # ==============================================================   
    if len(ypred)>0:
        pid_neutrals = pid_neutrals.reshape(-1)
        if not frozen:
            unique_pids, counts = pid_neutrals.unique(return_counts=True)
            for pid, cnt in zip(unique_pids.tolist(), counts.tolist()):
                stats["counts"][pid] = stats["counts"].get(pid, 0) + cnt
        # ==============================================================
        # 2) Convert stats to tensors for computing weights
        # ==============================================================    
        all_pids = list(stats["counts"].keys())
        all_counts = torch.tensor([stats["counts"][p] for p in all_pids], dtype=torch.float)

        freq = all_counts / all_counts.sum()
        raw_weights = 1.0 / freq                      # rarity → higher weight
        weights = raw_weights / raw_weights.mean()    # optional normalization
        # build weight tensor for each sample
        pid_weight_map = {str(int(pid)): w for pid, w in zip(all_pids, weights)}
        
        w_tensor = torch.tensor([pid_weight_map[str(int(p))] for p in pid_neutrals], 
                            device=ypred.device)
        # w_tensor = w_tensor/torch.sum(w_tensor)*len(w_tensor)

        w_tensor = w_tensor.to(ypred.device)
        mask_nans = torch.isnan(w_tensor)
        w_tensor[mask_nans] =0 
        # delta=0.02
        # log_pred = torch.log(ypred.float().clamp_min(1e-6) )
        # log_true = torch.log(ytrue.float().clamp_min(1e-6) )
        # return torch.mean(F.huber_loss(log_pred, log_true, delta=delta, reduction='none')*w_tensor), stats
        return torch.mean(F.l1_loss(ypred, torch.log(ytrue+1e-6), reduction='none')*w_tensor), stats
        # if step < 1000:
        #     return F.l1_loss(ypred, ytrue, weight=w_tensor)
        # else:
        #     # cut the top 5 % of losses by setting them to 0
        #     #losses = F.l1_loss(ypred, ytrue, reduction='none') #+ F.l1_loss(ypred, ytrue, reduction = 'none') / ytrue.abs()
        #     losses = F.l1_loss(ypred, ytrue, reduction = 'none', weight=w_tensor) / ytrue.abs()
        #     if len(losses.shape) > 0:
        #         if int(losses.size(0) * 0.05) > 1:
        #             top_percentile = torch.kthvalue(losses, int(losses.size(0) * 0.95)).values
        #             mask = (losses > top_percentile)
        #             losses[mask] = 0.0
        #     return losses.mean()
    else:
        return 0, stats


def pid_loss_weighted(neutral_PID_pred, neutral_PID_true_onehot,e_true, mask_neutral, stats, frozen=False, name=""):
    if len(neutral_PID_pred):
        """CrossEntropyLoss with PID class balancing based on accumulated stats."""
        # if "counts_pid" not in stats:
        #     stats["counts_pid"+name] = {}
        # Must have enough events
        mask_neutral = mask_neutral.bool()
        # if neutral_PID_true_onehot.shape[0] <= 20:
        #     return torch.nn.CrossEntropyLoss()(
        #         neutral_PID_pred[mask_neutral],
        #         neutral_PID_true_onehot[mask_neutral]
        #     )


        # Update statistics unless frozen
        if not frozen:
            true_labels = neutral_PID_true_onehot.argmax(dim=1)
            for c in true_labels.tolist():
                stats["counts_pid_"+name][c] = stats["counts_pid_"+name].get(c, 0) + 1

        # Build global weight tensor
        num_classes = neutral_PID_true_onehot.shape[1]
        counts = torch.tensor([stats["counts_pid_"+name].get(i,1) for i in range(num_classes)],
                            dtype=torch.float, device=neutral_PID_pred.device)
        counts[counts==0]=1
        weights = 1.0 / counts
        weights = weights / weights.mean()          # optional normalization
        pid_pred = neutral_PID_pred[mask_neutral]
        pid_true = neutral_PID_true_onehot[mask_neutral]

        if name =="charged":
            e_true_ = e_true[mask_neutral]
            mask_muons = ((torch.argmax(pid_true)==2)*(e_true_<1.5)).bool()
            print("mask_muons", torch.sum(mask_muons))
            pid_pred = pid_pred[~mask_muons]
            pid_true = pid_true[~mask_muons]

        if len(pid_pred):
            acc = torch.sum(pid_pred==pid_true)/len(pid_pred)
            return torch.nn.CrossEntropyLoss(weight=weights)(
                pid_pred,
                pid_true
            ), acc, stats 
        else:
            return 0,0, stats
    else:
        return 0,0, stats


def correct_mask_neutral(pid_neutral, neural_mask):
    """
    pid_neutral: tensor of PIDs (shape [N])
    neural_mask: tensor of indices of neutral candidates (e.g. LongTensor)

    we remove indices where pid is in remove list
    """
    pid_neutral = pid_neutral.to(neural_mask.device)
    pid_neutral = torch.abs(pid_neutral)
    # PIDs to remove
    #remove_list = torch.tensor([-211, 211, -11, 11, 13, -13, 2212, 321], device=pid_neutral.device)
    keep_list = torch.tensor([22, 130, 2112], device=pid_neutral.device)

    # get PIDs corresponding to the given indices
    selected_pids = pid_neutral[neural_mask]          # <- index access
    # build mask: True = keep, False = remove
    keep_mask = torch.isin(selected_pids, keep_list)

    # filter indices
    corrected_indices = neural_mask[keep_mask.to(neural_mask.device)]

    return corrected_indices

def correct_mask_charged(pid_neutral, neural_mask, E):
    """
    pid_neutral: tensor of PIDs (shape [N])
    neural_mask: tensor of indices of neutral candidates (e.g. LongTensor)

    we remove indices where pid is in remove list
    """
    pid_neutral = pid_neutral.to(neural_mask.device)
    pid_neutral = torch.abs(pid_neutral)
    # PIDs to remove
    #remove_list = torch.tensor([-211, 211, -11, 11, 13, -13, 2212, 321], device=pid_neutral.device)
    keep_list = torch.tensor([22, 130, 2112], device=pid_neutral.device)

    # get PIDs corresponding to the given indices
    selected_pids = pid_neutral[neural_mask]          # <- index access
    # build mask: True = keep, False = remove
    keep_mask = torch.isin(selected_pids, keep_list)

    # filter indices
    corrected_indices = neural_mask[keep_mask.to(neural_mask.device)]

    return corrected_indices