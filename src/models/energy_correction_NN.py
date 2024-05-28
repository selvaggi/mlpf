'''
    PID predict energy correction
    The model taken from notebooks/13_NNs.py
    At first the model is fixed and the weights are loaded from earlier training
'''

import pickle
from copy import deepcopy
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.models.gravnet_3_L import GravnetModel
from torch_geometric.nn.models import GAT, GraphSAGE
from torch_scatter import scatter_mean
from gatr import GATr


class Net(nn.Module):
    def __init__(self, in_features=13, out_features=1):
        super(Net, self).__init__()
        self.out_features = out_features
        self.model = nn.ModuleList([
            #nn.BatchNorm1d(13),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)]
        )
        self.explainer_mode = False

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        for layer in self.model:
            x = layer(x)
        if self.out_features > 1:
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
            self.model.model = pickle.load(open(ckpt_file, 'rb'))
            print("Loaded energy correction model weights from", ckpt_file)
        #print("Temporarily not loading the model weights")
        self.model.to(device)
    def predict(self, x):
        #if isinstance(pred, tuple):
        #    return (pred[0].flatten(), pred[1])
        #return self.model(x).flatten(), None
        return self.model(x)


class ECNetWrapperGNN(torch.nn.Module):
    # use the GNN+NN model for energy correction
    def __init__(self, device, in_features=13, arch="vanilla"):
        super(ECNetWrapperGNN, self).__init__()
        gnn_features = 64
        self.model = Net(in_features=gnn_features, out_features=1)
        # use a GAT
        if arch == "GAT":
            self.gnn = GAT(in_features, out_channels=gnn_features, heads=4, concat=True, hidden_channels=64, num_layers=3)
        elif arch == "vanilla":
            self.gnn = GraphSAGE(in_features, gnn_features, hidden_channels=64, num_layers=3)
        #elif arch == "GATr":
        #    self.gnn = GATr(in_features,
        else:
            raise NotImplementedError
        self.model.to(device)
    def predict(self, x_global_features, graphs_new, explain=False):
        '''
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        '''
        assert explain is False, "Explain not implemented for this GNN"
        batch_num_nodes = graphs_new.batch_num_nodes()  # num hits in each graph
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
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class ECNetWrapperGNNGlobalFeaturesSeparate(torch.nn.Module):
    # use the GNN+NN model for energy correction
    # This one concatenates GNN features to the global features
    def __init__(self, device, in_features_global=13, in_features_gnn=13, out_features_gnn=32, ckpt_file=None, gnn=True, pos_regression=False):
        super(ECNetWrapperGNNGlobalFeaturesSeparate, self).__init__()
        out_f = 1
        self.pos_regression = pos_regression
        if pos_regression:
            out_f += 3
        self.model = Net(in_features=out_features_gnn + in_features_global, out_features=out_f)
        self.model.explainer_mode = False
        # use a GAT
        if gnn:
            self.gnn = GAT(in_features_gnn, out_channels=out_features_gnn, heads=4, concat=True, hidden_channels=64, num_layers=3)
            #self.gnn = GraphSAGE(in_channels=in_features_gnn, out_channels=out_features_gnn, hidden_channels=64, num_layers=3)
        else:
            self.gnn = None
        if ckpt_file is not None:
            # self.model.model = pickle.load(open(ckpt_file, 'rb'))
            with open(ckpt_file, 'rb') as f:
                self.model.model = CPU_Unpickler(f).load()
            print("Loaded energy correction model weights from", ckpt_file)
        self.model.to(device)
    def predict(self, x_global_features, graphs_new=None, explain=False):
        '''
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        '''
        if graphs_new is not None and self.gnn is not None:
            batch_num_nodes = graphs_new.batch_num_nodes()  # num hits in each graph
            batch_idx = []
            batch_bounds = []
            for i, n in enumerate(batch_num_nodes):
                batch_idx.extend([i] * n)
                batch_bounds.append(n)
            batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
            node_global_features = x_global_features  #
            x = graphs_new.ndata["h"]
            edge_index = torch.stack(graphs_new.edges())
            gnn_output = self.gnn(x, edge_index)
            gnn_output = scatter_mean(gnn_output, batch_idx, dim=0)
        else:
            # normally distr. 32 features
            gnn_output = torch.randn(x_global_features.shape[0], 32).to(x_global_features.device)
        model_x = torch.cat([x_global_features, gnn_output], dim=1).to(self.model.model[0].weight.device)
        if explain:
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
                explainer = shap.KernelExplainer(model_exp, model_x[:n_samples].detach().cpu().numpy())
                shap_vals = explainer.shap_values(model_x.detach().cpu().numpy(), nsamples=200)
            return self.model(model_x).flatten(), shap_vals, model_x.detach().cpu()
        res = self.model(model_x)
        if self.pos_regression:
            # normalize res[1] vectors
            return torch.clamp(res[0].flatten(), min=0, max=None), res[1] / torch.norm(res[1], dim=1).unsqueeze(1)
        return torch.clamp(res.flatten(), min=0, max=None)

class PickPAtDCA(torch.nn.Module):
    # Same layout of the module as the GNN one, but just picks the track
    def __init__(self, device, in_features_global=13, in_features_gnn=13, out_features_gnn=32, ckpt_file=None, gnn=True, pos_regression=False):
        super(PickPAtDCA, self).__init__()
    def predict(self, x_global_features, graphs_new=None, explain=False):
        '''
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        '''
        assert graphs_new is not None
        batch_num_nodes = graphs_new.batch_num_nodes()  # num hits in each graph
        batch_idx = []
        batch_bounds = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
        #ht = graphs_new.ndata["hit_type"]
        ht = graphs_new.ndata["h"][:, 3:7].argmax(dim=1)
        filt = (ht == 1) # track
        #if "pos_pxpypz_at_vertex" in graphs_new.ndata.keys():
        #    key = "pos_pxpypz_at_vertex"
        #else:
        #    key = "pos_pxpypz"
        p_direction = scatter_mean(graphs_new.ndata["pos_pxpypz_at_vertex"][filt], batch_idx[filt], dim=0)
        p_tracks = torch.norm(p_direction, dim=1)
        p_direction = p_direction / torch.norm(p_direction, dim=1).unsqueeze(1)
        #if self.pos_regression:
        return p_tracks, p_direction
        #return p_tracks
