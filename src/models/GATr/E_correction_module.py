import torch
import torch.nn as nn
import io
import pickle

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

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
