'''
    PID predict energy correction
    The model taken from notebooks/13_NNs.py
    At first the model is fixed and the weights are loaded from earlier training
'''

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        if self.out_features > 1:
            return x[:, 0], x[:, 1:]
        return x

    def freeze_batchnorm(self):
        for layer in self.model:
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()
                print("Frozen batchnorm in 1st layer only - ", layer)
                break

class NetWrapper(torch.nn.Module):
    def __init__(self, ckpt_file, device, in_features=13, pid_predict_channels=4):
        super(NetWrapper, self).__init__()
        self.model = Net(in_features=in_features, out_features=1 + pid_predict_channels)
        # load weights from pickle
        if ckpt_file is not None:
            self.model.model = pickle.load(open(ckpt_file, 'rb'))
            print("Loaded energy correction model weights from", ckpt_file)
        #print("Temporarily not loading the model weights")
        self.model.to(device)
    def predict(self, x):
        pred = self.model(x)
        if isinstance(pred, tuple):
            return (pred[0].flatten(), pred[1])
        return self.model(x).flatten(), None
