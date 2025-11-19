#!/usr/bin/env python

import os
import ast
import sys
import shutil
import glob
import argparse
import functools
import numpy as np
import math
import torch
import wandb
import warnings

sys.path.append("/afs/cern.ch/work/m/mgarciam/private/mlpf/")
# warnings.filterwarnings("ignore")

from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
from src.utils.parser_args import parser
from lightning.pytorch.loggers import WandbLogger

from src.utils.train_utils import (
    train_load,
    test_load,
)
from src.utils.import_tools import import_module
import wandb
from src.utils.logger_wandb import log_wandb_init
from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.profilers import AdvancedProfiler
from src.models.gravnet_3_L import FreezeClustering

datasets = {
    "test": "/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia/gun_fakeCalo_g1/reco_gun_1000.root",
    "train": "/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia/gun_fakeCalo_g1/reco_gun_1000.root",
}


class Args:
    def __init__(self, datasets):
        self.data_train = [datasets]
        self.data_val = [datasets]
        # self.data_train = files_train
        self.data_config = "/afs/cern.ch/work/m/mgarciam/private/mlpf/config_files/config_tracking_global.yaml"
        self.extra_selection = None
        self.train_val_split = 1
        self.data_fraction = 0.1
        self.file_fraction = 1
        self.fetch_by_files = False
        self.fetch_step = 1
        self.steps_per_epoch = None
        self.in_memory = False
        self.local_rank = None
        self.copy_inputs = False
        self.no_remake_weights = False
        self.batch_size = 1
        self.num_workers = 0
        self.demo = False
        self.laplace = False
        self.diffs = False
        self.class_edges = False
        self.correction = False


args = {key: Args(value) for key, value in datasets.items()}

from src.models.gravnet_3_L import GravnetModel

load_model_weights = "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/test_L_tracking9/_epoch=30.ckpt"
model = GravnetModel.load_from_checkpoint(load_model_weights, args=args["train"], dev=0)

x = model.gravnet_blocks[0].gravnet_layer.lin_s.weight
torch.save(x.detach().cpu(), "tensor_0.pt")
x = model.gravnet_blocks[0].gravnet_layer.lin_h.weight
torch.save(x.detach().cpu(), "tensor_1.pt")
# x = model.ScaledGooeyBatchNorm2_1.weight
# torch.save(x.detach().cpu(), "tensor_2.pt")
# x = model.ScaledGooeyBatchNorm2_1.bias
# torch.save(x.detach().cpu(), "tensor_3.pt")
# # x = model.postgn_dense[2].weight
# torch.save(x.detach().cpu(), "tensor_1.pt")

# x = model.postgn_dense[4].weight
# torch.save(x.detach().cpu(), "tensor_2.pt")
