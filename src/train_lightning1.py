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

# warnings.filterwarnings("ignore")

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from src.utils.parser_args import parser
from lightning.pytorch.loggers import WandbLogger

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
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

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"


def get_samples_steps_per_epoch(args):
    if args.samples_per_epoch is not None:
        if args.steps_per_epoch is None:
            args.steps_per_epoch = args.samples_per_epoch // args.batch_size
        else:
            raise RuntimeError(
                "Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!"
            )
    if args.samples_per_epoch_val is not None:
        if args.steps_per_epoch_val is None:
            args.steps_per_epoch_val = args.samples_per_epoch_val // args.batch_size
        else:
            raise RuntimeError(
                "Please use either `--steps-per-epoch-val` or `--samples-per-epoch-val`, but not both!"
            )
    if args.steps_per_epoch_val is None and args.steps_per_epoch is not None:
        args.steps_per_epoch_val = round(
            args.steps_per_epoch * (1 - args.train_val_split) / args.train_val_split
        )
    if args.steps_per_epoch_val is not None and args.steps_per_epoch_val < 0:
        args.steps_per_epoch_val = None
    return args


# TODO change this to use it from config file
def model_setup(args, data_config):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config, name="_network_module")
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}

    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]  # ?
        dev = torch.device(gpus[0])
        print("using GPUs:", gpus)
    else:
        gpus = None
        local_rank = 0
        dev = torch.device("cpu")
    model, model_info = network_module.get_model(
        data_config, args=args, dev=dev, **network_options
    )
    return model.mod


def get_gpu_dev(args):
    if args.gpus != "":
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = 0
        devices = 0
    return accelerator, devices


def main():
    args = parser.parse_args()
    args = get_samples_steps_per_epoch(args)
    args.local_rank = 0

    training_mode = not args.predict
    if training_mode:
        train_loader, val_loader, data_config, train_input_names = train_load(args)
    else:
        test_loaders, data_config = test_load(args)

    model = model_setup(args, data_config)
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]
        print("Using GPUs:", gpus)
    else:
        print("No GPUs flag provided - Setting GPUs to [0]")
        gpus = [0]
        raise Exception("Please provide GPU number")
    wandb_logger = WandbLogger(
        project=args.wandb_projectname,
        entity=args.wandb_entity,
        name=args.wandb_displayname,
        log_model="all",
    )
    if training_mode:
        # wandb.init(project=args.wandb_projectname, entity=args.wandb_entity)
        # wandb.run.name = args.wandb_displayname
        if args.load_model_weights is not None and args.correction:
            from src.models.GATr.Gatr_pf_e import ExampleWrapper as GravnetModel

            model = GravnetModel.load_from_checkpoint(
                args.load_model_weights, args=args, dev=0
            )

        elif args.load_model_weights is not None:
            from src.models.GATr.Gatr_pf_e import ExampleWrapper as GravnetModel

            model = GravnetModel.load_from_checkpoint(
                args.load_model_weights, args=args, dev=0
            )

        accelerator, devices = get_gpu_dev(args)
        val_every_n_epochs = 1
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.model_prefix,  # checkpoints_path, # <--- specify this on the trainer itself for version control
            filename="_{epoch}_{step}",
            # every_n_epochs=val_every_n_epochs,
            every_n_train_steps=1000,
            save_top_k=-1,  # <--- this is important!
            save_weights_only=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks = [
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
            lr_monitor,
        ]
        if args.freeze_clustering:
            callbacks.append(FreezeClustering())
        #profiler = AdvancedProfiler(dirpath=".", filename="perf_logs_train_23042024_2")
        trainer = L.Trainer(
            callbacks=callbacks,
            accelerator="gpu",
            devices=gpus,
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
            #profiler=profiler,
            max_epochs=args.num_epochs,
            # accumulate_grad_batches=1,
            strategy="ddp",
            # limit_train_batches=100,
            limit_val_batches=100,
            # precision=16
            # resume_from_checkpoint=args.load_model_weig
            # hts,
        )
        args.local_rank = trainer.global_rank
        train_loader, val_loader, data_config, train_input_names = train_load(args)

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            # ckpt_path=args.load_model_weights,
        )

        # TODO save checkpoints and hyperparameters
        # TODO use accumulate_grad_batches=7

    if args.data_test:
        if args.load_model_weights is not None and args.correction:
            from src.models.GATr.Gatr_pf_e import ExampleWrapper as GravnetModel
            model = GravnetModel.load_from_checkpoint(
                args.load_model_weights, args=args, dev=0
            )
        #profiler = AdvancedProfiler(dirpath=".", filename="perf_logs_eval_23042024")
        trainer = L.Trainer(
            callbacks=[TQDMProgressBar(refresh_rate=1)],
            accelerator="gpu",
            #profiler=profiler,
            devices=gpus,
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
            # limit_val_batches=19,
        )
        if args.correction:
            for name, get_test_loader in test_loaders.items():
                test_loader = get_test_loader()
                trainer.validate(
                    model=model,
                    dataloaders=test_loader,
                )
        else:
            for name, get_test_loader in test_loaders.items():
                test_loader = get_test_loader()
                trainer.validate(
                    model=model,
                    ckpt_path=args.load_model_weights,
                    dataloaders=test_loader,
                )


if __name__ == "__main__":
    main()
