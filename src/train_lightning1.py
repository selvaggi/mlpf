#!/usr/bin/env python

import os
import ast  # abstract syntax trees - maybe unused
import sys
import torch  # pytorch
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # GM: added by Rami - needed?
from torch.utils.data import DataLoader
import lightning as L  # lightning: high-level framework that abstracts and simplifies much of the boilerplate code needed in PyTorch
from src.utils.parser_args import parser
from lightning.pytorch.loggers import WandbLogger  # log using Weights and Biases (wandb)
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.utils.train_utils import (
    train_load,
    test_load,
)

import wandb  # to integrate W&B - not used and not needed (use WandbLogger instead)
import glob 
from src.utils.train_utils import get_samples_steps_per_epoch, model_setup, set_gpus
from src.utils.load_pretrained_models import load_train_model, load_test_model
from src.utils.callbacks import get_callbacks, get_callbacks_eval
from lightning.pytorch.profilers import AdvancedProfiler  # profiler - record info about time spent in each function call during vicen action - not used here


def main():
    # parse arguments
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)
   
    # training or inference?
    training_mode = not args.predict
    args.local_rank = 0
    # get dataloader (list of files to be used for training and validation; data config)
    # the data passed to the data loader is an iterable dataset of type SimpleIterDataset, derived from torch.utils.data.IterableDataset
    if training_mode:
        args.data_train = glob.glob(args.data_train[0]+ "*.parquet")
        args = get_samples_steps_per_epoch(args)
        train_loader, val_loader, data_config, train_input_names = train_load(args)
    else:
        args = get_samples_steps_per_epoch(args)
        test_loaders, data_config = test_load(args)
    args.is_muons = True
    # Set up model
    # loads model using the get_model() method of the python module passed via args.network_config
    model = model_setup(args, data_config)
    gpus, dev = set_gpus(args)
    # profiler = AdvancedProfiler(dirpath=".", filename="perf_logs_28112024")
    # start logger
    wandb_logger = WandbLogger(
        project=args.wandb_projectname,
        entity=args.wandb_entity,
        name=args.wandb_displayname,
        log_model="all",
    )
    # wandb_logger.experiment.config.update(args)

    # Training (on train/validation datasets)
    if training_mode:
        # optionally, initialize model with pre-trained weights
        if args.load_model_weights is not None:
            model = load_train_model(args, dev)

        # get pytorch callbacks: TQDMProgressBar, ModelCheckpoint, LearningRateMonitor
        # optionally also the FreezeClustering callback
        callbacks = get_callbacks(args)

        # setup training
        # ddp = DistributedDataParallel: PyTorch-native way to perform multi-GPU/multi-node training.
        # Each GPU gets its own copy of the model.
        # Input batches are split automatically across GPUs.
        # Gradients are synchronized between processes after backward.
        # This is the recommended and most efficient way to train with multiple GPUs.
        #
        # limit_xxx_batches: how many xxx batches to check
        trainer = L.Trainer(
            callbacks=callbacks,
            accelerator="gpu",
            devices=gpus,
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
            max_epochs=args.num_epochs,
            # strategy="ddp",
            limit_train_batches=args.train_batches, #! It is important that all gpus have the same number of batches, adjust this number acoordingly
            limit_val_batches=5,
        )
        args.local_rank = trainer.global_rank
        train_loader, val_loader, data_config, train_input_names = train_load(args)

        # do the training (fit the model to data)
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )


    # Evaluation (on test dataset)
    if args.data_test:
        # initialize model with pre-trained weights
        if args.load_model_weights:
            model = load_test_model(args, dev)

        print("\n====== Model structure ======")
        print(model)

        # setup the trainer
        trainer = L.Trainer(
            callbacks=get_callbacks_eval(args),
            accelerator="gpu",
            devices=[3],
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
        )

        # run an evaluation epoch over the test dataset
        # GM: NO DIFFERENCE BETWEEN THE TWO BLOCKS BELOW?
        if args.correction:
            for name, get_test_loader in test_loaders.items():
                test_loader = get_test_loader()
                trainer.validate(
                    model=model,
                    dataloaders=test_loader,
                    # ckpt_path=args.load_model_weights,
                )
        else:
            for name, get_test_loader in test_loaders.items():
                test_loader = get_test_loader()
                trainer.validate(
                    model=model,
                    # ckpt_path=args.load_model_weights,
                    dataloaders=test_loader,
                )


if __name__ == "__main__":
    main()
