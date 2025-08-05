#!/usr/bin/env python

import os
import ast
import sys
import torch
from torch.utils.data import DataLoader
import lightning as L
from src.utils.parser_args import parser
from lightning.pytorch.loggers import WandbLogger
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.utils.train_utils import (
    train_load,
    test_load,
)
import wandb

from src.utils.train_utils import get_samples_steps_per_epoch, model_setup, set_gpus
from src.utils.load_pretrained_models import load_train_model, load_test_model
from src.utils.callbacks import get_callbacks, get_callbacks_eval
from lightning.pytorch.profilers import AdvancedProfiler


def main():
    # parse arguments 
    args = parser.parse_args()
    args = get_samples_steps_per_epoch(args)
    args.local_rank = 0
    training_mode = not args.predict

    # get dataloader 
    if training_mode:
        train_loader, val_loader, data_config, train_input_names = train_load(args)
    else:
        test_loaders, data_config = test_load(args)
    args.is_muons = data_config.graph_config.get("muons", False)
    # Set up model
    model = model_setup(args, data_config)
    gpus, dev = set_gpus(args)
    #profiler = AdvancedProfiler(dirpath=".", filename="perf_logs_28112024")
    # start logger
    wandb_logger = WandbLogger(
        project=args.wandb_projectname,
        entity=args.wandb_entity,
        name=args.wandb_displayname,
        log_model="all",
    )
    # wandb_logger.experiment.config.update(args)

    # Training
    if training_mode:
        if args.load_model_weights is not None:
            model = load_train_model(args, dev)
        
        callbacks = get_callbacks(args)
        trainer = L.Trainer(
            callbacks=callbacks,
            accelerator="gpu",
            devices=gpus,
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
            max_epochs=args.num_epochs,
            strategy="ddp",
            limit_train_batches=12300, #! It is important that all gpus have the same number of batches, adjust this number acoordingly
            limit_val_batches=50,
        )
        args.local_rank = trainer.global_rank
        train_loader, val_loader, data_config, train_input_names = train_load(args)
        
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        
    # Evaluating
    if args.data_test:
        model = load_test_model(args, dev)

        trainer = L.Trainer(
            callbacks=get_callbacks_eval(args),
            accelerator="gpu",
            devices=[3],
            default_root_dir=args.model_prefix,
            logger=wandb_logger,
        )
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
