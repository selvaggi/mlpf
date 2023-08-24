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
import sys
import os

torch.autograd.set_detect_anomaly(True)
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.logger.logger import _logger, _configLogger
from src.dataset.dataset import SimpleIterDataset
from src.utils.import_tools import import_module
from src.utils.train_utils import (
    to_filelist,
    train_load,
    onnx,
    test_load,
    iotest,
    model_setup,
    profile,
    optim,
    save_root,
    save_parquet,
)
from src.dataset.functions_graph import graph_batch_func
from src.utils.parser_args import parser

radius = 0.16

def find_free_port():
    """https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number"""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def _main(args, radius=0.6, batches=15):
    from src.utils.nn.tools_condensation import inference_statistics
    train_loader, val_loader, data_config, train_input_names = train_load(args)
    print(train_loader)
    # device
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]
        dev = torch.device(gpus[0])
        local_rank = 0
    else:
        gpus = None
        local_rank = 0
        dev = torch.device("cpu")
    model, model_info, loss_func = model_setup(args, data_config)
    from src.utils.train_utils import count_parameters
    num_parameters_counted = count_parameters(model)
    print(num_parameters_counted)
    orig_model = model
    training_mode = not args.predict
    if args.log_wandb and local_rank == 0:
        import wandb
        from src.utils.logger_wandb import log_wandb_init
        wandb.init(project=args.wandb_projectname, entity=args.wandb_entity)
        wandb.run.name = args.wandb_displayname
        log_wandb_init(args, data_config)
    model = orig_model.to(dev)
    if args.model_pretrained:
        model_path = args.model_pretrained
        _logger.info("Loading model %s for training from there on" % model_path)
        model.load_state_dict(torch.load(model_path, map_location=dev))
    print("MODEL DEVICE", next(model.parameters()).is_cuda)
    # DistributedDataParallel
    if args.backend is not None:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=gpus,
            output_device=local_rank,
            find_unused_parameters=True,
        )
    # DataParallel
    if args.backend is None:
        if gpus is not None and len(gpus) > 1:
            # model becomes `torch.nn.DataParallel` w/ model.module being the original `torch.nn.Module`
            model = torch.nn.DataParallel(model, device_ids=gpus)
    if args.log_wandb and local_rank == 0:
        wandb.watch(model, log="all", log_freq=10)
        # model = model.to(dev)
    grad_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    add_energy_loss = False
    if args.clustering_and_energy_loss:
        add_energy_loss = True
        if args.energy_loss_delay > 0:
            add_energy_loss = False
    result = inference_statistics(
        model,
        train_loader,
        dev,
        grad_scaler,
        loss_terms=[args.clustering_loss_only, add_energy_loss],
        args=args,
        radius=radius,
        total_num_batches=batches,
        save_ckpt_to_folder="/eos/user/g/gkrzmanc/summ_results/frac_energy_plots/23_08_larger_DS_known_particle_ckpts"
    )

    return result

def main():
    args = parser.parse_args()

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

    if "{auto}" in args.model_prefix or "{auto}" in args.log:
        import hashlib
        import time

        model_name = (
            time.strftime("%Y%m%d-%H%M%S")
            + "_"
            + os.path.basename(args.network_config).replace(".py", "")
        )
        if len(args.network_option):
            model_name = (
                model_name
                + "_"
                + hashlib.md5(str(args.network_option).encode("utf-8")).hexdigest()
            )
        model_name += "_{optim}_lr{lr}_batch{batch}".format(
            lr=args.start_lr, optim=args.optimizer, batch=args.batch_size
        )
        args._auto_model_name = model_name
        args.model_prefix = args.model_prefix.replace("{auto}", model_name)
        args.log = args.log.replace("{auto}", model_name)
        print("Using auto-generated model prefix %s" % args.model_prefix)

    if args.predict_gpus is None:
        args.predict_gpus = args.gpus

    args.local_rank = (
        None if args.backend is None else int(os.environ.get("LOCAL_RANK", "0"))
    )
    if args.backend is not None:
        port = find_free_port()
        args.port = port
        world_size = torch.cuda.device_count()
    stdout = sys.stdout
    if args.local_rank is not None:
        args.log += ".%03d" % args.local_rank
        if args.local_rank != 0:
            stdout = None
    _configLogger("weaver", stdout=stdout, filename=args.log)

    results = {}
    for rad in [0.4]:
        results[rad] = _main(args, radius=rad, batches=99999999999)
        #print(results[rad]["loss_e_fracs"])
    import pickle
    with open("/eos/user/g/gkrzmanc/summ_results/frac_energy_plots/23_08_larger_DS_known_particles_partial_results.pkl", "wb") as f:
        pickle.dump(results, f)
main()

