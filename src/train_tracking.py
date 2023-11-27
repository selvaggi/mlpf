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
import wandb

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
import warnings


def find_free_port():
    """https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number"""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def _main(args):
    warnings.filterwarnings("ignore")
    if args.condensation:
        from src.utils.nn.tools_condensation_tracking import train_regression as train
        from src.utils.nn.tools_condensation_tracking import (
            evaluate_regression as evaluate,
        )
    else:
        from src.utils.nn.tools import train_regression as train
        from src.utils.nn.tools import evaluate_regression as evaluate
        from src.utils.nn.tools_condensation import plot_regression_resolution

    # training/testing mode
    training_mode = not args.predict

    # load data
    if training_mode:
        train_loader, val_loader, data_config, train_input_names = train_load(args)
    else:
        test_loaders, data_config = test_load(args)
    # device
    if args.gpus:
        # distributed training
        if args.backend is not None:
            local_rank = args.local_rank
            print("localrank", local_rank)
            torch.cuda.set_device(local_rank)
            gpus = [local_rank]
            dev = torch.device(local_rank)
            print("initizaing group process")
            torch.distributed.init_process_group(backend=args.backend)
            _logger.info(f"Using distributed PyTorch with {args.backend} backend")
            print("ended initizaing group process")
        else:
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

    # note: we should always save/load the state_dict of the original model, not the one wrapped by nn.DataParallel
    # so we do not convert it to nn.DataParallel now
    orig_model = model
    training_mode = not args.predict
    if training_mode:
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

        # optimizer & learning rate
        opt, scheduler = optim(args, model, dev)

        # DataParallel
        if args.backend is None:
            if gpus is not None and len(gpus) > 1:
                # model becomes `torch.nn.DataParallel` w/ model.module being the original `torch.nn.Module`
                model = torch.nn.DataParallel(model, device_ids=gpus)
        if args.log_wandb and local_rank == 0:
            wandb.watch(model, log="all", log_freq=10)
            # model = model.to(dev)

        # training loop
        best_valid_metric = np.inf if args.regression_mode else 0
        grad_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        tb = None
        steps = 0  # for wandb logging
        add_energy_loss = False
        if args.clustering_and_energy_loss:
            add_energy_loss = True
            if args.energy_loss_delay > 0:
                add_energy_loss = False
        for epoch in range(args.num_epochs):
            if args.load_epoch is not None:
                if epoch <= args.load_epoch:
                    continue
            _logger.info("-" * 50)
            _logger.info("Epoch #%d training" % epoch)
            if args.clustering_and_energy_loss and epoch > args.energy_loss_delay:
                print("Switching on energy loss!")
                add_energy_loss = True
            steps += train(
                model,
                loss_func,
                opt,
                scheduler,
                train_loader,
                dev,
                epoch,
                steps_per_epoch=args.steps_per_epoch,
                grad_scaler=grad_scaler,
                tb_helper=tb,
                logwandb=args.log_wandb,
                local_rank=local_rank,
                current_step=steps,
                loss_terms=[args.clustering_loss_only, add_energy_loss],
                args=args,
                args_model=data_config,
                alternate_steps=args.alternate_steps_beta_clustering,
            )

            if args.model_prefix and (args.backend is None or local_rank == 0):
                dirname = os.path.dirname(args.model_prefix)
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)

                state_dict = (
                    model.module.state_dict()
                    if isinstance(
                        model,
                        (
                            torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel,
                        ),
                    )
                    else model.state_dict()
                )

                torch.save(state_dict, args.model_prefix + "_epoch-%d_state.pt" % epoch)
                torch.save(
                    opt.state_dict(),
                    args.model_prefix + "_epoch-%d_optimizer.pt" % epoch,
                )
            # if args.backend is not None and local_rank == 0:
            # TODO: save checkpoint
            #     save_checkpoint()

            _logger.info("Epoch #%d validating" % epoch)
            valid_metric = evaluate(
                model,
                val_loader,
                dev,
                epoch,
                loss_func=loss_func,
                steps_per_epoch=args.steps_per_epoch_val,
                tb_helper=tb,
                logwandb=args.log_wandb,
                energy_weighted=args.energy_loss,
                local_rank=local_rank,
                step=steps,
                loss_terms=[args.clustering_loss_only, args.clustering_and_energy_loss],
                args=args,
            )
            is_best_epoch = (
                (valid_metric < best_valid_metric)
                if args.regression_mode
                else (valid_metric > best_valid_metric)
            )
            if is_best_epoch:
                print("Best epoch!")
                best_valid_metric = valid_metric
                if args.model_prefix and (args.backend is None or local_rank == 0):
                    shutil.copy2(
                        args.model_prefix + "_epoch-%d_state.pt" % epoch,
                        args.model_prefix + "_best_epoch_state.pt",
                    )
                    # torch.save(model, args.model_prefix + '_best_epoch_full.pt')
            _logger.info(
                "Epoch #%d: Current validation metric: %.5f (best: %.5f)"
                % (epoch, valid_metric, best_valid_metric),
                color="bold",
            )

    if args.data_test:
        tb = None
        if args.backend is not None and local_rank != 0:
            return
        if args.log_wandb and local_rank == 0:
            import wandb
            from src.utils.logger_wandb import log_wandb_init

            wandb.init(project=args.wandb_projectname, entity=args.wandb_entity)
            wandb.run.name = args.wandb_displayname
            log_wandb_init(args, data_config)

        if training_mode:
            del train_loader, val_loader
            test_loaders, data_config = test_load(args)

        if not args.model_prefix.endswith(".onnx"):
            if args.predict_gpus:
                gpus = [int(i) for i in args.predict_gpus.split(",")]
                dev = torch.device(gpus[0])
            else:
                gpus = None
                dev = torch.device("cpu")
            model = orig_model.to(dev)
            if args.model_prefix:
                model_path = (
                    args.model_prefix
                    if args.model_prefix.endswith(".pt")
                    else args.model_prefix + "_best_epoch_state.pt"
                )
                _logger.info("Loading model %s for eval" % model_path)
                model.load_state_dict(torch.load(model_path, map_location=dev))
            if gpus is not None and len(gpus) > 1:
                model = torch.nn.DataParallel(model, device_ids=gpus)
            model = model.to(dev)

        for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()
            # run prediction
            # if args.model_prefix.endswith(".onnx"):
            #     _logger.info("Loading model %s for eval" % args.model_prefix)
            #     from src.utils.nn.tools import evaluate_onnx

            #     test_metric, scores, labels, observers = evaluate_onnx(
            #         args.model_prefix, test_loader
            #     )
            # else:
            # if len(args.data_plot):
            #     from pathlib import Path
            #     Path(args.data_plot).mkdir(parents=True, exist_ok=True)
            #     import matplotlib.pyplot as plt

            #     print("Plotting")
            #     figs = plot_regression_resolution(model, test_loader, dev)
            #     for name, fig in figs.items():
            #         fname = os.path.join(args.data_plot, name + ".pdf")
            #         fig.savefig(fname)
            #         print("Wrote to", fname)
            #         plt.close(fig)
            #     # write all cmdline arguments to a txt file
            #     with open(os.path.join(args.data_plot, "args.txt"), "w") as f:
            #         f.write(" ".join(sys.argv))
            #         f.write("\n")
            # else:
            test_metric, scores, labels, observers = evaluate(
                model,
                test_loader,
                dev,
                epoch=None,
                for_training=False,
                loss_func=loss_func,
                steps_per_epoch=args.steps_per_epoch_val,
                tb_helper=tb,
                logwandb=args.log_wandb,
                energy_weighted=args.energy_loss,
                local_rank=local_rank,
                loss_terms=[args.clustering_loss_only, args.clustering_and_energy_loss],
                args=args,
            )

            _logger.info("Test metric %.5f" % test_metric, color="bold")
            del test_loader

            # if args.predict_output:
            #     if "/" not in args.predict_output:
            #         predict_output = os.path.join(
            #             os.path.dirname(args.model_prefix),
            #             "predict_output",
            #             args.predict_output,
            #         )
            #     else:
            #         predict_output = args.predict_output
            #     os.makedirs(os.path.dirname(predict_output), exist_ok=True)
            #     if name == "":
            #         output_path = predict_output
            #     else:
            #         base, ext = os.path.splitext(predict_output)
            #         output_path = base + "_" + name + ext
            #     if output_path.endswith(".root"):
            #         save_root(args, output_path, data_config, scores, labels, observers)
            #     else:
            #         save_parquet(args, output_path, scores, labels, observers)
            #     _logger.info("Written output to %s" % output_path, color="bold")


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

    if args.cross_validation:
        model_dir, model_fn = os.path.split(args.model_prefix)
        var_name, kfold = args.cross_validation.split("%")
        kfold = int(kfold)
        for i in range(kfold):
            _logger.info(f"\n=== Running cross validation, fold {i} of {kfold} ===")
            args.model_prefix = os.path.join(f"{model_dir}_fold{i}", model_fn)
            args.extra_selection = f"{var_name}%{kfold}!={i}"
            args.extra_test_selection = f"{var_name}%{kfold}=={i}"
            _main(args)
    else:
        _main(args)


if __name__ == "__main__":
    main()
