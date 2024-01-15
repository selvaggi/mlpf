import numpy as np
import awkward as ak
import tqdm
import time
import torch
from collections import defaultdict, Counter
from src.utils.metrics import evaluate_metrics
from src.data.tools import _concat
from src.logger.logger import _logger
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pathlib import Path
import os
import pickle
from src.models.gravnet_3 import object_condensation_loss_tracking
from src.layers.inference_oc_tracks import evaluate_efficiency_tracks
from src.layers.object_cond import (
    onehot_particles_arr,
    get_clustering,
    calc_LV_Lbeta_inference,
)
from src.utils.logger_wandb import plot_clust


class_names = ["other"] + [str(i) for i in onehot_particles_arr]  # quick fix


def lst_nonzero(x):
    return x[x != 0.0]


def clip_list(l, clip_val=4.0):
    result = []
    for item in l:
        if abs(item) > clip_val:
            if item > 0:
                result.append(clip_val)
            else:
                result.append(-clip_val)
        elif np.isnan(item):
            result.append(0.0)  # i don't know why the hell we need this
        else:
            result.append(item)
    return result


def train_regression(
    model,
    loss_func,
    opt,
    scheduler,
    train_loader,
    dev,
    epoch,
    steps_per_epoch=None,
    grad_scaler=None,
    tb_helper=None,
    logwandb=False,
    local_rank=0,
    current_step=0,  # current_step: used for logging correctly
    loss_terms=[],  # whether to only optimize the clustering loss
    args=None,
    args_model=None,
    alternate_steps=None,  # alternate_steps: after how many steps to switch between beta and clustering loss
):
    model.train()
    # print("starting to train")
    iterator = iter(train_loader)
    g, y = next(iterator)
    iterator = iter(train_loader)
    # print("LEN DATALOADER", g)
    # print(y)
    data_config = train_loader.dataset.config
    clust_loss_only = loss_terms[0]
    add_energy_loss = loss_terms[1]  # whether to add energy loss to the clustering loss
    total_loss = 0
    num_batches = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    step_count = current_step
    start_time = time.time()
    prev_time = time.time()
    loss_epoch_total, losses_epoch_total = [], []
    if alternate_steps is not None:
        if not hasattr(model.mod, "current_state_alternate_steps"):
            model.mod.current_state_alternate_steps = 0
    with tqdm.tqdm(train_loader) as tq:
        for batch_g, y in tq:
            load_end_time = time.time()
            label = y
            step_count += 1
            num_examples = label.shape[0]
            label = label.to(dev)
            opt.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                batch_g = batch_g.to(dev)
                calc_e_frac_loss = (num_batches % 250) == 0
                if args.loss_regularization:
                    model_output, loss_regularizing_neig, loss_ll = model(batch_g)
                else:
                    if local_rank == 0:
                        model_output = model(batch_g, step_count)
                    else:
                        model_output = model(batch_g, 1)
                preds = model_output.squeeze()
                (loss, losses) = object_condensation_loss_tracking(
                    batch_g,
                    model_output,
                    y,
                    clust_loss_only=clust_loss_only,
                    add_energy_loss=add_energy_loss,
                    calc_e_frac_loss=calc_e_frac_loss,
                    q_min=args.qmin,
                    frac_clustering_loss=args.frac_cluster_loss,
                    attr_weight=args.L_attractive_weight,
                    repul_weight=args.L_repulsive_weight,
                    fill_loss_weight=args.fill_loss_weight,
                    use_average_cc_pos=args.use_average_cc_pos,
                    hgcalloss=args.hgcalloss,
                    tracking=True,
                )

                loss = loss  # add energy loss
                if args.loss_regularization:
                    loss = loss + loss_regularizing_neig + loss_ll
                betas = (
                    torch.sigmoid(
                        torch.reshape(preds[:, args.clustering_space_dim], [-1, 1])
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                # wandb log betas hist
                if logwandb and local_rank == 0:
                    wandb.log(
                        {
                            "betas": wandb.Histogram(
                                torch.nan_to_num(torch.tensor(betas), 0.0)
                            ),
                            "qs": wandb.Histogram(
                                np.arctanh(betas.clip(0.0, 1 - 1e-4) / 1.002) ** 2
                                + args.qmin
                            ),
                        }
                    )  # , step=step_count)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()
            step_end_time = time.time()
            if logwandb and (num_batches % 10) == 0 and local_rank == 0:
                wandb.log(
                    {
                        "load_time": load_end_time - prev_time,
                        "step_time": step_end_time - load_end_time,
                    }
                )  # , step=step_count)
            loss = loss.item()

            num_batches += 1
            count += num_examples
            total_loss += loss

            if scheduler and getattr(scheduler, "_update_per_step"):
                if args.lr_scheduler == "reduceplateau":
                    scheduler.step(loss)  # loss
                else:
                    scheduler.step()  # loss
                if logwandb and local_rank == 0:
                    if args.lr_scheduler == "reduceplateau":
                        wandb.log({"lr": opt.param_groups[0]["lr"]})
                    else:
                        wandb.log({"lr": scheduler.get_last_lr()[0]})

            if tb_helper:
                print("tb_helper!", tb_helper)
                tb_helper.write_scalars(
                    [
                        ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ]
                )
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(
                            model_output=model_output,
                            model=model,
                            epoch=epoch,
                            i_batch=num_batches,
                            mode="train",
                        )

            if logwandb and ((num_batches - 1) % 10) == 0 and local_rank == 0:
                # pid_true, pid_pred = losses[7], losses[8]
                loss_epoch_total.append(loss)
                losses_epoch_total.append(losses)

                wandb.log(
                    {
                        "loss regression": loss,
                        "loss lv": losses[0],
                        "loss beta": losses[1],
                        "loss attractive": losses[2],
                        "loss repulsive": losses[3],
                    }
                )

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break
            prev_time = time.time()

    time_diff = time.time() - start_time
    _logger.info(
        "Processed %d entries in total (avg. speed %.1f entries/s)"
        % (count, count / time_diff)
    )
    if count > 0 and num_batches > 0:
        _logger.info(
            "Train AvgLoss: %.5f, AvgMSE: %.5f, AvgMAE: %.5f"
            % (total_loss / num_batches, sum_sqr_err / count, sum_abs_err / count)
        )

        if tb_helper:
            tb_helper.write_scalars(
                [
                    ("Loss/train (epoch)", total_loss / num_batches, epoch),
                    ("MSE/train (epoch)", sum_sqr_err / count, epoch),
                    ("MAE/train (epoch)", sum_abs_err / count, epoch),
                ]
            )
            if tb_helper.custom_fn:
                with torch.no_grad():
                    tb_helper.custom_fn(
                        model_output=model_output,
                        model=model,
                        epoch=epoch,
                        i_batch=-1,
                        mode="train",
                    )
            # update the batch state
            tb_helper.batch_train_count += num_batches
        if scheduler and getattr(scheduler, "_update_per_step") == False:
            if args.lr_scheduler == "reduceplateau":
                scheduler.step(total_loss / num_batches)  # loss
                if logwandb and local_rank == 0:
                    wandb.log({"total_loss batch": total_loss / num_batches})
            else:
                scheduler.step()  # loss
            if logwandb and local_rank == 0:
                if args.lr_scheduler == "reduceplateau":
                    wandb.log({"lr": opt.param_groups[0]["lr"]})
                else:
                    wandb.log({"lr": scheduler.get_last_lr()[0]})
    return step_count


def evaluate_regression(
    model,
    test_loader,
    dev,
    epoch,
    for_training=True,
    loss_func=None,
    steps_per_epoch=None,
    eval_metrics=[
        "mean_squared_error",
        "mean_absolute_error",
        "median_absolute_error",
        "mean_gamma_deviance",
    ],
    tb_helper=None,
    logwandb=False,
    energy_weighted=False,
    local_rank=0,
    step=0,
    loss_terms=[],
    args=None,
):
    model.eval()

    data_config = test_loader.dataset.config

    center, scales = _check_scales_centers(iter(test_loader.dataset))
    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    clust_loss_only = loss_terms[0]
    add_energy_loss = loss_terms[1]  # whether to add energy loss to the clustering loss
    count = 0
    scores = []
    results = []  # resolution results
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    all_val_loss, all_val_losses = [], []
    step = 0
    df_showers = []
    df_showers_pandora = []
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch_g, y in tq:
                calc_e_frac_loss = num_batches % 10 == 0
                batch_g = batch_g.to(dev)
                label = y
                num_examples = label.shape[0]
                label = label.to(dev)
                if args.loss_regularization:
                    model_output, loss_regularizing_neig, loss_ll = model(batch_g)
                else:
                    if args.predict:
                        step_plotting = 0
                    else:
                        step_plotting = 1
                    model_output = model(batch_g, step_plotting)
                (loss, losses) = object_condensation_loss_tracking(
                    batch_g,
                    model_output,
                    y,
                    frac_clustering_loss=0,
                    q_min=args.qmin,
                    clust_loss_only=args.clustering_loss_only,
                    use_average_cc_pos=args.use_average_cc_pos,
                    hgcalloss=args.hgcalloss,
                    tracking=True,
                )

                step += 1

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples

                tq.set_postfix(
                    {
                        "Loss": "%.5f" % loss,
                        "AvgLoss": "%.5f" % (total_loss / count),
                    }
                )

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break
                # if args.predict:
                #     df_batch, df_batch_pandora = create_and_store_graph_output(
                #         batch_g,
                #         model_output,
                #         y,
                #         local_rank,
                #         step,
                #         epoch,
                #         path_save=args.model_prefix + "/showers_df_evaluation",
                #         store=False,
                #         predict=True,
                #     )
                #     df_showers.append(df_batch)
                #     df_showers_pandora.append(df_batch_pandora)
    # calculate showers at the end of every epoch
    # if logwandb and local_rank == 0:
    #     if args.predict:
    #         from src.layers.inference_oc import store_at_batch_end
    #         import pandas as pd

    #         df_showers = pd.concat(df_showers)
    #         df_showers_pandora = pd.concat(df_showers_pandora)
    #         store_at_batch_end(
    #             path_save=args.model_prefix + "/showers_df_evaluation",
    #             df_batch=df_showers,
    #             df_batch_pandora=df_showers_pandora,
    #             step=0,
    #             predict=True,
    #         )
    #     else:
    #         create_and_store_graph_output(
    #             batch_g,
    #             model_output,
    #             y,
    #             local_rank,
    #             step,
    #             epoch,
    #             path_save=args.model_prefix + "/showers_df_evaluation",
    #             store=True,
    #             predict=False,
    #         )
    if logwandb and local_rank == 0:

        wandb.log(
            {
                "loss val regression": loss,
                "loss val lv": losses[0],
                "loss val beta": losses[1],
                "loss val attractive": losses[2],
                "loss val repulsive": losses[3],
            }
        )
        evaluate_efficiency_tracks(
            batch_g,
            model_output,
            y,
            local_rank,
            step,
            epoch,
            path_save=args.model_prefix + "showers_df_evaluation",
            store=True,
            predict=False,
            tracks=args.tracks,
        )
    time_diff = time.time() - start_time
    _logger.info(
        "Processed %d entries in total (avg. speed %.1f entries/s)"
        % (count, count / time_diff)
    )
    if count > 0:
        if for_training:
            return total_loss / count
        else:
            # convert 2D labels/scores
            observers = {k: _concat(v) for k, v in observers.items()}
            return total_loss / count, scores, labels, observers


def _check_scales_centers(iterator):
    regress_items = ["part_theta", "part_phi"]
    centers = np.zeros(2)
    scales = np.zeros(2)
    for ii, item in enumerate(regress_items):
        centers[ii] = iterator._data_config.preprocess_params[item]["center"]
        scales[ii] = iterator._data_config.preprocess_params[item]["scale"]
    return centers, scales


def upd_dict(d, small_dict):
    for k in small_dict:
        if k not in d:
            d[k] = []
        d[k] += small_dict[k]
    return d
