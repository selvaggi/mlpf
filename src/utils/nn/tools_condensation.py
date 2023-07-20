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


from src.layers.object_cond import onehot_particles_arr


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
):
    model.train()
    # print("starting to train")
    iterator = iter(train_loader)
    g, y = next(iterator)
    iterator = iter(train_loader)
    # print("LEN DATALOADER", g)
    # print(y)
    data_config = train_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    step_count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for batch_g, y in tq:
            # print(batch_g)
            # print(y)
            label = y
            step_count += 1
            num_examples = label.shape[0]
            label = label.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                batch_g = batch_g.to(dev)
                model_output = model(batch_g)
                preds = model_output.squeeze()
                loss, losses = model.mod.object_condensation_loss2(
                    batch_g, model_output, y
                )
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, "_update_per_step", False):
                scheduler.step()

            loss = loss.item()

            num_batches += 1
            count += num_examples
            total_loss += loss

            tq.set_postfix(
                {
                    "lr": "%.2e" % scheduler.get_last_lr()[0]
                    if scheduler
                    else opt.defaults["lr"],
                    "Loss": "%.5f" % loss,
                    "AvgLoss": "%.5f" % (total_loss / num_batches),
                }
            )

            if tb_helper:
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

            if logwandb and (num_batches % 50):
                wandb.log({"loss regression": loss,
                           "loss lv": losses[0],
                           "loss beta": losses[1],
                           "loss E": losses[2],
                           "loss X": losses[3],
                           "loss PID": losses[4]}, step=num_batches)

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

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

    if scheduler and not getattr(scheduler, "_update_per_step", False):
        scheduler.step()


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
        local_rank=0
):
    '''

    :param model:
    :param test_loader:
    :param dev:
    :param epoch:
    :param for_training:
    :param loss_func:
    :param steps_per_epoch:
    :param eval_metrics:
    :param tb_helper:
    :param logwandb:
    :param energy_weighted:
    :param local_rank:
    :return:
    '''
    model.eval()

    data_config = test_loader.dataset.config

    center, scales = _check_scales_centers(iter(test_loader.dataset))
    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores = []
    results = []  # resolution results
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch_g, y in tq:
                batch_g = batch_g.to(dev)
                label = y
                num_examples = label.shape[0]
                label = label.to(dev)
                model_output = model(batch_g)

                preds = model_output.squeeze().float()

                loss, losses = model.mod.object_condensation_loss2(
                    batch_g, model_output, y
                )

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples

                tq.set_postfix(
                    {
                        "Loss": "%.5f" % loss,
                        "AvgLoss": "%.5f" % (total_loss / count),
                    }
                )

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(
                                model_output=model_output,
                                model=model,
                                epoch=epoch,
                                i_batch=num_batches,
                                mode="eval" if for_training else "test",
                            )

                if logwandb and (num_batches % 50):
                    wandb.log({
                        "loss val regression": loss,
                        "loss val lv": losses[0],
                        "loss val beta": losses[1],
                        "loss val E": losses[2],
                        "loss val X": losses[3]
                    })

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info(
        "Processed %d entries in total (avg. speed %.1f entries/s)"
        % (count, count / time_diff)
    )

    if tb_helper:
        tb_mode = "eval" if for_training else "test"
        tb_helper.write_scalars(
            [
                ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
                ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
                ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
            ]
        )
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(
                    model_output=model_output,
                    model=model,
                    epoch=epoch,
                    i_batch=-1,
                    mode=tb_mode,
                )

    # scores = np.concatenate(scores)
    # labels = {k: _concat(v) for k, v in labels.items()}
    # metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    # _logger.info('Evaluation metrics: \n%s', '\n'.join(
    #    ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))
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


def plot_regression_resolution(
        model,
        test_loader,
        dev,
        **kwargs
):
    model.eval()
    results = []  # resolution results
    pid_classification_results = []
    c = 0
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch_g, y in tq:
                c += 1
                if c > 5:
                    break  # TEMPORARY
                batch_g = batch_g.to(dev)
                model_output = model(batch_g)
                resolutions, pid_true, pid_pred = model.mod.object_condensation_loss2(
                    batch_g, model_output, y, return_resolution=True
                )
                results.append(resolutions)
                pid_classification_results.append((pid_true, pid_pred))
    result_dict = {}
    for key in results[0]:
        result_dict[key] = np.concatenate([r[key] for r in results])
    result_dict["event_by_event_accuracy"] = [accuracy_score(pid_true.argmax(dim=0), pid_pred.argmax(dim=0)) for pid_true, pid_pred in pid_classification_results]
    # just plot all for now
    result = {}
    for key in results[0]:
        data = result_dict[key]
        fig, ax = plt.subplots()
        ax.hist(data, bins=100, range=(-1.5, 1.5), histtype="step", label=key)
        ax.set_xlabel("resolution")
        ax.set_ylabel("count")
        ax.legend()
        result[key] = fig
    conf_mat = confusion_matrix(pid_true.argmax(dim=0), pid_pred.argmax(dim=0))
    # confusion matrix
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # add onehot_particle_arr as class names
    class_names = onehot_particles_arr
    im = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(class_names)), class_names, rotation=45)
    ax.set_yticks(np.arange(len(class_names)), class_names)
    result["PID_confusion_matrix"] = fig

    return result

