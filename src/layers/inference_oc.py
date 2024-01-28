import dgl
import torch
import os
from sklearn.cluster import DBSCAN, HDBSCAN
from torch_scatter import scatter_max, scatter_add, scatter_mean
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
import wandb


def create_and_store_graph_output(
    batch_g,
    model_output,
    y,
    local_rank,
    step,
    epoch,
    path_save,
    store=False,
    predict=False,
    tracking=False,
    e_corr=None,
    tracks=False,
):
    number_of_showers_total = 0
    number_of_showers_total1 = 0
    batch_g.ndata["coords"] = model_output[:, 0:3]
    batch_g.ndata["beta"] = model_output[:, 3]
    if not tracking:
        if e_corr is None:
            batch_g.ndata["correction"] = model_output[:, 4]
    graphs = dgl.unbatch(batch_g)
    batch_id = y[:, -1].view(-1)

    df_list = []
    df_list1 = []
    df_list_pandora = []
    for i in range(0, len(graphs)):
        mask = batch_id == i
        dic = {}
        dic["graph"] = graphs[i]
        dic["part_true"] = y[mask]

        # betas = torch.sigmoid(dic["graph"].ndata["beta"])
        if not tracking:
            if e_corr is None:
                correction_e = dic["graph"].ndata["correction"].view(-1)
        X = dic["graph"].ndata["coords"]
        # clustering_mode = "dbscan"
        # if clustering_mode == "clustering_normal":
        #     clustering = get_clustering(betas, X)
        # elif clustering_mode == "dbscan":
        labels = dbscan_obtain_labels(X, model_output.device)
        labels_hdb = hfdb_obtain_labels(X, model_output.device)
        if predict:
            labels_pandora = get_labels_pandora(tracks, dic, model_output.device)

        particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
        shower_p_unique = torch.unique(labels)
        shower_p_unique, row_ind, col_ind, i_m_w = match_showers(
            labels,
            dic,
            particle_ids,
            model_output,
            local_rank,
            i,
            path_save,
            tracks=tracks,
        )
        shower_p_unique_hdb, row_ind_hdb, col_ind_hdb, i_m_w_hdb = match_showers(
            labels_hdb,
            dic,
            particle_ids,
            model_output,
            local_rank,
            i,
            path_save,
            tracks=tracks,
            hdbscan=True,
        )
        if predict:
            (
                shower_p_unique_pandora,
                row_ind_pandora,
                col_ind_pandora,
                i_m_w_pandora,
            ) = match_showers(
                labels_pandora,
                dic,
                particle_ids,
                model_output,
                local_rank,
                i,
                path_save,
                pandora=True,
                tracks=tracks,
            )

        # if len(shower_p_unique_hdb) < len(particle_ids):
        # print("storing  event", local_rank, step, i)
        # torch.save(
        #     dic,
        #     path_save
        #     + "/graphs_all_debug/"
        #     + str(local_rank)
        #     + "_"
        #     + str(step)
        #     + "_"
        #     + str(i)
        #     + ".pt",
        # )
        if len(shower_p_unique) > 1:
            df_event, number_of_showers_total = generate_showers_data_frame(
                labels,
                dic,
                shower_p_unique,
                particle_ids,
                row_ind,
                col_ind,
                i_m_w,
                e_corr=e_corr,
                number_of_showers_total=number_of_showers_total,
                step=step,
                number_in_batch=i,
                tracks=tracks,
            )
            df_event1, number_of_showers_total1 = generate_showers_data_frame(
                labels_hdb,
                dic,
                shower_p_unique_hdb,
                particle_ids,
                row_ind_hdb,
                col_ind_hdb,
                i_m_w_hdb,
                e_corr=e_corr,
                number_of_showers_total=number_of_showers_total1,
                step=step,
                number_in_batch=i,
                tracks=tracks,
            )
            df_list.append(df_event)
            df_list1.append(df_event1)
            if predict:
                df_event_pandora = generate_showers_data_frame(
                    labels_pandora,
                    dic,
                    shower_p_unique_pandora,
                    particle_ids,
                    row_ind_pandora,
                    col_ind_pandora,
                    i_m_w_pandora,
                    pandora=True,
                    tracking=tracking,
                    step=step,
                    number_in_batch=i,
                    tracks=tracks,
                )
                df_list_pandora.append(df_event_pandora)

    df_batch = pd.concat(df_list)
    df_batch1 = pd.concat(df_list1)
    if predict:
        df_batch_pandora = pd.concat(df_list_pandora)
    #
    if store:
        store_at_batch_end(
            path_save,
            df_batch,
            df_batch1,
            df_list_pandora,
            local_rank,
            step,
            epoch,
            predict=True,
        )
    if predict:
        return df_batch, df_batch_pandora, df_batch1
    else:
        return df_batch


def store_at_batch_end(
    path_save,
    df_batch,
    df_batch1,
    df_batch_pandora,
    local_rank=0,
    step=0,
    epoch=None,
    predict=False,
):
    path_save_ = (
        path_save + "/" + str(local_rank) + "_" + str(step) + "_" + str(epoch) + ".pt"
    )
    df_batch.to_pickle(path_save_)
    path_save_ = (
        path_save
        + "/"
        + str(local_rank)
        + "_"
        + str(step)
        + "_"
        + str(epoch)
        + "_hdbscan.pt"
    )
    df_batch1.to_pickle(path_save_)
    if predict:
        path_save_pandora = (
            path_save
            + "/"
            + str(local_rank)
            + "_"
            + str(step)
            + "_"
            + str(epoch)
            + "_pandora.pt"
        )
        df_batch_pandora.to_pickle(path_save_pandora)
    log_efficiency(df_batch)
    if predict:
        log_efficiency(df_batch_pandora, pandora=True)


def log_efficiency(df, pandora=False):
    mask = ~np.isnan(df["reco_showers_E"])
    eff = np.sum(~np.isnan(df["pred_showers_E"][mask].values)) / len(
        df["pred_showers_E"][mask].values
    )
    if pandora:
        wandb.log({"efficiency validation pandora": eff})
    else:
        wandb.log({"efficiency validation": eff})


def generate_showers_data_frame(
    labels,
    dic,
    shower_p_unique,
    particle_ids,
    row_ind,
    col_ind,
    i_m_w,
    pandora=False,
    tracking=False,
    e_corr=None,
    number_of_showers_total=None,
    step=0,
    number_in_batch=0,
    tracks=False,
):
    e_pred_showers = scatter_add(dic["graph"].ndata["e_hits"].view(-1), labels)
    if pandora:
        e_pred_showers_cali = scatter_mean(
            dic["graph"].ndata["pandora_cluster_energy"].view(-1), labels
        )
        e_pred_showers_pfo = scatter_mean(
            dic["graph"].ndata["pandora_pfo_energy"].view(-1), labels
        )
    else:
        if e_corr is None:
            corrections_per_shower = get_correction_per_shower(labels, dic)
            e_pred_showers_cali = e_pred_showers * corrections_per_shower
        else:
            corrections_per_shower = e_corr.view(-1)

    e_reco_showers = scatter_add(
        dic["graph"].ndata["e_hits"].view(-1),
        dic["graph"].ndata["particle_number"].long(),
    )
    e_reco_showers = e_reco_showers[1:]
    row_ind = torch.Tensor(row_ind).to(e_pred_showers.device).long()
    col_ind = torch.Tensor(col_ind).to(e_pred_showers.device).long()
    pred_showers = shower_p_unique
    energy_t = dic["part_true"][:, 3].to(e_pred_showers.device)

    pid_t = dic["part_true"][:, 6].to(e_pred_showers.device)
    index_matches = col_ind + 1
    index_matches = index_matches.to(e_pred_showers.device).long()
    matched_es = torch.zeros_like(energy_t) * (torch.nan)
    matched_es = matched_es.to(e_pred_showers.device)

    matched_es[row_ind] = e_pred_showers[index_matches]
    if pandora:
        matched_es_cali = matched_es.clone()
        matched_es_cali[row_ind] = e_pred_showers_cali[index_matches]
        matched_es_cali_pfo = matched_es.clone()
        matched_es_cali_pfo[row_ind] = e_pred_showers_pfo[index_matches]
    else:
        if e_corr is None:
            matched_es_cali = matched_es.clone()
            matched_es_cali[row_ind] = e_pred_showers_cali[index_matches]
            calibration_per_shower = matched_es.clone()
            calibration_per_shower[row_ind] = corrections_per_shower[index_matches]
        else:
            matched_es_cali = matched_es.clone()
            number_of_showers = e_pred_showers[index_matches].shape[0]
            matched_es_cali[row_ind] = (
                corrections_per_shower[
                    number_of_showers_total : number_of_showers_total
                    + number_of_showers
                ]
                * e_pred_showers[index_matches]
            )
            calibration_per_shower = matched_es.clone()
            calibration_per_shower[row_ind] = corrections_per_shower[
                number_of_showers_total : number_of_showers_total + number_of_showers
            ]
            number_of_showers_total = number_of_showers_total + number_of_showers

    intersection_E = torch.zeros_like(energy_t) * (torch.nan)
    ie_e = obtain_intersection_values(i_m_w, row_ind, col_ind)
    intersection_E[row_ind] = ie_e.to(e_pred_showers.device)

    pred_showers[index_matches] = -1
    pred_showers[
        0
    ] = (
        -1
    )  # this takes into account that the class 0 for pandora and for dbscan is noise
    mask = pred_showers != -1
    fake_showers_e = e_pred_showers[mask]
    if e_corr is None or pandora:
        fake_showers_e_cali = e_pred_showers_cali[mask]
    else:
        fake_showers_e_cali = e_pred_showers[mask] * (torch.nan)
    if not pandora:
        if e_corr is None:
            fake_showers_e_cali_factor = corrections_per_shower[mask]
        else:
            fake_showers_e_cali_factor = fake_showers_e_cali
    fake_showers_showers_e_truw = torch.zeros((fake_showers_e.shape[0])) * (torch.nan)
    fake_showers_showers_e_truw = fake_showers_showers_e_truw.to(e_pred_showers.device)

    energy_t = torch.cat(
        (energy_t, fake_showers_showers_e_truw),
        dim=0,
    )
    pid_t = torch.cat(
        (pid_t, fake_showers_showers_e_truw),
        dim=0,
    )
    e_reco = torch.cat((e_reco_showers, fake_showers_showers_e_truw), dim=0)
    e_pred = torch.cat((matched_es, fake_showers_e), dim=0)

    e_pred_cali = torch.cat((matched_es_cali, fake_showers_e_cali), dim=0)
    if pandora:
        e_pred_cali_pfo = torch.cat((matched_es_cali_pfo, fake_showers_e_cali), dim=0)
    if not pandora:
        calibration_factor = torch.cat(
            (calibration_per_shower, fake_showers_e_cali_factor), dim=0
        )

    e_pred_t = torch.cat(
        (
            intersection_E,
            torch.zeros_like(fake_showers_e) * (torch.nan),
        ),
        dim=0,
    )
    # e_pred_t_pandora = torch.cat(
    #     (
    #         intersection_E,
    #         torch.zeros_like(fake_showers_e) * (-200),
    #         torch.zeros_like(fake_showers_e_pandora) * (-100),
    #     ),
    #     dim=0,
    # )
    if pandora:
        d = {
            "true_showers_E": energy_t.detach().cpu(),
            "reco_showers_E": e_reco.detach().cpu(),
            "pred_showers_E": e_pred.detach().cpu(),
            "e_pred_and_truth": e_pred_t.detach().cpu(),
            "pandora_calibrated_E": e_pred_cali.detach().cpu(),
            "pandora_calibrated_pfo": e_pred_cali_pfo.detach().cpu(),
            "pid": pid_t.detach().cpu(),
            "step": torch.ones_like(energy_t.detach().cpu()) * step,
            "number_batch": torch.ones_like(energy_t.detach().cpu()) * number_in_batch,
        }
    else:
        d = {
            "true_showers_E": energy_t.detach().cpu(),
            "reco_showers_E": e_reco.detach().cpu(),
            "pred_showers_E": e_pred.detach().cpu(),
            "e_pred_and_truth": e_pred_t.detach().cpu(),
            "pid": pid_t.detach().cpu(),
            "calibration_factor": calibration_factor.detach().cpu(),
            "calibrated_E": e_pred_cali.detach().cpu(),
            "step": torch.ones_like(energy_t.detach().cpu()) * step,
            "number_batch": torch.ones_like(energy_t.detach().cpu()) * number_in_batch,
        }
    df = pd.DataFrame(data=d)
    if number_of_showers_total is None:
        return df
    else:
        return df, number_of_showers_total


def get_correction_per_shower(labels, dic):
    unique_labels = torch.unique(labels)
    list_corr = []
    for ii, pred_label in enumerate(unique_labels):
        if ii == 0:
            if pred_label != 0:
                list_corr.append(dic["graph"].ndata["correction"][0].view(-1) * 0)
        mask = labels == pred_label
        corrections_E_label = dic["graph"].ndata["correction"][mask]
        betas_label_indmax = torch.argmax(dic["graph"].ndata["beta"][mask])
        list_corr.append(corrections_E_label[betas_label_indmax].view(-1))

    corrections = torch.cat(list_corr, dim=0)
    return corrections


def obtain_intersection_matrix(shower_p_unique, particle_ids, labels, dic, e_hits):
    len_pred_showers = len(shower_p_unique)
    intersection_matrix = torch.zeros((len_pred_showers, len(particle_ids))).to(
        shower_p_unique.device
    )
    intersection_matrix_w = torch.zeros((len_pred_showers, len(particle_ids))).to(
        shower_p_unique.device
    )

    for index, id in enumerate(particle_ids):
        counts = torch.zeros_like(labels)
        mask_p = dic["graph"].ndata["particle_number"] == id
        h_hits = e_hits.clone()
        counts[mask_p] = 1
        h_hits[~mask_p] = 0
        intersection_matrix[:, index] = scatter_add(counts, labels)
        intersection_matrix_w[:, index] = scatter_add(h_hits, labels.to(h_hits.device))
    return intersection_matrix, intersection_matrix_w


def obtain_union_matrix(shower_p_unique, particle_ids, labels, dic):
    len_pred_showers = len(shower_p_unique)
    union_matrix = torch.zeros((len_pred_showers, len(particle_ids)))

    for index, id in enumerate(particle_ids):
        counts = torch.zeros_like(labels)
        mask_p = dic["graph"].ndata["particle_number"] == id
        for index_pred, id_pred in enumerate(shower_p_unique):
            mask_pred_p = labels == id_pred
            mask_union = mask_pred_p + mask_p
            union_matrix[index_pred, index] = torch.sum(mask_union)

    return union_matrix


def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=0.1, td=0.5):
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes torch.Tensors as input.
    """
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points)
    clustering = -1 * torch.ones(n_points, dtype=torch.long)
    for index_condpoint in indices_condpoints:
        d = torch.norm(X[unassigned] - X[index_condpoint][0], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint[0]
        unassigned = unassigned[~(d < td)]
    return clustering


def obtain_intersection_values(intersection_matrix_w, row_ind, col_ind):
    list_intersection_E = []
    # intersection_matrix_w = intersection_matrix_w
    intersection_matrix_wt = torch.transpose(intersection_matrix_w[1:, :], 1, 0)
    for i in range(0, len(col_ind)):
        list_intersection_E.append(
            intersection_matrix_wt[row_ind[i], col_ind[i]].view(-1)
        )
    return torch.cat(list_intersection_E, dim=0)


def plot_iou_matrix(iou_matrix, image_path, hdbscan=False):
    iou_matrix = torch.transpose(iou_matrix[1:, :], 1, 0)
    fig, ax = plt.subplots()
    iou_matrix = iou_matrix.detach().cpu().numpy()
    ax.matshow(iou_matrix, cmap=plt.cm.Blues)
    for i in range(0, iou_matrix.shape[1]):
        for j in range(0, iou_matrix.shape[0]):
            c = np.round(iou_matrix[j, i], 1)
            ax.text(i, j, str(c), va="center", ha="center")
    fig.savefig(image_path, bbox_inches="tight")
    if hdbscan:
        wandb.log({"iou_matrix_hdbscan": wandb.Image(image_path)})
    else:
        wandb.log({"iou_matrix": wandb.Image(image_path)})


def match_showers(
    labels,
    dic,
    particle_ids,
    model_output,
    local_rank,
    i,
    path_save,
    pandora=False,
    tracks=False,
    hdbscan=False,
):
    iou_threshold = 0.3
    shower_p_unique = torch.unique(labels)
    if torch.sum(labels == 0) == 0:
        shower_p_unique = torch.cat(
            (
                torch.Tensor([0]).to(shower_p_unique.device).view(-1),
                shower_p_unique.view(-1),
            ),
            dim=0,
        )
    e_hits = dic["graph"].ndata["e_hits"].view(-1)
    i_m, i_m_w = obtain_intersection_matrix(
        shower_p_unique, particle_ids, labels, dic, e_hits
    )
    i_m = i_m.to(model_output.device)
    i_m_w = i_m_w.to(model_output.device)
    u_m = obtain_union_matrix(shower_p_unique, particle_ids, labels, dic)
    u_m = u_m.to(model_output.device)
    iou_matrix = i_m / u_m
    iou_matrix_num = (
        torch.transpose(iou_matrix[1:, :], 1, 0).clone().detach().cpu().numpy()
    )
    iou_matrix_num[iou_matrix_num < iou_threshold] = 0
    row_ind, col_ind = linear_sum_assignment(-iou_matrix_num)
    # next three lines remove solutions where there is a shower that is not associated and iou it's zero (or less than threshold)
    mask_matching_matrix = iou_matrix_num[row_ind, col_ind] > 0
    row_ind = row_ind[mask_matching_matrix]
    col_ind = col_ind[mask_matching_matrix]
    if i == 0 and local_rank == 0:
        if path_save is not None:
            if pandora:
                image_path = path_save + "/example_1_clustering_pandora.png"
            else:
                image_path = path_save + "/example_1_clustering.png"
            # plot_iou_matrix(iou_matrix, image_path, hdbscan)
    # row_ind are particles that are matched and col_ind the ind of preds they are matched to
    return shower_p_unique, row_ind, col_ind, i_m_w


def hfdb_obtain_labels(X, device):
    hdb = HDBSCAN(min_cluster_size=8, min_samples=8, cluster_selection_epsilon=0.1).fit(
        X.detach().cpu()
    )
    labels_hdb = hdb.labels_ + 1
    labels_hdb = np.reshape(labels_hdb, (-1))
    labels_hdb = torch.Tensor(labels_hdb).long().to(device)
    return labels_hdb


def dbscan_obtain_labels(X, device):
    distance_scale = (
        (torch.min(torch.abs(torch.min(X, dim=0)[0] - torch.max(X, dim=0)[0])) / 30)
        .view(-1)
        .detach()
        .cpu()
        .numpy()[0]
    )

    db = DBSCAN(eps=distance_scale, min_samples=15).fit(X.detach().cpu())
    # DBSCAN has clustering labels -1,0,.., our cluster 0 is noise so we add 1
    labels = db.labels_ + 1
    labels = np.reshape(labels, (-1))
    labels = torch.Tensor(labels).long().to(device)
    return labels


def get_labels_pandora(tracks, dic, device):
    if tracks:
        labels_pandora = dic["graph"].ndata["pandora_pfo"].long()
    else:
        labels_pandora = dic["graph"].ndata["pandora_cluster"].long()
    labels_pandora = labels_pandora + 1
    map_from = list(np.unique(labels_pandora.detach().cpu()))
    cluster_id = map(lambda x: map_from.index(x), labels_pandora.detach().cpu())
    labels_pandora = torch.Tensor(list(cluster_id)).long().to(device)
    return labels_pandora
