import dgl
import torch
import os
from sklearn.cluster import DBSCAN
from torch_scatter import scatter_max, scatter_add, scatter_mean
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
import wandb


def create_and_store_graph_output(
    batch_g, model_output, y, local_rank, step, epoch, path_save, store=False
):
    batch_g.ndata["coords"] = model_output[:, 0:3]
    batch_g.ndata["beta"] = model_output[:, 3]
    graphs = dgl.unbatch(batch_g)
    batch_id = y[:, -1].view(-1)
    df_list = []
    df_list_pandora = []
    for i in range(0, len(graphs)):
        # print("llooking into graph,", i)
        mask = batch_id == i
        dic = {}
        dic["graph"] = graphs[i]
        dic["part_true"] = y[mask]
        # print("loaded graph and particles ", i)
        # print("STORING GRAPH")
        # torch.save(
        #     dic,
        #     path_save + "/" + str(local_rank) + "_" + str(step) + "_" + str(i) + ".pt",
        # )
        betas = torch.sigmoid(dic["graph"].ndata["beta"])
        X = dic["graph"].ndata["coords"]
        clustering_mode = "dbscan"
        if clustering_mode == "clustering_normal":
            clustering = get_clustering(betas, X)
        elif clustering_mode == "dbscan":
            distance_scale = (
                (
                    torch.min(
                        torch.abs(torch.min(X, dim=0)[0] - torch.max(X, dim=0)[0])
                    )
                    / 20
                )
                .view(-1)
                .detach()
                .cpu()
                .numpy()[0]
            )
            db = DBSCAN(eps=distance_scale, min_samples=100).fit(X.detach().cpu())
            labels = db.labels_ + 1
            labels = np.reshape(labels, (-1))
            labels = torch.Tensor(labels).long().to(model_output.device)

        labels_pandora = dic["graph"].ndata["pandora_cluster"].long()
        labels_pandora[labels_pandora == -1] = 0
        # print("obtained clustering ")
        particle_ids = torch.unique(dic["graph"].ndata["particle_number"])

        shower_p_unique, row_ind, col_ind, i_m_w = match_showers(
            labels, dic, particle_ids, model_output, local_rank, i, path_save
        )

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
        )

        df_event = generate_showers_data_frame(
            labels, dic, shower_p_unique, particle_ids, row_ind, col_ind, i_m_w
        )
        # print("past dataframe generation")
        df_list.append(df_event)
        df_event_pandora = generate_showers_data_frame(
            labels_pandora,
            dic,
            shower_p_unique_pandora,
            particle_ids,
            row_ind_pandora,
            col_ind_pandora,
            i_m_w_pandora,
        )
        df_list_pandora.append(df_event_pandora)

    # print("concatenating list")
    df_batch = pd.concat(df_list)

    df_batch_pandora = pd.concat(df_list_pandora)
    #
    if store:
        store_at_batch_end(path_save, df_list, df_list_pandora, local_rank, step, epoch)
    return df_batch, df_batch_pandora


def store_at_batch_end(
    path_save, df_batch, df_batch_pandora, local_rank=0, step=0, epoch=None
):
    path_save_ = (
        path_save + "/" + str(local_rank) + "_" + str(step) + "_" + str(epoch) + ".pt"
    )
    df_batch.to_pickle(path_save_)
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
    log_efficiency(df_batch_pandora, pandora=True)


def log_efficiency(df, pandora=False):
    eff = np.sum(~np.isnan(df["pred_showers_E"].values)) / len(
        df["pred_showers_E"].values
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
    # labels_pandora,
    # shower_p_unique_pandora,
    # row_ind_pandora,
    # col_ind_pandora,
    # i_m_w_pandora,
):
    e_pred_showers = scatter_add(dic["graph"].ndata["e_hits"].view(-1), labels)
    row_ind = torch.Tensor(row_ind).to(e_pred_showers.device).long()
    col_ind = torch.Tensor(col_ind).to(e_pred_showers.device).long()
    pred_showers = shower_p_unique

    # e_pred_showers_pandora = scatter_add(
    #     dic["graph"].ndata["e_hits"].view(-1), labels_pandora
    # )
    # row_ind_pandora = torch.Tensor(row_ind_pandora).to(e_pred_showers.device).long()
    # col_ind_pandora = torch.Tensor(col_ind_pandora).to(e_pred_showers.device).long()
    # pred_showers_pandora = shower_p_unique_pandora

    # Add true showers (matched and unmatched)
    energy_t = dic["part_true"][:, 3].to(e_pred_showers.device)
    index_matches = col_ind + 1
    index_matches = index_matches.to(e_pred_showers.device).long()
    matched_es = torch.zeros_like(energy_t) * (torch.nan)
    matched_es = matched_es.to(e_pred_showers.device)

    # index_matches_pandora = col_ind_pandora + 1
    # index_matches_pandora = index_matches_pandora.to(e_pred_showers.device).long()
    # matched_es_pandora = torch.zeros_like(energy_t) * (-100)
    # matched_es_pandora = matched_es_pandora.to(e_pred_showers.device)

    matched_es[row_ind] = e_pred_showers[index_matches]
    # matched_es_pandora[row_ind_pandora] = e_pred_showers_pandora[index_matches_pandora]

    intersection_E = torch.zeros_like(energy_t) * (torch.nan)
    ie_e = obtain_intersection_values(i_m_w, row_ind, col_ind)
    intersection_E[row_ind] = ie_e.to(e_pred_showers.device)

    # intersection_E_pandora = torch.zeros_like(energy_t) * (-100)
    # ie_e_pandora = obtain_intersection_values(
    #     i_m_w_pandora, row_ind_pandora, col_ind_pandora
    # )
    # intersection_E_pandora[row_ind_pandora] = ie_e_pandora.to(e_pred_showers.device)

    # pred_showers_pandora[index_matches_pandora] = -1
    # mask = pred_showers_pandora != -1
    # fake_showers_e_pandora = pred_showers_pandora[mask]
    # fake_showers_showers_e_truw_pandora = torch.zeros(
    #     (fake_showers_e_pandora.shape[0])
    # ) * (-100)
    # fake_showers_showers_e_truw_pandora = fake_showers_showers_e_truw_pandora.to(
    #     e_pred_showers.device
    # )

    pred_showers[index_matches] = -1
    mask = pred_showers != -1
    fake_showers_e = e_pred_showers[mask]
    fake_showers_showers_e_truw = torch.zeros((fake_showers_e.shape[0])) * (torch.nan)
    fake_showers_showers_e_truw = fake_showers_showers_e_truw.to(e_pred_showers.device)

    energy_t = torch.cat(
        (energy_t, fake_showers_showers_e_truw),
        dim=0,
    )
    e_pred = torch.cat((matched_es, fake_showers_e), dim=0)
    # e_pred_pandora = torch.cat(
    #     (matched_es_pandora, fake_showers_showers_e_truw), dim=0
    # )
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
    # print("here9")
    d = {
        "true_showers_E": energy_t.detach().cpu(),
        "pred_showers_E": e_pred.detach().cpu(),
        "e_pred_and_truth": e_pred_t.detach().cpu(),
        # "pred_showers_E_pandora": e_pred_pandora.detach().cpu(),
        # "e_pred_and_truth_pandora": e_pred_t_pandora.detach().cpu(),
    }
    df = pd.DataFrame(data=d)
    # print("here10 finished")
    return df


def obtain_intersection_matrix(shower_p_unique, particle_ids, labels, dic, e_hits):
    intersection_matrix = torch.zeros((len(shower_p_unique), len(particle_ids))).to(
        shower_p_unique.device
    )
    intersection_matrix_w = torch.zeros((len(shower_p_unique), len(particle_ids))).to(
        shower_p_unique.device
    )

    for index, id in enumerate(particle_ids):
        counts = torch.zeros_like(labels)
        mask_p = dic["graph"].ndata["particle_number"] == id
        h_hits = e_hits.clone()
        counts[mask_p] = 1
        h_hits[~mask_p] = 0
        print(
            counts.shape,
            labels.shape,
            intersection_matrix.shape,
            scatter_add(counts, labels).shape,
        )
        intersection_matrix[:, index] = scatter_add(counts, labels)
        # print(h_hits.device, labels.device)
        intersection_matrix_w[:, index] = scatter_add(h_hits, labels.to(h_hits.device))
    return intersection_matrix, intersection_matrix_w


def obtain_union_matrix(shower_p_unique, particle_ids, labels, dic):
    union_matrix = torch.zeros((len(shower_p_unique), len(particle_ids)))

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
    # print(row_ind)
    # print(col_ind)
    # print(intersection_matrix_wt.shape)
    # print(range(0, len(col_ind) - 1))
    for i in range(0, len(col_ind)):
        # print("i", i)
        # print(row_ind[i], col_ind[i])
        # print(intersection_matrix_wt[row_ind[i], col_ind[i]])
        list_intersection_E.append(
            intersection_matrix_wt[row_ind[i], col_ind[i]].view(-1)
        )
    # print("finized list")
    return torch.cat(list_intersection_E, dim=0)


def plot_iou_matrix(iou_matrix, image_path):
    iou_matrix = torch.transpose(iou_matrix[1:, :], 1, 0)
    fig, ax = plt.subplots()
    iou_matrix = iou_matrix.detach().cpu().numpy()
    ax.matshow(iou_matrix, cmap=plt.cm.Blues)
    for i in range(0, iou_matrix.shape[1]):
        for j in range(0, iou_matrix.shape[0]):
            c = np.round(iou_matrix[j, i], 1)
            ax.text(i, j, str(c), va="center", ha="center")
    fig.savefig(image_path, bbox_inches="tight")
    wandb.log({"iou_matrix": wandb.Image(image_path)})


def match_showers(
    labels, dic, particle_ids, model_output, local_rank, i, path_save, pandora=False
):
    shower_p_unique = torch.unique(labels)
    print("showers p unique", shower_p_unique.shape, pandora)
    print(shower_p_unique)
    e_hits = dic["graph"].ndata["e_hits"].view(-1)
    # print("asking for intersection matrix  ")
    i_m, i_m_w = obtain_intersection_matrix(
        shower_p_unique, particle_ids, labels, dic, e_hits
    )
    # print("got intersection matrix  ")
    i_m = i_m.to(model_output.device)
    i_m_w = i_m_w.to(model_output.device)
    u_m = obtain_union_matrix(shower_p_unique, particle_ids, labels, dic)
    # print("got union matrix  ")
    u_m = u_m.to(model_output.device)
    iou_matrix = i_m / u_m
    iou_matrix_num = (
        torch.transpose(iou_matrix[1:, :], 1, 0).clone().detach().cpu().numpy()
    )
    # print("askind for LSA")
    row_ind, col_ind = linear_sum_assignment(-iou_matrix_num)
    # print("got LSA matrix  ")
    if i == 0 and local_rank == 0:
        if pandora:
            image_path = path_save + "/example_1_clustering_pandora.png"
        else:
            image_path = path_save + "/example_1_clustering.png"
        plot_iou_matrix(iou_matrix, image_path)
    # row_ind are particles that are matched and col_ind the ind of preds they are matched to
    return shower_p_unique, row_ind, col_ind, i_m_w
