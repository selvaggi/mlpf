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
    for i in range(0, len(graphs)):
        mask = batch_id == 0
        dic = {}
        dic["graph"] = graphs[i]
        dic["part_true"] = y[mask]
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

        particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
        shower_p_unique = torch.unique(labels)
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
        row_ind, col_ind = linear_sum_assignment(-iou_matrix_num)
        if i == 0 and local_rank == 0:
            image_path = path_save + "/example_1_clustering.png"
            plot_iou_matrix(iou_matrix, image_path)
        # row_ind are particles that are matched and col_ind the ind of preds they are matched to

        df_event = generate_showers_data_frame(
            labels, dic, shower_p_unique, particle_ids, row_ind, col_ind, i_m_w
        )
        df_list.append(df_event)

    df_batch = pd.concat(df_list)
    #
    if store:
        path_save = (
            path_save
            + "/"
            + str(local_rank)
            + "_"
            + str(step)
            + "_"
            + str(epoch)
            + ".pt"
        )
        df_batch.to_pickle(path_save)
    return df_batch


def log_efficiency(df):
    eff = np.sum(~np.isnan(df["pred_showers_E"].values)) / len(
        df["pred_showers_E"].values
    )
    wandb.log({"efficiency validation": eff})


def generate_showers_data_frame(
    labels, dic, shower_p_unique, particle_ids, row_ind, col_ind, i_m_w
):
    e_pred_showers = scatter_add(dic["graph"].ndata["e_hits"].view(-1), labels)
    row_ind = torch.Tensor(row_ind).to(e_pred_showers.device).long()
    col_ind = torch.Tensor(col_ind).to(e_pred_showers.device).long()
    pred_showers = shower_p_unique
    # true_showers = particle_ids
    # max_num_showers = torch.max(torch.Tensor([len(pred_showers), len(true_showers)]))

    # Add true showers (matched and unmatched)
    energy_t = dic["part_true"][:, 3].to(e_pred_showers.device)
    index_matches = col_ind + 1
    index_matches = index_matches.to(e_pred_showers.device).long()
    matched_es = torch.zeros_like(energy_t) * torch.nan
    matched_es = matched_es.to(e_pred_showers.device)
    print(
        matched_es.device, row_ind.device, e_pred_showers.device, index_matches.device
    )
    matched_es[row_ind] = e_pred_showers[index_matches]
    intersection_E = torch.zeros_like(energy_t) * torch.nan
    ie_e = obtain_intersection_values(i_m_w, row_ind, col_ind)
    intersection_E[row_ind] = ie_e.to(e_pred_showers.device)
    ## showers that are not in the true showers:
    pred_showers[index_matches] = -1
    mask = pred_showers != -1
    fake_showers_e = e_pred_showers[mask]
    fake_showers_showers_e_truw = torch.zeros((fake_showers_e.shape[0])) * torch.nan
    fake_showers_showers_e_truw = fake_showers_showers_e_truw.to(e_pred_showers.device)
    energy_t = torch.cat((energy_t, fake_showers_showers_e_truw), dim=0)
    e_pred = torch.cat((matched_es, fake_showers_e), dim=0)
    e_pred_t = torch.cat(
        (intersection_E, torch.zeros_like(fake_showers_e) * torch.nan), dim=0
    )
    d = {
        "true_showers_E": energy_t.detach().cpu(),
        "pred_showers_E": e_pred.detach().cpu(),
        "e_pred_and_truth": e_pred_t.detach().cpu(),
    }
    df = pd.DataFrame(data=d)

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
        intersection_matrix[:, index] = scatter_add(counts, labels)
        print(h_hits.device, labels.device)
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
    intersection_matrix_w = intersection_matrix_w.detach().cpu()
    intersection_matrix_wt = torch.transpose(intersection_matrix_w[1:, :], 1, 0)
    for i in range(len(col_ind)):
        list_intersection_E.append(
            intersection_matrix_wt[row_ind[i], col_ind[i]].view(-1)
        )
    return torch.cat(list_intersection_E, dim=0)


def plot_iou_matrix(iou_matrix, image_path):
    iou_matrix = torch.transpose(iou_matrix[1:, :], 1, 0)
    fig, ax = plt.subplots()
    iou_matrix = iou_matrix.detach().cpu().numpy()
    ax.matshow(iou_matrix, cmap=plt.cm.Blues)
    for i in range(0, iou_matrix.shape[1]):
        for j in range(0, iou_matrix.shape[0]):
            c = np.round(iou_matrix[j, i], 2)
            ax.text(i, j, str(c), va="center", ha="center")
    fig.savefig(image_path, bbox_inches="tight")
    wandb.log({"iou_matrix": wandb.Image(image_path)})
