from typing import Tuple, Union
import numpy as np
import torch
from torch_scatter import scatter_max, scatter_add, scatter_mean
from src.layers.object_cond import assert_no_nans, scatter_count, batch_cluster_indices
import dgl


def calc_energy_loss(
    batch, cluster_space_coords, beta, beta_stabilizing="soft_q_scaling", qmin=0.1
):
    list_graphs = dgl.unbatch(batch)
    node_counter = 0
    if beta_stabilizing == "paper":
        q = beta.arctanh() ** 2 + qmin
    elif beta_stabilizing == "clip":
        beta = beta.clip(0.0, 1 - 1e-4)
        q = beta.arctanh() ** 2 + qmin
    elif beta_stabilizing == "soft_q_scaling":
        q = (beta.clip(0.0, 1 - 1e-4) / 1.002).arctanh() ** 2 + qmin
    else:
        raise ValueError(f"beta_stablizing mode {beta_stabilizing} is not known")

    loss_E_frac = []
    loss_E_frac_true = []
    for g in list_graphs:
        particle_id = g.ndata["particle_number"]
        number_of_objects = len(particle_id.unique())
        non = g.number_of_nodes()
        q_g = q[node_counter : non + node_counter]
        betas = beta[node_counter : non + node_counter]
        sorted, indices = torch.sort(q_g, descending=False)
        selected_centers = indices[0:number_of_objects]
        if len((particle_id[selected_centers]).unique()) < number_of_objects:
            print("there are two or more clusters for one GT object")
            print("objects have ids:", particle_id[selected_centers])
            print("there are", number_of_objects, "objects")
        X = cluster_space_coords[node_counter : non + node_counter]
        clusterings = get_clustering(selected_centers, X, betas, td=0.7)
        clusterings = clusterings.to(g.device)
        counter = 0
        frac_energy = []
        frac_energy_true = []
        # for each clustering find the percentage of energy that has been assigned (this could contain particles from the wrong cluster)
        for alpha in indices:
            id_particle = particle_id[alpha]
            true_mask_particle = particle_id == id_particle
            true_energy = torch.sum(g.ndata["e_hits"][true_mask_particle])
            mask_clustering_particle = clusterings == counter
            clustered_energy = torch.sum(g.ndata["e_hits"][mask_clustering_particle])
            clustered_energy_true = torch.sum(
                g.ndata["e_hits"][mask_clustering_particle * true_mask_particle]
            )  # only consider how much has been correctly assigned
            frac_energy.append(clustered_energy / (true_energy + 1e-7))
            frac_energy_true.append(clustered_energy_true / (true_energy + 1e-7))
        frac_energy = torch.stack(frac_energy, dim=0)
        frac_energy = torch.mean(frac_energy)
        frac_energy_true = torch.stack(frac_energy_true, dim=0)
        frac_energy_true = torch.mean(frac_energy_true)
        loss_E_frac.append(frac_energy)
        loss_E_frac_true.append(frac_energy_true)

    loss_E_frac = torch.mean(torch.stack(loss_E_frac, dim=0))
    loss_E_frac_true = torch.mean(torch.stack(loss_E_frac_true, dim=0))
    return loss_E_frac, loss_E_frac_true


def get_clustering(index_alpha_i, X, betas, td=0.7):
    n_points = betas.size(0)
    unassigned = torch.arange(n_points).to(betas.device)
    clustering = -1 * torch.ones(n_points, dtype=torch.long)
    counter = 0
    for index_condpoint in index_alpha_i:
        d = torch.norm(X[unassigned] - X[index_condpoint], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = counter
        unassigned = unassigned[~(d < td)]
        counter = counter + 1
    counter = 0
    for index_condpoint in index_alpha_i:
        clustering[index_condpoint] = counter
        counter = counter + 1

    return clustering
