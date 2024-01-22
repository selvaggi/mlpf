import numpy as np
import torch
import dgl
from torch_scatter import scatter_add, scatter_sum
from sklearn.preprocessing import StandardScaler

from torch_scatter import scatter_sum


def get_number_hits(e_hits, part_idx):
    number_of_hits = scatter_sum(torch.ones_like(e_hits), part_idx.long(), dim=0)
    return (number_of_hits[1:].flatten()).tolist()


def find_cluster_id(hit_particle_link):
    unique_list_particles = list(np.unique(hit_particle_link))
    if np.sum(np.array(unique_list_particles) == -1) > 0:
        non_noise_idx = torch.where(unique_list_particles != -1)[0]
        noise_idx = torch.where(unique_list_particles == -1)[0]
        non_noise_particles = unique_list_particles[non_noise_idx]
        cluster_id = map(lambda x: non_noise_particles.index(x), hit_particle_link)
        cluster_id = torch.Tensor(list(cluster_id)) + 1
        unique_list_particles[non_noise_idx] = cluster_id
        unique_list_particles[noise_idx] = 0
    else:
        cluster_id = map(lambda x: unique_list_particles.index(x), hit_particle_link)
        cluster_id = torch.Tensor(list(cluster_id)) + 1
    return cluster_id, unique_list_particles


def scatter_count(input: torch.Tensor):
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


def create_inputs_from_table(output):
    number_hits = np.int32(np.sum(output["pf_mask"][0]))
    # print("number_hits", number_hits)
    number_part = np.int32(np.sum(output["pf_mask"][1]))
    #! idx of particle does not start at 1
    hit_particle_link = torch.tensor(output["pf_vectoronly"][0, 0:number_hits])

    features_hits = torch.permute(
        torch.tensor(output["pf_features"][:, 0:number_hits]), (1, 0)
    )
    hit_type = features_hits[:, -1].clone()
    mask_DC = hit_type == 0
    hit_type_one_hot = torch.nn.functional.one_hot(hit_type.long(), num_classes=2)
    hit_type_one_hot = hit_type_one_hot[mask_DC]
    features_hits = features_hits[mask_DC]
    hit_particle_link = hit_particle_link[mask_DC]

    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)

    # features particles
    unique_list_particles = torch.Tensor(unique_list_particles).to(torch.int64)

    features_particles = torch.permute(
        torch.tensor(output["pf_vectors"][:, list(unique_list_particles)]),
        (1, 0),
    )

    # y_mass = features_particles[:, 3].view(-1).unsqueeze(1)
    # y_mom = features_particles[:, 2].view(-1).unsqueeze(1)
    # y_energy = torch.sqrt(y_mass**2 + y_mom**2)

    y_data_graph = features_particles

    assert len(y_data_graph) == len(unique_list_particles)

    result = [
        number_hits,
        number_part,
        y_data_graph,
        hit_type_one_hot,  # [no_tracks],
        cluster_id,
        hit_particle_link,
        features_hits,
    ]
    return result


def create_graph_tracking(
    output,
):

    (
        number_hits,
        number_part,
        y_data_graph,
        hit_type_one_hot,  # [no_tracks],
        cluster_id,
        hit_particle_link,
        features_hits,
    ) = create_inputs_from_table(output)
    mask_not_loopers, mask_particles = remove_loopers(hit_particle_link, y_data_graph)
    hit_type_one_hot = hit_type_one_hot[mask_not_loopers]
    cluster_id = cluster_id[mask_not_loopers]
    hit_particle_link = hit_particle_link[mask_not_loopers]
    features_hits = features_hits[mask_not_loopers]
    y_data_graph = y_data_graph[mask_particles]
    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)
    if hit_type_one_hot.shape[0] > 0:
        graph_empty = False

        g = dgl.DGLGraph()
        g.add_nodes(hit_type_one_hot.shape[0])

        # hit_features_graph = torch.cat(
        #     (features_hits[:, 4:-1], hit_type_one_hot), dim=1
        # )  # dims = 7
        hit_features_graph = features_hits[:, 4:-1]
        uvz = convert_to_conformal_coordinates(features_hits[:, 0:3])
        polar = convert_to_polar_coordinates(uvz)
        hit_features_graph = torch.cat(
            (uvz, polar), dim=1
        )  # dim =8 #features_hits[:, 0:3],
        #! currently we are not doing the pid or mass regression
        g.ndata["h"] = hit_features_graph
        g.ndata["hit_type"] = hit_type_one_hot
        g.ndata["particle_number"] = cluster_id
        g.ndata["particle_number_nomap"] = hit_particle_link
        g.ndata["pos_hits_xyz"] = features_hits[:, 0:3]
        g.ndata["e_dep"] = features_hits[:, 3]
        if len(y_data_graph) < 4:
            graph_empty = True
    else:
        graph_empty = True
        g = 0
        y_data_graph = 0
    if features_hits.shape[0] < 10:
        graph_empty = True

    return [g, y_data_graph], graph_empty


def remove_loopers(
    hit_particle_link,
    y,
):
    unique_p_numbers = torch.unique(hit_particle_link)
    mask_p = y[:, 5] < 1
    list_remove = unique_p_numbers[mask_p.view(-1)]
    if len(list_remove) > 0:
        mask = torch.tensor(np.full((len(hit_particle_link)), False, dtype=bool))
        for p in list_remove:
            mask1 = hit_particle_link == p
            mask = mask1 + mask
    else:
        mask = np.full((len(hit_particle_link)), False, dtype=bool)
    list_p = unique_p_numbers
    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles
    else:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
    return ~mask.to(bool), ~mask_particles.to(bool)


def convert_to_conformal_coordinates(xyz):
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/polar.html
    x = xyz[:, 0]
    y = xyz[:, 1]
    u = x / (torch.square(x) + torch.square(y))
    v = y / (torch.square(x) + torch.square(y))
    uvz = torch.cat((u.view(-1, 1), v.view(-1, 1), xyz[:, 2].view(-1, 1)), dim=1)
    return uvz


def convert_to_polar_coordinates(uvz):
    cart = uvz[:, 0:2]
    rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)
    from math import pi as PI

    theta = torch.atan2(cart[:, 1], cart[:, 0]).view(-1, 1)
    theta = theta + (theta < 0).type_as(theta) * (2 * PI)
    rho = rho / (rho.max())
    # theta = theta / (2 * PI)

    polar = torch.cat([rho, theta], dim=-1)
    return polar
