import numpy as np
import torch
import dgl
from torch_scatter import scatter_add, scatter_sum
from sklearn.preprocessing import StandardScaler
from torch_scatter import scatter_sum
from src.dataset.functions_data import (
    get_ratios,
    find_mask_no_energy,
    find_cluster_id,
    get_particle_features,
    get_hit_features,
)


def create_inputs_from_table(output, hits_only, prediction=False):
    number_hits = np.int32(np.sum(output["pf_mask"][0]))
    number_part = np.int32(np.sum(output["pf_mask"][1]))

    (
        pos_xyz_hits,
        p_hits,
        e_hits,
        hit_type_one_hot,
        hit_particle_link,
        pandora_cluster,
        pandora_cluster_energy,
        pfo_energy,
        unique_list_particles,
        cluster_id,
        hit_type_feature,
    ) = get_hit_features(output, number_hits, prediction)

    # features particles
    y_data_graph = get_particle_features(unique_list_particles, output, prediction)

    assert len(y_data_graph) == len(unique_list_particles)
    # old_cluster_id = cluster_id
    mask_hits, mask_particles = find_mask_no_energy(
        cluster_id, hit_type_feature, e_hits, y_data_graph, prediction
    )
    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link[~mask_hits])

    result = [
        y_data_graph[~mask_particles],
        hit_type_one_hot[~mask_hits],  # [no_tracks],
        p_hits[~mask_hits],  # [no_tracks],
        e_hits[~mask_hits],  # [no_tracks],
        cluster_id,
        hit_particle_link[~mask_hits],
        pos_xyz_hits[~mask_hits],
        pandora_cluster[~mask_hits],
        pandora_cluster_energy[~mask_hits],
        pfo_energy[~mask_hits],
    ]
    hit_type = result[1].argmax(dim=1)
    if hits_only:
        hit_mask = (hit_type == 0) | (hit_type == 1)
        hit_mask = ~hit_mask
        for i in range(1, len(result)):
            result[i] = result[i][hit_mask]

    return result


def create_graph_synthetic(config, n_noise=0, npart_min=3, npart_max=5):
    num_hits_per_particle_min, num_hits_per_particle_max = 5, 60
    num_part = np.random.randint(npart_min, npart_max)
    num_hits_per_particle = np.random.randint(
        num_hits_per_particle_min, num_hits_per_particle_max, size=(num_part,)
    )
    # create a synthetic graph - random hits uniformly between -4 and 4, distribution of hits is gaussian
    y_coords = torch.zeros((num_part, 3)).float()
    # uniformly picked x,y,z coords saved in y_coords
    y_coords[:, 0] = torch.rand((num_part)).float() * 8 - 4
    y_coords[:, 1] = torch.rand((num_part)).float() * 8 - 4
    y_coords[:, 2] = torch.rand((num_part)).float() * 8 - 4
    nh = np.sum(num_hits_per_particle)
    graph_coordinates = torch.zeros((nh, 3)).float()
    hit_type_one_hot = torch.zeros((nh, 4)).float()
    e_hits = torch.zeros((nh, 1)).float() + 1.0
    p_hits = torch.zeros((nh, 1)).float() + 1.0  # to avoid nans
    for i in range(num_part):
        index = np.sum(num_hits_per_particle[:i])
        graph_coordinates[index : index + num_hits_per_particle[i]] = (
            torch.randn((num_hits_per_particle[i], 3)).float()
            * torch.tensor([0.12, 0.5, 0.4])
            + y_coords[i]
        )
        hit_type_one_hot[index : index + num_hits_per_particle[i], 3] = 1.0
    g = dgl.knn_graph(
        graph_coordinates, config.graph_config.get("k", 7), exclude_self=True
    )
    i, j = g.edges()
    edge_attr = torch.norm(
        graph_coordinates[i] - graph_coordinates[j], p=2, dim=1
    ).view(-1, 1)
    hit_features_graph = torch.cat(
        (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
    )
    hit_particle_link = torch.zeros((nh, 1)).float()
    for i in range(num_part):
        index = np.sum(num_hits_per_particle[:i])
        hit_particle_link[index : index + num_hits_per_particle[i]] = (
            i + 1
        )  # 0 is for noise
    if n_noise > 0:
        noise = torch.zeros((p_hits.shape[0], n_noise)).float()
        noise.normal_(mean=0, std=1)
        hit_features_graph = torch.cat(
            (graph_coordinates, hit_type_one_hot, e_hits, p_hits, noise), dim=1
        )
    g.ndata["h"] = hit_features_graph
    g.ndata["pos_hits"] = graph_coordinates
    g.ndata["pos_hits_xyz"] = graph_coordinates
    g.ndata["pos_hits_norm"] = graph_coordinates
    g.ndata["hit_type"] = hit_type_one_hot
    g.ndata["p_hits"] = p_hits
    g.ndata["e_hits"] = e_hits
    g.ndata["particle_number"] = hit_particle_link
    g.ndata["particle_number_nomap"] = hit_particle_link
    g.edata["h"] = edge_attr

    y_data_graph = torch.cat(
        (
            y_coords,
            torch.zeros((num_part, 4)).float(),
        ),
        dim=1,
    )
    return [g, y_data_graph], False


def to_hetero(g, all_hit_types=[2, 3]):
    # Convert the dgl graph object to a heterograph
    # We probably won't be using this
    hit_types = g.ndata["hit_type"]
    hit_types = torch.argmax(hit_types, dim=1)
    ht_idx = [all_hit_types.index(i) for i in hit_types]
    edges = g.edges()
    graph_data = {}
    for i in all_hit_types:
        for j in all_hit_types:
            edge_mask = hit_types[edges[0]] == i
            edge_mask = edge_mask & (hit_types[edges[1]] == j)
            graph_data[(str(i), "-", str(j))] = (
                edges[0][edge_mask],
                edges[1][edge_mask],
            )
    old_g = g
    g = dgl.heterograph(graph_data)
    g.nodes["2"].data = {key: old_g.ndata[key][ht_idx == 2] for key in old_g.ndata}
    g.nodes["3"].data = {key: old_g.ndata[key][ht_idx == 3] for key in old_g.ndata}
    return g


def create_graph(
    output,
    config=None,
    n_noise=0,
):
    hits_only = config.graph_config.get(
        "only_hits", False
    )  # Whether to only include hits in the graph
    # standardize_coords = config.graph_config.get("standardize_coords", False)
    extended_coords = config.graph_config.get("extended_coords", False)
    prediction = config.graph_config.get("prediction", False)
    (
        y_data_graph,
        hit_type_one_hot,
        p_hits,
        e_hits,
        cluster_id,
        hit_particle_link,
        pos_xyz_hits,
        pandora_cluster,
        pandora_cluster_energy,
        pandora_pfo_energy,
    ) = create_inputs_from_table(output, hits_only=hits_only, prediction=prediction)
    graph_coordinates = pos_xyz_hits  # / 3330  # divide by detector size

    if pos_xyz_hits.shape[0] > 0:
        graph_empty = False
        g = dgl.DGLGraph()
        g.add_nodes(graph_coordinates.shape[0])
        if extended_coords:
            hit_features_graph = torch.cat(
                (
                    graph_coordinates,
                    hit_type_one_hot,
                    e_hits,
                    p_hits,
                    torch.log(e_hits),
                ),
                dim=1,
            )  # dims = 3+4+1+1+1+1
        else:
            hit_features_graph = torch.cat(
                (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
            )  # dims = 9
        #! currently we are not doing the pid or mass regression
        g.ndata["h"] = hit_features_graph
        # g.ndata["pos_hits"] = coord_cart_hits
        g.ndata["pos_hits_xyz"] = pos_xyz_hits
        # g.ndata["pos_hits_norm"] = coord_cart_hits_norm
        g.ndata["hit_type"] = hit_type_one_hot
        g.ndata["p_hits"] = p_hits
        g.ndata["e_hits"] = e_hits
        g.ndata["particle_number"] = cluster_id
        g.ndata["particle_number_nomap"] = hit_particle_link
        # g.ndata["theta_hits"] = theta_hits
        # g.ndata["phi_hits"] = phi_hits
        g.ndata["pandora_cluster"] = pandora_cluster
        if prediction:
            g.ndata["pandora_cluster_energy"] = pandora_cluster_energy
            g.ndata["pandora_pfo_energy"] = pandora_pfo_energy
        if len(y_data_graph) < 4:
            graph_empty = True
    else:
        graph_empty = True
        g = 0
        y_data_graph = 0
    if pos_xyz_hits.shape[0] < 10:
        graph_empty = True

    return [g, y_data_graph], graph_empty


def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    list_graphs_g = [el[0] for el in list_graphs]
    list_y = add_batch_number(list_graphs)
    ys = torch.cat(list_y, dim=0)
    ys = torch.reshape(ys, [-1, list_y[0].shape[1]])
    bg = dgl.batch(list_graphs_g)
    # reindex particle number
    return bg, ys


def add_batch_number(list_graphs):
    list_y = []
    for i, el in enumerate(list_graphs):
        y = el[1]
        batch_id = torch.ones(y.shape[0], 1) * i
        y = torch.cat((y, batch_id), dim=1)
        list_y.append(y)
    return list_y
