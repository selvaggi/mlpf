import numpy as np
import torch
import dgl


def create_inputs_from_table(output):
    number_hits = np.int32(np.sum(output["pf_mask"][0]))
    number_part = np.int32(np.sum(output["pf_mask"][1]))

    #! idx of particle does not start at 1
    hit_particle_link = torch.tensor(output["pf_vectoronly"][0, 0:number_hits])
    unique_list_particles = list(np.unique(hit_particle_link))
    if np.sum(np.array(unique_list_particles) == -1) > 0:
        cluster_id = map(lambda x: unique_list_particles.index(x), hit_particle_link)
        cluster_id = torch.Tensor(list(cluster_id))
    else:
        cluster_id = map(lambda x: unique_list_particles.index(x), hit_particle_link)
        cluster_id = torch.Tensor(list(cluster_id)) + 1

    features_hits = torch.permute(
        torch.tensor(output["pf_vectors"][0:7, 0:number_hits]), (1, 0)
    )
    pos_hits = torch.permute(
        torch.tensor(output["pf_points"][:, 0:number_hits]), (1, 0)
    )
    hit_type_feature = features_hits[:, 0].to(torch.int64)
    tracks = hit_type_feature == 0
    no_tracks = ~tracks
    no_tracks[0] = True
    hit_type_one_hot = torch.nn.functional.one_hot(hit_type_feature, num_classes=4)
    # build the features (theta,phi,p)
    pf_features_hits = torch.permute(
        torch.tensor(output["pf_features"][0:4, 0:number_hits]), (1, 0)
    )
    p_hits = pf_features_hits[:, 2].unsqueeze(1)
    p_hits[p_hits == -1] = 0  # correct p  of Hcal hits to be 0
    e_hits = pf_features_hits[:, 3].unsqueeze(1)
    e_hits[e_hits == -1] = 0  # correct the energy of the tracks to be 0
    theta = pf_features_hits[:, 0]
    phi = pf_features_hits[:, 1]
    r = p_hits.view(-1)
    coord_cart_hits = spherical_to_cartesian(theta, phi, r, normalized=False)
    coord_cart_hits_norm = spherical_to_cartesian(theta, phi, r, normalized=True)

    # features particles
    unique_list_particles = torch.Tensor(unique_list_particles).to(torch.int64)
    features_particles = torch.permute(
        torch.tensor(output["pf_features"][4:7, list(unique_list_particles)]), (1, 0)
    )
    particle_coord = spherical_to_cartesian(
        features_particles[:, 0],
        features_particles[:, 1],
        features_particles[:, 2],
        normalized=True,
    )
    y_data_graph = torch.cat(
        (
            particle_coord,
            features_particles[:, 2].view(-1).unsqueeze(1),
        ),
        dim=1,
    )

    return (
        number_hits,
        number_part,
        y_data_graph,
        coord_cart_hits,  # [no_tracks],
        coord_cart_hits_norm,  # [no_tracks],
        hit_type_one_hot,  # [no_tracks],
        p_hits,  # [no_tracks],
        e_hits,  # [no_tracks],
        cluster_id,
    )


def create_graph(output):
    (
        number_hits,
        number_part,
        y_data_graph,
        coord_cart_hits,
        coord_cart_hits_norm,
        hit_type_one_hot,
        p_hits,
        e_hits,
        hit_particle_link,
    ) = create_inputs_from_table(output)
    # print("n hits:", number_hits, "number_part", number_part)
    # this builds fully connected graph
    # TODO build graph using the hit links (hit_particle_link) which assigns to each node the particle it belongs to
    # i, j = torch.tril_indices(number_hits, number_hits)
    # g = dgl.graph((i, j))
    # g = dgl.to_simple(g)
    # g = dgl.to_bidirected(g)
    g = dgl.knn_graph(coord_cart_hits, 7, exclude_self=True)
    hit_features_graph = torch.cat(
        (coord_cart_hits_norm, hit_type_one_hot, e_hits), dim=1
    )

    g.ndata["h"] = hit_features_graph
    g.ndata["pos_hits"] = coord_cart_hits
    g.ndata["pos_hits_norm"] = coord_cart_hits_norm
    g.ndata["hit_type"] = hit_type_one_hot
    g.ndata["p_hits"] = p_hits
    g.ndata["e_hits"] = e_hits
    g.ndata["particle_number"] = hit_particle_link
    return g, y_data_graph


def create_dif_interactions(i, j, pos, number_p):
    x_interactions = pos
    x_interactions = torch.reshape(x_interactions, [number_p, 1, 2])
    x_interactions = x_interactions.repeat(1, number_p, 1)
    xi = x_interactions[i, j, :]
    xj = x_interactions[j, i, :]
    x_interactions_m = xi - xj
    return x_interactions_m


def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    list_graphs_g = [el[0] for el in list_graphs]

    list_y = [el[1] for el in list_graphs]
    ys = torch.cat(list_y, dim=0)
    ys = torch.reshape(ys, [-1, 4])
    bg = dgl.batch(list_graphs_g)

    return bg, ys


def spherical_to_cartesian(theta, phi, r, normalized=False):
    if normalized:
        r = torch.ones_like(theta)
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)
