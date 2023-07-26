import numpy as np
import torch
import dgl


def find_mask_no_energy(hit_particle_link, hit_type_a):
    list_p = np.unique(hit_particle_link)
    list_remove = []
    for p in list_p:
        mask = hit_particle_link == p
        hit_types = np.unique(hit_type_a[mask])
        if np.array_equal(hit_types, [0, 1]):
            list_remove.append(p)
    if len(list_remove) > 0:
        mask = torch.tensor(np.full((len(hit_particle_link)), False, dtype=bool))
        for p in list_remove:
            mask1 = hit_particle_link == p
            mask = mask1 + mask

    else:
        mask = np.full((len(hit_particle_link)), False, dtype=bool)

    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles

    else:
        mask_particles = np.full((len(list_p)), False, dtype=bool)

    return mask, mask_particles


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

def create_inputs_from_table(output, hits_only):
    number_hits = np.int32(np.sum(output["pf_mask"][0]))
    number_part = np.int32(np.sum(output["pf_mask"][1]))
    #! idx of particle does not start at 1
    hit_particle_link = torch.tensor(output["pf_vectoronly"][0, 0:number_hits])

    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)
    features_hits = torch.permute(
        torch.tensor(output["pf_vectors"][0:7, 0:number_hits]), (1, 0)
    )
    # pos_hits = torch.permute(
    #     torch.tensor(output["pf_points"][:, 0:number_hits]), (1, 0)
    # )
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
        torch.tensor(output["pf_features"][4:9, list(unique_list_particles)]), (1, 0)
    )
    particle_coord = spherical_to_cartesian(
        features_particles[:, 0],
        features_particles[:, 1],
        features_particles[:, 2],
        normalized=True,
    )
    y_mass = features_particles[:, 3].view(-1).unsqueeze(1)
    y_mom = features_particles[:, 2].view(-1).unsqueeze(1)
    y_energy = torch.sqrt(y_mass**2 + y_mom**2)
    y_data_graph = torch.cat(
        (
            particle_coord,
            y_energy,
            y_mom,
            y_mass,
            features_particles[:, 4].view(-1).unsqueeze(1),  # particle ID (discrete)
        ),
        dim=1,
    )

    assert len(y_data_graph) == len(unique_list_particles)

    mask_hits, mask_particles = find_mask_no_energy(cluster_id, hit_type_feature)
    if hits_only:
        energy_hits_mask = no_tracks
        mask_hits = mask_hits & energy_hits_mask.numpy().flatten()
    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link[~mask_hits])

    return (
        number_hits,
        number_part,
        y_data_graph[~mask_particles],
        coord_cart_hits[~mask_hits],  # [no_tracks],
        coord_cart_hits_norm[~mask_hits],  # [no_tracks],
        hit_type_one_hot[~mask_hits],  # [no_tracks],
        p_hits[~mask_hits],  # [no_tracks],
        e_hits[~mask_hits],  # [no_tracks],
        cluster_id,
        hit_particle_link[~mask_hits],
    )


def create_graph(output, config=None):
    hits_only =  config.graph_config.get("only_hits", False) # whether to only include hits in the graph
    (
        number_hits,
        number_part,
        y_data_graph,
        coord_cart_hits,
        coord_cart_hits_norm,
        hit_type_one_hot,
        p_hits,
        e_hits,
        cluster_id,
        hit_particle_link,
    ) = create_inputs_from_table(output, hits_only=hits_only)
    # print("n hits:", number_hits, "number_part", number_part)
    # this builds fully connected graph
    # TODO build graph using the hit links (hit_particle_link) which assigns to each node the particle it belongs to
    # i, j = torch.tril_indices(number_hits, number_hits)
    # g = dgl.graph((i, j))
    # g = dgl.to_simple(g)
    # g = dgl.to_bidirected(g)
    if coord_cart_hits.shape[0] > 0:
        graph_empty = False
        g = dgl.knn_graph(coord_cart_hits_norm, 7, exclude_self=True)
        hit_features_graph = torch.cat(
            (coord_cart_hits_norm, hit_type_one_hot, e_hits, p_hits), dim=1
        )
        #! currently we are not doing the pid or mass regression
        g.ndata["h"] = hit_features_graph
        g.ndata["pos_hits"] = coord_cart_hits
        g.ndata["pos_hits_norm"] = coord_cart_hits_norm
        g.ndata["hit_type"] = hit_type_one_hot
        g.ndata["p_hits"] = p_hits
        g.ndata["e_hits"] = e_hits
        g.ndata["particle_number"] = cluster_id
        g.ndata["particle_number_nomap"] = hit_particle_link
    else:
        #print("graph empty")
        graph_empty = True
        g = 0
        y_data_graph = 0
    #print("found non-empty graph")
    return [g, y_data_graph], graph_empty


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
    ys = torch.reshape(ys, [-1, list_y[0].shape[1]])
    bg = dgl.batch(list_graphs_g)
    # reindex particle number
    return bg, ys


def spherical_to_cartesian(theta, phi, r, normalized=False):
    if normalized:
        r = torch.ones_like(theta)
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)
