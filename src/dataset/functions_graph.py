import numpy as np
import torch
import dgl
from torch_scatter import scatter_add, scatter_sum
from sklearn.preprocessing import StandardScaler

from torch_scatter import scatter_sum


def get_ratios(e_hits, part_idx, y):
    """Obtain the percentage of energy of the particle present in the hits

    Args:
        e_hits (_type_): _description_
        part_idx (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    energy_from_showers = scatter_sum(e_hits, part_idx.long(), dim=0)
    y_energy = y[:, 3]
    energy_from_showers = energy_from_showers[1:]
    assert len(energy_from_showers) > 0
    return (energy_from_showers.flatten() / y_energy).tolist()


def get_number_hits(e_hits, part_idx):
    number_of_hits = scatter_sum(torch.ones_like(e_hits), part_idx.long(), dim=0)
    return (number_of_hits.flatten()).tolist()


def find_mask_no_energy(hit_particle_link, hit_type_a, hit_energies, y):
    """This function remove particles with tracks only and remove particles with low fractions

    Args:
        hit_particle_link (_type_): _description_
        hit_type_a (_type_): _description_
        hit_energies (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    energy_cut = 0.50
    # REMOVE THE WEIRD ONES
    list_p = np.unique(hit_particle_link)
    # print(list_p)
    list_remove = []
    part_frac = torch.tensor(get_ratios(hit_energies, hit_particle_link, y))
    number_of_hits = get_number_hits(hit_energies, hit_particle_link)
    # print(part_frac)
    filt1 = (
        (torch.where(part_frac >= energy_cut)[0] + 1).long().tolist()
    )  # only keep these particles

    for index, p in enumerate(list_p):
        mask = hit_particle_link == p
        hit_types = np.unique(hit_type_a[mask])
        # if np.array_equal(hit_types, [0, 1]):
        #     print("will remove particle", p)
        if (
            np.array_equal(hit_types, [0, 1])
            or int(p) not in filt1
            or (number_of_hits[index] < 20)
        ):  # This is commented to disable filtering
            list_remove.append(p)
            # print(
            #     "percentage of energy, number of hits",
            #     part_frac[int(p) - 1],
            #     number_of_hits[index],
            #     y[index, 3],
            # )
            # assert part_frac[int(p) - 1] <= energy_cut

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
    # print(
    #     "removing",
    #     np.sum(mask_particles),
    #     "particles out of ",
    #     len(list_p),
    #     np.sum(mask_particles) / len(list_p),
    # )
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


def scatter_count(input: torch.Tensor):
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


def create_inputs_from_table(output, hits_only):
    number_hits = np.int32(np.sum(output["pf_mask"][0]))
    # print("number_hits", number_hits)
    number_part = np.int32(np.sum(output["pf_mask"][1]))
    #! idx of particle does not start at 1
    hit_particle_link = torch.tensor(output["pf_vectoronly"][0, 0:number_hits])
    pandora_cluster = torch.tensor(output["pf_vectoronly"][1, 0:number_hits])
    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)
    features_hits = torch.permute(
        torch.tensor(output["pf_vectors"][0:7, 0:number_hits]), (1, 0)
    )
    pos_hits = torch.permute(
        torch.tensor(output["pf_points"][0:3, 0:number_hits]), (1, 0)
    )
    hit_type_feature = features_hits[:, 0].to(torch.int64)
    tracks = (hit_type_feature == 0) | (hit_type_feature == 1)
    # no_tracks = ~tracks
    # no_tracks[0] = True
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
    pos_xyz_hits = pos_hits
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
    # old_cluster_id = cluster_id
    mask_hits, mask_particles = find_mask_no_energy(
        cluster_id, hit_type_feature, e_hits, y_data_graph
    )
    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link[~mask_hits])

    result = [
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
        pos_xyz_hits[~mask_hits],
        theta[~mask_hits],
        phi[~mask_hits],
        pandora_cluster[~mask_hits],
    ]
    hit_type = result[5].argmax(dim=1)
    if hits_only:
        hit_mask = (hit_type == 0) | (hit_type == 1)
        hit_mask = ~hit_mask
        result[0] = hit_mask.sum()
        for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            result[i] = result[i][hit_mask]

    return result


def standardize_coordinates(coord_cart_hits):
    if len(coord_cart_hits) == 0:
        return coord_cart_hits, None
    std_scaler = StandardScaler()
    coord_cart_hits = std_scaler.fit_transform(coord_cart_hits)
    return torch.tensor(coord_cart_hits).float(), std_scaler


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


def create_graph(output, config=None, n_noise=0):
    hits_only = config.graph_config.get(
        "only_hits", False
    )  # Whether to only include hits in the graph
    standardize_coords = config.graph_config.get("standardize_coords", False)
    extended_coords = config.graph_config.get("extended_coords", False)
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
        pos_xyz_hits,
        theta_hits,
        phi_hits,
        pandora_cluster,
    ) = create_inputs_from_table(output, hits_only=hits_only)
    pos_xyz_hits = pos_xyz_hits  # / 3330  # divide by detector size
    if standardize_coords:
        # Standardize the coordinates of the hits
        coord_cart_hits, scaler = standardize_coordinates(coord_cart_hits)
        coord_cart_hits_norm, scaler_norm = standardize_coordinates(
            coord_cart_hits_norm
        )
        pos_xyz_hits, scaler_norm_xyz = standardize_coordinates(pos_xyz_hits)
        if scaler_norm is not None:
            y_coords_std = scaler_norm.transform(y_data_graph[:, :3])
            y_data_graph[:, :3] = torch.tensor(y_coords_std).float()

    graph_coordinates = pos_xyz_hits
    graph_coordinates_norm = torch.norm(pos_xyz_hits, dim=1, p=2)

    if coord_cart_hits.shape[0] > 0:
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
        g.ndata["pos_hits"] = coord_cart_hits
        g.ndata["pos_hits_xyz"] = pos_xyz_hits
        g.ndata["pos_hits_norm"] = coord_cart_hits_norm
        g.ndata["hit_type"] = hit_type_one_hot
        g.ndata["p_hits"] = p_hits
        g.ndata["e_hits"] = e_hits
        g.ndata["particle_number"] = cluster_id
        g.ndata["particle_number_nomap"] = hit_particle_link
        g.ndata["theta_hits"] = theta_hits
        g.ndata["phi_hits"] = phi_hits
        g.ndata["pandora_cluster"] = pandora_cluster
        if len(y_data_graph) < 4:
            graph_empty = True
    else:
        graph_empty = True
        g = 0
        y_data_graph = 0
    if coord_cart_hits_norm.shape[0] < 10:
        graph_empty = True

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
    list_y = add_batch_number(list_graphs)
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


def add_batch_number(list_graphs):
    list_y = []
    for i, el in enumerate(list_graphs):
        y = el[1]
        batch_id = torch.ones(y.shape[0], 1) * i
        y = torch.cat((y, batch_id), dim=1)
        list_y.append(y)
    return list_y
