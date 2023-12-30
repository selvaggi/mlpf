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
    return (number_of_hits[1:].flatten()).tolist()


def find_mask_no_energy(hit_particle_link, hit_type_a, hit_energies, y, predict=False):
    """This function remove particles with tracks only and remove particles with low fractions

    Args:
        hit_particle_link (_type_): _description_
        hit_type_a (_type_): _description_
        hit_energies (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    energy_cut = 0.01
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
        if predict:
            if (
                np.array_equal(hit_types, [0, 1])
                or int(p) not in filt1
                or (number_of_hits[index] < 1)
                or (y[index, 8] == 1)
            ):  # This is commented to disable filtering
                list_remove.append(p)
                # print(
                #     "percentage of energy, number of hits",
                #     part_frac[int(p) - 1],
                #     number_of_hits[index],
                #     y[index, 3],
                #     y[index, 7],
                #     y[index, 8],
                # )
                # assert part_frac[int(p) - 1] <= energy_cut
        else:
            if (
                np.array_equal(hit_types, [0, 1])
                or int(p) not in filt1
                or (number_of_hits[index] < 1)
            ):  # This is commented to disable filtering
                list_remove.append(p)
                # print(
                #     "percentage of energy, number of hits",
                #     part_frac[int(p) - 1],
                #     number_of_hits[index],
                #     y[index, 3],
                # )

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


def get_particle_features(unique_list_particles, output, prediction):
    unique_list_particles = torch.Tensor(unique_list_particles).to(torch.int64)
    if prediction:
        number_particle_features = 12 - 2  # include these if added phi, theta
    else:
        number_particle_features = 9 - 2
    features_particles = torch.permute(
        torch.tensor(
            output["pf_features"][
                2:number_particle_features, list(unique_list_particles)
            ]
        ),
        (1, 0),
    )  #
    particle_coord = spherical_to_cartesian(
        features_particles[:, 0],
        features_particles[:, 1],
        features_particles[:, 2],
        normalized=True,
    )
    y_mass = features_particles[:, 3].view(-1).unsqueeze(1)
    y_mom = features_particles[:, 2].view(-1).unsqueeze(1)
    y_energy = torch.sqrt(y_mass**2 + y_mom**2)
    y_pid = features_particles[:, 4].view(-1).unsqueeze(1)
    if prediction:
        y_data_graph = torch.cat(
            (
                particle_coord,
                y_energy,
                y_mom,
                y_mass,
                y_pid,  # particle ID (discrete)
                features_particles[:, 5].view(-1).unsqueeze(1),  # decayed in calo
                features_particles[:, 6].view(-1).unsqueeze(1),  # decayed in tracker
            ),
            dim=1,
        )
    else:
        y_data_graph = torch.cat(
            (
                particle_coord,
                y_energy,
                y_mom,
                y_mass,
                y_pid,  # particle ID (discrete)
            ),
            dim=1,
        )
    return y_data_graph


def get_hit_features(output, number_hits, prediction):
    # identification of particles, clusters, pfos
    hit_particle_link = torch.tensor(output["pf_vectoronly"][0, 0:number_hits])
    pandora_cluster = torch.tensor(output["pf_vectoronly"][1, 0:number_hits])
    if prediction:
        pandora_cluster_energy = torch.tensor(output["pf_features"][-2, 0:number_hits])
        pfo_energy = torch.tensor(output["pf_features"][-1, 0:number_hits])
    else:
        pandora_cluster_energy = pandora_cluster * 0
        pfo_energy = pandora_cluster * 0
    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)

    # hit type
    hit_type_feature = torch.permute(
        torch.tensor(output["pf_vectors"][:, 0:number_hits]), (1, 0)
    )[:, 0].to(torch.int64)
    hit_type_one_hot = torch.nn.functional.one_hot(hit_type_feature, num_classes=4)

    # position, e, p
    pos_xyz_hits = torch.permute(
        torch.tensor(output["pf_points"][:, 0:number_hits]), (1, 0)
    )
    pf_features_hits = torch.permute(
        torch.tensor(output["pf_features"][0:2, 0:number_hits]), (1, 0)
    )  # removed theta, phi
    p_hits = pf_features_hits[:, 0].unsqueeze(1)
    p_hits[p_hits == -1] = 0  # correct p  of Hcal hits to be 0
    e_hits = pf_features_hits[:, 1].unsqueeze(1)
    e_hits[e_hits == -1] = 0  # correct the energy of the tracks to be 0

    return (
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
    )


def standardize_coordinates(coord_cart_hits):
    if len(coord_cart_hits) == 0:
        return coord_cart_hits, None
    std_scaler = StandardScaler()
    coord_cart_hits = std_scaler.fit_transform(coord_cart_hits)
    return torch.tensor(coord_cart_hits).float(), std_scaler


def create_dif_interactions(i, j, pos, number_p):
    x_interactions = pos
    x_interactions = torch.reshape(x_interactions, [number_p, 1, 2])
    x_interactions = x_interactions.repeat(1, number_p, 1)
    xi = x_interactions[i, j, :]
    xj = x_interactions[j, i, :]
    x_interactions_m = xi - xj
    return x_interactions_m


def spherical_to_cartesian(theta, phi, r, normalized=False):
    if normalized:
        r = torch.ones_like(theta)
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)
