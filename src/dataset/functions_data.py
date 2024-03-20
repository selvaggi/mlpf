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


def get_number_of_daughters(hit_type_feature, hit_particle_link, daughters):
    a = hit_particle_link
    b = daughters
    a_u = torch.unique(a)
    number_of_p = torch.zeros_like(a_u)
    for p, i in enumerate(a_u):
        mask2 = a == i
        print(torch.unique(b[mask2]))
        number_of_p[p] = torch.sum(torch.unique(b[mask2]) != -1)
    return number_of_p


def find_mask_no_energy(
    hit_particle_link, hit_type_a, hit_energies, y, daughters, predict=False
):
    """This function remove particles with tracks only and remove particles with low fractions
    # Remove 2212 going to multiple particles without tracks for now
    # remove particles below energy cut
    # remove particles that decayed in the tracker
    # remove particles with two tracks (due to bad tracking)
    # remove particles with daughters for the moment

    Args:
        hit_particle_link (_type_): _description_
        hit_type_a (_type_): _description_
        hit_energies (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """

    number_of_daughters = get_number_of_daughters(
        hit_type_a, hit_particle_link, daughters
    )
    list_p = np.unique(hit_particle_link)
    list_remove = []
    part_frac = torch.tensor(get_ratios(hit_energies, hit_particle_link, y))
    number_of_hits = get_number_hits(hit_energies, hit_particle_link)
    if predict:
        energy_cut = 0.1
        filt1 = (torch.where(part_frac >= energy_cut)[0] + 1).long().tolist()
    else:
        energy_cut = 0.01
        filt1 = (torch.where(part_frac >= energy_cut)[0] + 1).long().tolist()
    number_of_tracks = scatter_add(1 * (hit_type_a == 1), hit_particle_link.long())[1:]
    for index, p in enumerate(list_p):
        mask = hit_particle_link == p
        hit_types = np.unique(hit_type_a[mask])

        if predict:
            if (
                np.array_equal(hit_types, [0, 1])
                or int(p) not in filt1
                or (number_of_hits[index] < 2)
                or (y[index, 8] == 1)
                or number_of_tracks[index] == 2
                or number_of_daughters[index] > 1
            ):
                list_remove.append(p)
        else:
            if (
                np.array_equal(hit_types, [0, 1])
                or int(p) not in filt1
                or (number_of_hits[index] < 2)
                or number_of_tracks[index] == 2
                or number_of_daughters[index] > 1
            ):
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
                y_energy,  # 3
                y_mom,  # 4
                y_mass,  # 5
                y_pid,  # 6 particle ID (discrete)
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


def modify_index_link_for_gamma_e(
    hit_type_feature, hit_particle_link, daughters, output, number_part
):
    mask = hit_type_feature > 1
    a = hit_particle_link[mask]
    b = daughters[mask]
    a_u = torch.unique(a)
    number_of_p = torch.zeros_like(a_u)
    for p, i in enumerate(a_u):
        mask2 = a == i
        number_of_p[p] = len(torch.unique(b[mask2]))
    pid_particles = torch.tensor(output["pf_features"][6, 0:number_part])
    electron_photon_mask = (torch.abs(pid_particles[a_u.long()]) == 11) + (
        pid_particles[a_u.long()] == 22
    )
    electron_photon_mask = electron_photon_mask * (number_of_p > 1)
    index_change = a_u[electron_photon_mask]
    # print("index change", index_change)
    # if len(index_change) > 0:
    #     for i in index_change:
    #         print("daughters changed", daughters[hit_particle_link == i])

    for i in index_change:
        mask_n = mask * (hit_particle_link == i)
        hit_particle_link[mask_n] = daughters[mask_n]
    return hit_particle_link


def get_hit_features(output, number_hits, prediction, number_part):
    # identification of particles, clusters, pfos
    hit_particle_link = torch.tensor(output["pf_vectoronly"][0, 0:number_hits])
    pandora_cluster = torch.tensor(output["pf_vectoronly"][1, 0:number_hits])
    pandora_pfo_link = torch.tensor(output["pf_vectoronly"][2, 0:number_hits])
    daughters = torch.tensor(output["pf_vectoronly"][3, 0:number_hits])
    if prediction:
        pandora_cluster_energy = torch.tensor(output["pf_features"][-2, 0:number_hits])
        pfo_energy = torch.tensor(output["pf_features"][-1, 0:number_hits])
    else:
        pandora_cluster_energy = pandora_cluster * 0
        pfo_energy = pandora_cluster * 0
    # hit type
    hit_type_feature = torch.permute(
        torch.tensor(output["pf_vectors"][:, 0:number_hits]), (1, 0)
    )[:, 0].to(torch.int64)

    hit_particle_link = modify_index_link_for_gamma_e(
        hit_type_feature, hit_particle_link, daughters, output, number_part
    )

    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)

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
        hit_particle_link,
        pandora_cluster,
        pandora_cluster_energy,
        pfo_energy,
        unique_list_particles,
        cluster_id,
        hit_type_feature,
        pandora_pfo_link,
        daughters,
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
