import numpy as np
import torch
import dgl
from torch_scatter import scatter_add
from src.dataset.utils_hits import CachedIndexList, get_number_of_daughters, get_ratios, get_number_hits, modify_index_link_for_gamma_e, get_e_reco
def find_mask_no_energy(
    hit_particle_link,
    hit_type_a,
    hit_energies,
    y,
    daughters,
    predict=False,
    is_Ks=False,
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
    if is_Ks == False:
        for index, p in enumerate(list_p):
            mask = hit_particle_link == p
            hit_types = np.unique(hit_type_a[mask])

            if predict:
                if (
                    np.array_equal(hit_types, [0, 1])
                    or int(p) not in filt1
                    or (number_of_hits[index] < 2)
                    or (y.decayed_in_tracker[index] == 1)
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
        non_noise_idx = torch.where(hit_particle_link != -1)[0]  #
        noise_idx = torch.where(hit_particle_link == -1)[0]  #
        unique_list_particles1 = torch.unique(hit_particle_link)[1:]
        cluster_id_ = torch.searchsorted(
            unique_list_particles1, hit_particle_link[non_noise_idx], right=False
        )
        cluster_id_small = 1.0 * cluster_id_ + 1
        cluster_id = hit_particle_link.clone()
        cluster_id[non_noise_idx] = cluster_id_small
        cluster_id[noise_idx] = 0
    else:
        c_unique_list_particles = CachedIndexList(unique_list_particles)
        cluster_id = map(
            lambda x: c_unique_list_particles.index(x), hit_particle_link.tolist()
        )
        cluster_id = torch.Tensor(list(cluster_id)) + 1
    return cluster_id, unique_list_particles




def get_hit_features(
    output, number_hits, prediction, number_part, hit_chis, pos_pxpy, is_Ks=False
):
    
    hit_particle_link = torch.tensor(output["pf_vectoronly"][0, 0:number_hits])
    if prediction:
        indx_daugthers = 3
    else:
        indx_daugthers = 1
    daughters = torch.tensor(output["pf_vectoronly"][indx_daugthers, 0:number_hits])
    if prediction:
        pandora_cluster = torch.tensor(output["pf_vectoronly"][1, 0:number_hits])
        pandora_pfo_link = torch.tensor(output["pf_vectoronly"][2, 0:number_hits])
        if is_Ks:
            pandora_mom = torch.permute(
                torch.tensor(output["pf_points_pfo"][0:3, 0:number_hits]), (1, 0)
            )
            pandora_ref_point = torch.permute(
                torch.tensor(output["pf_points_pfo"][3:6, 0:number_hits]), (1, 0)
            )
            if output["pf_points_pfo"].shape[0] > 6:
                pandora_pid = torch.tensor(output["pf_points_pfo"][6, 0:number_hits])
            else:
                # zeros
                # print("Zeros for pandora pid!")
                pandora_pid=torch.zeros(number_hits)
            
        else:
            pandora_mom = None
            pandora_ref_point = None
            pandora_pid = None
        if is_Ks:
            pandora_cluster_energy = torch.tensor(
                output["pf_features"][9, 0:number_hits]
            )
            pfo_energy = torch.tensor(output["pf_features"][10, 0:number_hits])
            chi_squared_tracks = torch.tensor(output["pf_features"][11, 0:number_hits])
        elif hit_chis:
            pandora_cluster_energy = torch.tensor(
                output["pf_features"][-3, 0:number_hits]
            )
            pfo_energy = torch.tensor(output["pf_features"][-2, 0:number_hits])
            chi_squared_tracks = torch.tensor(output["pf_features"][-1, 0:number_hits])
        else:
            pandora_cluster_energy = torch.tensor(
                output["pf_features"][-2, 0:number_hits]
            )
            pfo_energy = torch.tensor(output["pf_features"][-1, 0:number_hits])
            chi_squared_tracks = None

    else:
        pandora_cluster = None
        pandora_pfo_link = None
        pandora_cluster_energy = None
        pfo_energy = None
        chi_squared_tracks = None
        pandora_mom = None
        pandora_ref_point = None
        pandora_pid = None
    # hit type
    # t0 = time.time()
    hit_type_feature = torch.permute(
        torch.tensor(output["pf_vectors"][:, 0:number_hits]), (1, 0)
    )[:, 0].to(torch.int64)
    # t1 = time.time()
    (
        hit_particle_link,
        hit_link_modified,
        connection_list,
    ) = modify_index_link_for_gamma_e(
        hit_type_feature, hit_particle_link, daughters, output, number_part, is_Ks
    )
    # t2 = time.time()
    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)
    # t3 = time.time()
    # wandb.log({"time_hit_type_feature": t1-t0, "time_modify_index_link_for_gamma_e": t2-t1, "time_find_cluster_id": t3-t2})
    # position, e, p
    pos_xyz_hits = torch.permute(
        torch.tensor(output["pf_points"][0:3, 0:number_hits]), (1, 0)
    )
    pf_features_hits = torch.permute(
        torch.tensor(output["pf_features"][0:2, 0:number_hits]), (1, 0)
    )  # removed theta, phi
    p_hits = pf_features_hits[:, 0].unsqueeze(1)
    p_hits[p_hits == -1] = 0  # correct p of Hcal hits to be 0
    e_hits = pf_features_hits[:, 1].unsqueeze(1)
    e_hits[e_hits == -1] = 0  # correct the energy of the tracks to be 0
    if pos_pxpy:
        pos_pxpypz = torch.permute(
            torch.tensor(output["pf_points"][3:, 0:number_hits]), (1, 0)
        )
    else:
        pos_pxpypz = pos_xyz_hits
    # pos_pxpypz = pos_theta_phi

    return (
        pos_xyz_hits,
        pos_pxpypz,
        p_hits,
        e_hits,
        hit_particle_link,
        pandora_cluster,
        pandora_cluster_energy,
        pfo_energy,
        pandora_mom,
        pandora_ref_point,
        pandora_pid, 
        unique_list_particles,
        cluster_id,
        hit_type_feature,
        pandora_pfo_link,
        daughters,
        hit_link_modified,
        connection_list,
        chi_squared_tracks,
    )


def calculate_distance_to_boundary(g):
    r = 2150
    r_in_endcap = 2307
    mask_endcap = (torch.abs(g.ndata["pos_hits_xyz"][:, 2]) - r_in_endcap) > 0
    mask_barrer = ~mask_endcap
    weight = torch.ones_like(g.ndata["pos_hits_xyz"][:, 0])
    C = g.ndata["pos_hits_xyz"]
    A = torch.Tensor([0, 0, 1]).to(C.device)
    P = (
        r
        * 1
        / (torch.norm(torch.cross(A.view(1, -1), C, dim=-1), dim=1)).unsqueeze(1)
        * C
    )
    P1 = torch.abs(r_in_endcap / g.ndata["pos_hits_xyz"][:, 2].unsqueeze(1)) * C
    weight[mask_barrer] = torch.norm(P - C, dim=1)[mask_barrer]
    weight[mask_endcap] = torch.norm(P1[mask_endcap] - C[mask_endcap], dim=1)
    g.ndata["radial_distance"] = weight
    weight_ = torch.exp(-(weight / 1000))
    g.ndata["radial_distance_exp"] = weight_
    return g




def create_noise_label(hit_energies, hit_particle_link, y, cluster_id):
    unique_p_numbers = torch.unique(cluster_id)
    number_of_hits = get_number_hits(hit_energies, cluster_id)
    e_reco = get_e_reco(hit_energies, cluster_id)
    mask_hits = torch.Tensor(number_of_hits) < 6
    mask_p = e_reco<0.10
    mask_all = mask_hits.view(-1) + mask_p.view(-1)
    list_remove = unique_p_numbers[mask_all.view(-1)]

    if len(list_remove) > 0:
        mask = torch.tensor(np.full((len(cluster_id)), False, dtype=bool))
        for p in list_remove:
            mask1 = cluster_id == p
            mask = mask1 + mask
    else:
        mask = torch.tensor(np.full((len(cluster_id)), False, dtype=bool))
    list_p = unique_p_numbers
    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles
    else:
        mask_particles = torch.tensor(np.full((len(list_p)), False, dtype=bool))
    return mask.to(bool), ~mask_particles.to(bool)