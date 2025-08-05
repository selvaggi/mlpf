from dataclasses import dataclass
from typing import Any, List, Optional
import torch 
from src.dataset.functions_data import modify_index_link_for_gamma_e
import numpy as np 
from src.dataset.utils_hits import CachedIndexList


@dataclass
class PandoraFeatures:
    # Features associated to the hits 
    pandora_cluster: Optional[Any] = None
    pandora_cluster_energy: Optional[Any] = None
    pfo_energy: Optional[Any] = None
    pandora_mom: Optional[Any] = None
    pandora_ref_point: Optional[Any] = None
    pandora_pid: Optional[Any] = None
    pandora_pfo_link: Optional[Any] = None


@dataclass
class Hits:
    pos_xyz_hits: Any
    pos_pxpypz: Any
    p_hits: Any
    e_hits: Any
    hit_particle_link: Any
    pandora_features: Any # type PandoraFeatures
    unique_list_particles: Any
    cluster_id: Any
    hit_type_feature: Any
    daughters: Any
    hit_link_modified: Any
    connection_list: Any
    chi_squared_tracks: Any
    hit_type_one_hot: Any

    
    @classmethod
    def from_data(cls, output, number_hits, prediction, number_part):
        hit_particle_link = torch.tensor(output["pf_vectoronly"][0, 0:number_hits])
        indx_daugthers = 3
        daughters = torch.tensor(output["pf_vectoronly"][indx_daugthers, 0:number_hits])
        if prediction:
            pandora_features = PandoraFeatures()
            pandora_features.pandora_cluster = torch.tensor(output["pf_vectoronly"][1, 0:number_hits])
            pandora_features.pandora_pfo_link = torch.tensor(output["pf_vectoronly"][2, 0:number_hits])
            
            pandora_features.pandora_mom = torch.permute(
                torch.tensor(output["pf_points_pfo"][0:3, 0:number_hits]), (1, 0)
            )
            pandora_features.pandora_ref_point = torch.permute(
                torch.tensor(output["pf_points_pfo"][3:6, 0:number_hits]), (1, 0)
            )
            if output["pf_points_pfo"].shape[0] > 6:
                pandora_features.pandora_pid = torch.tensor(output["pf_points_pfo"][6, 0:number_hits])
            else:
                pandora_features.pandora_pid=torch.zeros(number_hits)
                
            
            pandora_features.pandora_cluster_energy = torch.tensor(
                output["pf_features"][9, 0:number_hits]
            )
            pandora_features.pfo_energy = torch.tensor(output["pf_features"][10, 0:number_hits])
            chi_squared_tracks = torch.tensor(output["pf_features"][11, 0:number_hits])
        else:
            pandora_features = None
        
        # obtain hit type
        hit_type_feature = torch.permute(
            torch.tensor(output["pf_vectors"][:, 0:number_hits]), (1, 0)
        )[:, 0].to(torch.int64)
        
        # modify the index link for gamma and e (brems should point back to the photon)
        (
            hit_particle_link,
            hit_link_modified,
            connection_list,
        ) = modify_index_link_for_gamma_e(
            hit_type_feature, hit_particle_link, daughters, output, number_part
        )

        # obtain a 1,...,N id for the hits (the hit particle link might not be continuous)
        
        cluster_id, unique_list_particles = cls.find_cluster_id_static(hit_particle_link)

        # obtain the position of the hits and the energies and p
        pos_xyz_hits = torch.permute(
            torch.tensor(output["pf_points"][0:3, 0:number_hits]), (1, 0)
        )
        pf_features_hits = torch.permute(
            torch.tensor(output["pf_features"][0:2, 0:number_hits]), (1, 0)
        )  
        p_hits = pf_features_hits[:, 0].unsqueeze(1)
        p_hits[p_hits == -1] = 0  # correct p of Hcal hits to be 0
        e_hits = pf_features_hits[:, 1].unsqueeze(1)
        e_hits[e_hits == -1] = 0  # correct the energy of the tracks to be 0
        pos_pxpypz = torch.permute(
            torch.tensor(output["pf_points"][3:, 0:number_hits]), (1, 0)
        )
        hit_type_one_hot = torch.nn.functional.one_hot(
                    hit_type_feature, num_classes=5
                )
    
        return cls(
            pos_xyz_hits=pos_xyz_hits,
            pos_pxpypz=pos_pxpypz,
            p_hits=p_hits,
            e_hits=e_hits,
            hit_particle_link=hit_particle_link,
            pandora_features= pandora_features, 
            unique_list_particles=unique_list_particles,
            cluster_id=cluster_id,
            hit_type_feature=hit_type_feature,
            daughters=daughters,
            hit_link_modified=hit_link_modified,
            connection_list=connection_list,
            chi_squared_tracks=chi_squared_tracks,
            hit_type_one_hot = hit_type_one_hot
        )
    def find_cluster_id(self):
        cluster_id, unique_list_particles = self.find_cluster_id_static(self.hit_particle_link)
        self.cluster_id = cluster_id
    @staticmethod
    def find_cluster_id_static(hit_particle_link):
        hit_particle_link = hit_particle_link
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

    def mask_hits(self, mask_hits, prediction):
        self.p_hits = self.p_hits[mask_hits]
        self.e_hits = self.e_hits[mask_hits]
        self.hit_particle_link = self.hit_particle_link[~mask_hits]
        self.pos_xyz_hits = self.pos_xyz_hits[~mask_hits]
        self.pos_pxpypz = self.pos_pxpypz[~mask_hits]
        if prediction:
            self.pandora_features.pandora_cluster = self.pandora_cluster[~mask_hits]
            self.pandora_features.pandora_cluster_energy = self.pandora_cluster_energy[~mask_hits]
            self.pandora_features.pandora_mom = self.pandora_mom[~mask_hits]
            self.pandora_ref_point = self.pandora_ref_point[~mask_hits]
            self.pandora_pid = self.pandora_pid[~mask_hits]
            self.pfo_energy = self.pfo_energy[~mask_hits]
            self.pandora_pfo_link = self.pandora_pfo_link[~mask_hits]
        self.hit_type_feature = self.hit_type_feature[~mask_hits]
        self.hit_link_modified = self.hit_link_modified[~mask_hits]
        self.daughters = self.daughters[~mask_hits]
        self.chi_squared_tracks = self.chi_squared_tracks[~mask_hits]
