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
    pandora_mom_components: Optional[Any] = None


@dataclass
class Hits:
    pos_xyz_hits: Any
    pos_pxpypz: Any
    p_hits: Any
    e_hits: Any
    hit_particle_link: Any
    pandora_features: Any # type PandoraFeatures
    hit_type_feature: Any
    chi_squared_tracks: Any
    hit_type_one_hot: Any
    # hit_particle_link_calomother: Any

    
    @classmethod
    def from_data(cls, output, number_hits, prediction, number_part):
        hit_particle_link_hits = torch.tensor(output["ygen_hit"])
        hit_particle_link_hits_calomother = torch.tensor(output["ygen_hit_calomother"])
        hit_particle_link_tracks= torch.tensor(output["ygen_track"])
        hit_particle_link = torch.cat((hit_particle_link_hits, hit_particle_link_tracks), dim=0)
        # hit_particle_link_calomother = torch.cat((hit_particle_link_hits_calomother, hit_particle_link_tracks), dim=0)
        if prediction:
            pandora_features = PandoraFeatures()
            X_pandora = torch.tensor(output["X_pandora"])
            pfo_link_hits = torch.tensor(output["pfo_calohit"])
            pfo_link_tracks = torch.tensor(output["pfo_track"])
            pfo_link = torch.cat((pfo_link_hits, pfo_link_tracks), dim=0)
            pandora_features.pandora_pfo_link = pfo_link
            pfo_link_temp = pfo_link.clone()
            pfo_link_temp[pfo_link_temp==-1]=0
            
            pandora_features.pandora_mom = X_pandora[pfo_link_temp, 8]
            pandora_features.pandora_ref_point = X_pandora[pfo_link_temp, 4:7]
            pandora_features.pandora_mom_components = X_pandora[pfo_link_temp, 1:4]
            pandora_features.pandora_pid = X_pandora[pfo_link_temp, 0]
            pandora_features.pfo_energy = X_pandora[pfo_link_temp, 7]
            pandora_features.pandora_mom[pfo_link==-1]=0
            pandora_features.pandora_mom_components[pfo_link==-1]=0
            pandora_features.pandora_ref_point[pfo_link==-1]=0
            pandora_features.pandora_pid[pfo_link==-1]=0
            pandora_features.pfo_energy[pfo_link==-1]=0

        else:
            pandora_features = None
        X_hit = torch.tensor(output["X_hit"])
        X_track = torch.tensor(output["X_track"])
        # obtain hit type
        hit_type_feature_hit = X_hit[:,-2]+1 #tyep (1,2,3,4 hits)
        hit_type_feature_track = X_track[:,0] #elemtype (1 for tracks)
        hit_type_feature = torch.cat((hit_type_feature_hit, hit_type_feature_track), dim=0).to(torch.int64)
        # obtain the position of the hits and the energies and p
        pos_xyz_hits_hits = X_hit[:,6:9]
        pos_xyz_hits_tracks = X_track[:,12:15] #(referencePoint_calo.i)
        pos_xyz_hits = torch.cat((pos_xyz_hits_hits, pos_xyz_hits_tracks), dim=0)
        e_hits = X_hit[:,5]
        e_tracks =X_track[:,5]*0
        e = torch.cat((e_hits, e_tracks), dim=0).view(-1,1)
        p_hits = X_hit[:,5]*0
        p_tracks =X_track[:,5]
        pos_pxpypz_hits_tracks = X_track[:,6:9]
        pos_pxpypz = torch.cat((pos_xyz_hits_hits*0, pos_pxpypz_hits_tracks), dim=0)
        p = torch.cat((p_hits, p_tracks), dim=0).view(-1,1)
        hit_type_one_hot = torch.nn.functional.one_hot(
                    hit_type_feature, num_classes=5
                )
        chi_tracks = X_track[:,15]
        chi_squared_tracks = torch.cat((p_hits, chi_tracks), dim=0)
        return cls(
            pos_xyz_hits=pos_xyz_hits,
            pos_pxpypz=pos_pxpypz,
            p_hits=p,
            e_hits=e,
            hit_particle_link=hit_particle_link,
            pandora_features= pandora_features, 
            hit_type_feature=hit_type_feature,
            chi_squared_tracks=chi_squared_tracks,
            hit_type_one_hot = hit_type_one_hot, 
            # hit_particle_link_calomother = hit_particle_link_calomother
        )



