import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class Particles_GT:
    
    angle: Optional[Any] = None
    coord: Optional[Any] = None
    E: Optional[Any] = None
    E_corrected: Optional[Any] = None
    m: Optional[Any] = None
    mass: Optional[Any] = None
    pid: Optional[Any] = None
    vertex: Optional[Any] = None
    unique_list_particles: Optional[Any] = None
    decayed_in_calo: Optional[Any] = None
    decayed_in_tracker: Optional[Any] = None
    batch_number: Optional[Any] = None

    def fill(self,unique_list_particles, output, prediction):
        unique_list_particles = torch.Tensor(unique_list_particles).to(torch.int64)
        
        number_particle_features = 18
        features_particles = torch.permute(
            torch.tensor(
                output["pf_features"][
                    2:number_particle_features, list(unique_list_particles)
                ]
            ),
            (1, 0),
        )  #
        particle_coord_angle = features_particles[:,0:2]
        particle_coord = features_particles[:, 10:13]
        vertex_coord = features_particles[:, 13:16]

        y_mass = features_particles[:, 3].view(-1).unsqueeze(1)
        y_mom = features_particles[:, 2].view(-1).unsqueeze(1)
        y_energy = torch.sqrt(y_mass**2 + y_mom**2)
        y_pid = features_particles[:, 4].view(-1).unsqueeze(1)
  
        self.angle= particle_coord_angle
        self.coord = particle_coord
        self.E = y_energy
        self.m = y_mom
        self.mass = y_mass
        self.pid = y_pid
        self.decayed_in_calo = features_particles[:, 5].view(-1).unsqueeze(1)
        self.decayed_in_tracker = features_particles[:, 6].view(-1).unsqueeze(1)
        self.unique_list_particles=unique_list_particles
        self.vertex=vertex_coord
        

    def __len__(self):
        return len(self.E)

    def mask(self, mask):
        for k in self.__dict__:
            if getattr(self, k) is not None:
                if type(getattr(self, k)) == list:
                    if getattr(self, k)[0] is not None:
                        setattr(self, k, getattr(self, k)[mask])
                else:
                    setattr(self, k, getattr(self, k)[mask])

    def copy(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def calculate_corrected_E(self, g, connections_list):
        for element in connections_list:
            # checked there is track
            parent_particle = element[0]
            mask_i = g.ndata["particle_number_nomap"] == parent_particle
            track_number = torch.sum(g.ndata["hit_type"][mask_i] == 1)
            if track_number > 0:
                # find index in list
                index_parent = torch.argmax(
                    1 * (self.unique_list_particles == parent_particle)
                )
                energy_daugthers = 0
                for daugther in element[1]:
                    if daugther != parent_particle:
                        if torch.sum(self.unique_list_particles == daugther) > 0:
                            index_daugthers = torch.argmax(
                                1 * (self.unique_list_particles == daugther)
                            )
                            energy_daugthers = (
                                self.E[index_daugthers] + energy_daugthers
                            )
                self.E_corrected[index_parent] = (
                    self.E_corrected[index_parent] - energy_daugthers
                )
                self.coord[index_parent] *= (1 - energy_daugthers / torch.norm(self.coord[index_parent]))
        if not connections_list:
            self.E_corrected = self.E.clone()


def concatenate_Particles_GT(list_of_Particles_GT):
    list_coord = [p[1].coord for p in list_of_Particles_GT]
    list_angle = [p[1].angle for p in list_of_Particles_GT]
    list_angle = torch.cat(list_angle, dim=0)
    list_vertex = [p[1].vertex for p in list_of_Particles_GT]
    list_coord = torch.cat(list_coord, dim=0)
    list_E = [p[1].E for p in list_of_Particles_GT]
    list_E = torch.cat(list_E, dim=0)
    list_E_corr = [p[1].E_corrected for p in list_of_Particles_GT]
    list_E_corr = torch.cat(list_E_corr, dim=0)
    list_m = [p[1].m for p in list_of_Particles_GT]
    list_m = torch.cat(list_m, dim=0)
    list_mass = [p[1].mass for p in list_of_Particles_GT]
    list_mass = torch.cat(list_mass, dim=0)
    list_pid = [p[1].pid for p in list_of_Particles_GT]
    list_pid = torch.cat(list_pid, dim=0)
    if list_vertex[0] is not None:
        list_vertex = torch.cat(list_vertex, dim=0)
    if hasattr(list_of_Particles_GT[0], "decayed_in_calo"):
        list_dec_calo = [p[1].decayed_in_calo for p in list_of_Particles_GT]
        list_dec_track = [p[1].decayed_in_tracker for p in list_of_Particles_GT]
        list_dec_calo = torch.cat(list_dec_calo, dim=0)
        list_dec_track = torch.cat(list_dec_track, dim=0)
    else:
        list_dec_calo = None
        list_dec_track = None
    batch_number = add_batch_number(list_of_Particles_GT)
    particle_batch = Particles_GT()
    particle_batch.angle = list_angle
    particle_batch.coord = list_coord
    particle_batch.E = list_E
    particle_batch.E_corrected = list_E_corr
    particle_batch.m = list_m
    particle_batch.pid = list_pid
    particle_batch.vertex=  list_vertex
    particle_batch.decayed_in_calo = list_dec_calo
    particle_batch.decayed_in_tracker = list_dec_track
    particle_batch.batch_number = batch_number
    return particle_batch

def add_batch_number(list_graphs):
    list_y = []
    for i, el in enumerate(list_graphs):
        y = el[1]
        batch_id = torch.ones(y.E.shape[0], 1) * i
        list_y.append(batch_id)
    list_y = torch.cat(list_y, dim=0)
    return list_y

def spherical_to_cartesian(phi, theta, r, normalized=False):
    if normalized:
        r = torch.ones_like(phi)
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)
