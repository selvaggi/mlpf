import numpy as np
import torch
import dgl
from torch_scatter import scatter_add, scatter_sum, scatter_mean
from sklearn.preprocessing import StandardScaler
from torch_scatter import scatter_sum
from src.dataset.functions_data import (
    find_mask_no_energy,
    find_cluster_id,
    get_hit_features,
    calculate_distance_to_boundary,
    find_mask_no_energy_close_particles
)
from src.dataset.functions_particles import get_particle_features, concatenate_Particles_GT
from src.dataset.utils_hits import create_noise_label, create_noise_tracks
from src.layers.object_cond import scatter_count
import time

def create_inputs_from_table(
    output, hits_only, prediction=False, hit_chis=False, pos_pxpy=False, is_Ks=False
):
    """Used by graph creation to get nodes and edge features

    Args:
        output (_type_): input from the root reading
        hits_only (_type_): reading only hits or also tracks
        prediction (bool, optional): if running in eval mode. Defaults to False.

    Returns:
        _type_: all information to construct a graph
    """
    remove_close_particles = False
    graph_empty = False
    number_hits = np.int32(np.sum(output["pf_mask"][0]))
    number_part = np.int32(np.sum(output["pf_mask"][1]))
    (
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
    ) = get_hit_features(
        output,
        number_hits,
        prediction,
        number_part,
        hit_chis=hit_chis,
        pos_pxpy=pos_pxpy,
        is_Ks=is_Ks,
    )


    if torch.sum(torch.Tensor(unique_list_particles)>20000)>0:
        graph_empty = True
    else:
       
        y_data_graph = get_particle_features(
            unique_list_particles, output, prediction, connection_list
        )
        if remove_close_particles:
            mask_hits, mask_particles = find_mask_no_energy_close_particles(
                cluster_id,
                hit_type_feature,
                e_hits,
                y_data_graph,
                daughters,
                prediction,
                is_Ks=is_Ks,
            )
            
            cluster_id, unique_list_particles = find_cluster_id(hit_particle_link[~mask_hits])
            y_data_graph.mask(~mask_particles)
        if not is_Ks:
            mask_hits, mask_particles = find_mask_no_energy(
                cluster_id,
                hit_type_feature,
                e_hits,
                y_data_graph,
                daughters,
                prediction,
                is_Ks=is_Ks,
            )
            # create mapping from links to number of particles in the event
            cluster_id, unique_list_particles = find_cluster_id(hit_particle_link[~mask_hits])
            y_data_graph.mask(~mask_particles)
       
        if prediction:
            if is_Ks and not remove_close_particles:
                result = [
                    y_data_graph,  # y_data_graph[~mask_particles],
                    p_hits,
                    e_hits,
                    cluster_id,
                    hit_particle_link,
                    pos_xyz_hits,
                    pos_pxpypz,
                    pandora_cluster,
                    pandora_cluster_energy,
                    pandora_mom,
                    pandora_ref_point,
                    pandora_pid,
                    pfo_energy,
                    pandora_pfo_link,
                    hit_type_feature,
                    hit_link_modified,
                    daughters,
                ]
            else:
                result = [
                    y_data_graph,  # y_data_graph[~mask_particles],
                    p_hits[~mask_hits],
                    e_hits[~mask_hits],
                    cluster_id,
                    hit_particle_link[~mask_hits],
                    pos_xyz_hits[~mask_hits],
                    pos_pxpypz[~mask_hits],
                    pandora_cluster[~mask_hits],
                    pandora_cluster_energy[~mask_hits],
                    pandora_mom[~mask_hits],
                    pandora_ref_point[~mask_hits],
                    pandora_pid[~mask_hits], 
                    pfo_energy[~mask_hits],
                    pandora_pfo_link[~mask_hits],
                    hit_type_feature[~mask_hits],
                    hit_link_modified[~mask_hits],
                    daughters[~mask_hits],
                ]
        else:
       
            result_p = p_hits[~mask_hits]
            result_e = e_hits[~mask_hits]
            result_hit_link = hit_particle_link[~mask_hits]
            result_xyz = pos_xyz_hits[~mask_hits]
            result_pxyz = pos_pxpypz[~mask_hits]
            result_hit_type = hit_type_feature[~mask_hits]
            result_mod = hit_link_modified[~mask_hits]

            result = [
                y_data_graph,  # y_data_graph[~mask_particles],
                result_p,
                result_e,
                cluster_id,
                result_hit_link,
                result_xyz,
                result_pxyz,
                pandora_cluster,
                pandora_cluster_energy,
                pandora_mom,
                pandora_ref_point,
                pandora_pid, 
                pfo_energy,
                pandora_pfo_link,
                result_hit_type,
                result_mod,
            ]
        if hit_chis:
            if is_Ks and not remove_close_particles:
                result.append(
                        chi_squared_tracks,
                    )
            else:
                result.append(
                    chi_squared_tracks[~mask_hits],
                )
                
        else:
            result.append(None)

        if is_Ks and not remove_close_particles:
            hit_type = hit_type_feature
        else:
            hit_type = hit_type_feature[~mask_hits]
       
        if hits_only:
            hit_mask = (hit_type == 0) | (hit_type == 1)
            hit_mask = ~hit_mask
            for i in range(1, len(result)):
                if result[i] is not None:
                    result[i] = result[i][hit_mask]
            hit_type_one_hot = torch.nn.functional.one_hot(
                hit_type_feature[~mask_hits][hit_mask] - 2, num_classes=2
            )

        else:
            # # if we want the tracks keep only 1 track hit per charged particle.
            # hit_mask = hit_type == 10
            # hit_mask = ~hit_mask
            # for i in range(1, len(result)):
            #     if result[i] is not None:
            #         # if len(result[i].shape) == 2 and result[i].shape[0] == 3:
            #         #     result[i] = result[i][:, hit_mask]
            #         # else:
            #         #     result[i] = result[i][hit_mask]
            #         result[i] = result[i][hit_mask]
            if is_Ks and not remove_close_particles:
                hit_type_one_hot = torch.nn.functional.one_hot(
                    hit_type_feature, num_classes=5
                )
            else:
                hit_type_one_hot = torch.nn.functional.one_hot(
                    hit_type_feature[~mask_hits], num_classes=5
                )
        result.append(hit_type_one_hot)
        result.append(connection_list)
       
        return result
    if graph_empty:
        return [None]

def remove_hittype0(graph):
    filt = graph.ndata["hit_type"] == 0
    # graph.ndata["hit_type"] -= 1
    return dgl.remove_nodes(graph, torch.where(filt)[0])

def store_track_at_vertex_at_track_at_calo(graph):
    # To make it compatible with clustering, remove the 0 hit type nodes and store them as pos_pxpypz_at_vertex
    tracks_at_calo = graph.ndata["hit_type"] == 1
    tracks_at_vertex = graph.ndata["hit_type"] == 0
    part = graph.ndata["particle_number"].long()
    # assert (part[tracks_at_calo] == part[tracks_at_vertex]).all()
    graph.ndata["pos_pxpypz_at_vertex"] = torch.zeros_like(graph.ndata["pos_pxpypz"])
    graph.ndata["pos_pxpypz_at_vertex"][tracks_at_calo] = graph.ndata["pos_pxpypz"][tracks_at_vertex]
    return remove_hittype0(graph)

def create_graph(
    output,
    config=None,
    n_noise=0,
):
    graph_empty = False
    hits_only = config.graph_config.get(
        "only_hits", False
    )  
    prediction = config.graph_config.get("prediction", False)
    hit_chis = config.graph_config.get("hit_chis_track", False)
    pos_pxpy = config.graph_config.get("pos_pxpy", False)
    is_Ks = config.graph_config.get("ks", False)
    result = create_inputs_from_table(
        output,
        hits_only=hits_only,
        prediction=prediction,
        hit_chis=hit_chis,
        pos_pxpy=pos_pxpy,
        is_Ks=is_Ks,
    )
    if len(result) == 1:
        graph_empty = True
        g = 0
        y_data_graph = 0
    else:
        (
            y_data_graph,
            p_hits,
            e_hits,
            cluster_id,
            hit_particle_link,
            pos_xyz_hits,
            pos_pxpypz,
            pandora_cluster,
            pandora_cluster_energy,
            pandora_mom,
            pandora_ref_point,
            pandora_pid, 
            pandora_pfo_energy,
            pandora_pfo_link,
            hit_type,
            hit_link_modified,
            daughters, 
            chi_squared_tracks,
            hit_type_one_hot,
            connections_list
        ) = result
        # t0 = time.time()
        mask_loopers, mask_particles = create_noise_label(
        e_hits, hit_particle_link, y_data_graph, cluster_id
        )
        hit_particle_link[mask_loopers] = -1
        y_data_graph.mask(mask_particles)
        # t1 = time.time()
        cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)
        # t2 = time.time()
        # wandb.log({"time_create_noise_label": t1-t0, "time_find_cluster_id": t2-t1})
        # build graph
        graph_coordinates = pos_xyz_hits 
        graph_empty = False
        g = dgl.graph(([], []))
        g.add_nodes(graph_coordinates.shape[0])
        if hits_only == False:
            hit_features_graph = torch.cat(
                (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
            )  
        else:
            hit_features_graph = torch.cat(
                (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
            )  
        
        # fill graph features
        g.ndata["h"] = hit_features_graph
        g.ndata["pos_hits_xyz"] = pos_xyz_hits
        g.ndata["pos_pxpypz"] = pos_pxpypz
        g = calculate_distance_to_boundary(g)
        g.ndata["hit_type"] = hit_type
        g.ndata["e_hits"] = e_hits  
        if hit_chis:
            g.ndata["chi_squared_tracks"] = chi_squared_tracks
        g.ndata["particle_number"] = cluster_id
        g.ndata["hit_link_modified"] = hit_link_modified
        g.ndata["particle_number_nomap"] = hit_particle_link
        if prediction:
            g.ndata["pandora_cluster"] = pandora_cluster
            g.ndata["pandora_pfo"] = pandora_pfo_link
            g.ndata["pandora_cluster_energy"] = pandora_cluster_energy
            g.ndata["pandora_pfo_energy"] = pandora_pfo_energy
            if is_Ks:
                g.ndata["pandora_momentum"] = pandora_mom
                g.ndata["pandora_reference_point"] = pandora_ref_point
                g.ndata["daughters"] = daughters
                g.ndata["pandora_pid"] = pandora_pid
        y_data_graph.calculate_corrected_E(g, connections_list)
        if is_Ks == True:
            if y_data_graph.pid.flatten().shape[0] == 4 and np.count_nonzero(y_data_graph.pid.flatten() == 22) == 4:
                graph_empty = False
                print("Graph not empty")
            else:
                graph_empty = True
            graph_empty = False
        #   #  if g.ndata["h"].shape[0] < 10 or (set(g.ndata["hit_type"].unique().tolist()) == set([0, 1]) and g.ndata["hit_type"][g.ndata["hit_type"] == 1].shape[0] < 10):
        #         graph_empty = True  # less than 10 hits
        # print("y len", len(y_data_graph))
        # if is_Ks == False:
        #     if len(y_data_graph) < 4:
        #         graph_empty = True
        if torch.unique(hit_particle_link).shape[0]==1 and torch.unique(hit_particle_link)[0]==-1:
            graph_empty = True 
        if pos_xyz_hits.shape[0] < 10:
            graph_empty = True
        return [g, y_data_graph], graph_empty
    g = store_track_at_vertex_at_track_at_calo(g)
    g = make_bad_tracks_noise_tracks(g, y_data_graph)
    # g = make_graph_with_edges(g)
    return [g, y_data_graph], graph_empty


def connect_mask():
    def func(edges):
        hit_type_src = edges.src["hit_type"]
        hit_type_dst = edges.dst["hit_type"]
        pos_src = edges.src["pos_hits_xyz"]
        pos_dst = edges.dst["pos_hits_xyz"]
        ecal_src = hit_type_src == 2
        ecal_dst = hit_type_dst == 2
        track_src = hit_type_src == 1
        track_dst = hit_type_dst == 1
        hcal_src = hit_type_src == 3
        hcal_dst = hit_type_dst == 3
        muon_src = hit_type_src == 4
        muon_dst = hit_type_dst == 4
        distance = torch.norm(pos_src-pos_dst, dim=-1)
        angle = torch.sum(pos_src*pos_dst, dim=-1)
        angle = angle/(torch.norm(pos_src, dim=-1)*torch.norm(pos_dst, dim=-1))

        ecal_mask = ecal_src*ecal_dst*(angle>0.999)*(distance<50)
        hcal_mask = hcal_src*hcal_dst*(angle>0.999)*(distance<150)
        ecal_hcal_mask = ecal_src*hcal_dst*(angle>0.999)*(distance<250)
        hcal_muon_mask = hcal_src*muon_dst*(angle>0.999)*(distance<1200)
        muon_muon = muon_src*muon_dst*(angle>0.999)*(distance<300)
        track_ecal = track_src*ecal_dst*(angle>0.999)*(distance<15)
        mask_total = ecal_mask+hcal_mask+ecal_hcal_mask+hcal_muon_mask+muon_muon+track_ecal
        connect_mask = 1*(mask_total>0)
        return {"connect": connect_mask }

    return func

def make_graph_with_edges(g):
    number_p = g.number_of_nodes()-1
    i, j = torch.tril_indices(number_p, number_p)  # , offset=-1)
    g.add_edges(i, j) # create fully connected graph
    g = dgl.to_simple(g)  # remove repated edges
    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.remove_self_loop(g)
    g.apply_edges(connect_mask())
    g.remove_edges(torch.where(g.edata["connect"]==0)[0].long())
    return g 



def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    list_graphs_g = [el[0] for el in list_graphs]
    # list_y = add_batch_number(list_graphs)
    # ys = torch.cat(list_y, dim=0)
    # ys = torch.reshape(ys, [-1, list_y[0].shape[1]])
    ys = concatenate_Particles_GT(list_graphs)
    bg = dgl.batch(list_graphs_g)
    # reindex particle number
    return bg, ys

def make_bad_tracks_noise_tracks(g, y ):
    # is_chardged =scatter_add((g.ndata["hit_type"]==1).view(-1), g.ndata["particle_number"].long())[1:]
    mask_hit_type_t1 = g.ndata["hit_type"]==2
    mask_hit_type_t2 = g.ndata["hit_type"]==1
    mask_all = mask_hit_type_t1
    # the other error could come from no hits in the ECAL for a cluster
    # mean_pos_cluster = scatter_mean(g.ndata["pos_hits_xyz"][mask_all], g.ndata["particle_number"][mask_all].long().view(-1), dim=0)
    mean_pos_cluster_all = []
    mean_pos_cluster_ecal = []
    pos_track = g.ndata["pos_hits_xyz"][mask_hit_type_t2]
    particle_track = g.ndata["particle_number"][mask_hit_type_t2]
    if len(particle_track)>0:
        for index, i in enumerate(particle_track):
            if i ==0:
                mean_pos_cluster_all.append(torch.zeros((1,3)).view(-1,3))
                mean_pos_cluster_ecal.append(torch.zeros((1,3)).view(-1,3))
            else:
                mask_labels_i = g.ndata["particle_number"] ==i
                mean_pos_cluster = torch.mean(g.ndata["pos_hits_xyz"][mask_labels_i*mask_hit_type_t1], dim=0)
                mean_pos_cluster_all.append(mean_pos_cluster.view(-1,3))
                if len(g.ndata["radial_distance"][mask_labels_i*mask_hit_type_t1])>50:
                    index_search_ecal = 50
                else:
                    index_search_ecal = len(g.ndata["radial_distance"][mask_labels_i*mask_hit_type_t1])
    
                index_sort = torch.argsort(g.ndata["radial_distance"][mask_labels_i*mask_hit_type_t1])[0:index_search_ecal]
                distance_from_track = (torch.norm(g.ndata["pos_hits_xyz"][mask_labels_i*mask_hit_type_t1][index_sort]-pos_track[index], dim=1)/1000) < 0.1
                if torch.sum(distance_from_track)==0:
                    mean_pos_cluster_ecal.append(torch.zeros((1,3)).view(-1,3))
                else:
                    mean_cl = torch.sum(g.ndata["pos_hits_xyz"][mask_labels_i*mask_hit_type_t1][index_sort][distance_from_track]*g.ndata["e_hits"][mask_labels_i*mask_hit_type_t1][index_sort][distance_from_track], dim=0)/torch.sum(g.ndata["e_hits"][mask_labels_i*mask_hit_type_t1][index_sort][distance_from_track])
                    mean_pos_cluster_ecal.append(mean_cl.view(-1,3))
        mean_pos_cluster_all = torch.cat(mean_pos_cluster_all, dim=0)
        mean_pos_cluster_ecal = torch.cat(mean_pos_cluster_ecal, dim=0)
        angles = torch.sum(mean_pos_cluster_ecal*pos_track,dim=1)/(torch.norm(mean_pos_cluster_ecal, dim=1)*torch.norm(pos_track, dim=1))
        angles[torch.isnan(angles)]=0
        
        # if  torch.sum(g.ndata["particle_number"] == 0)==0:
        #     #then index 1 is at 0 
        #     mean_pos_cluster = mean_pos_cluster[1:,:]
        #     particle_track = particle_track-1
        # if mean_pos_cluster.shape[0]> torch.max(particle_track):
        #     distance_track_cluster = torch.norm(mean_pos_cluster[particle_track.long()]-pos_track,dim=1)/1000
        distance_track_cluster = torch.norm(mean_pos_cluster_all-pos_track,dim=1)/1000
        pid = y.pid[particle_track.long()-1]
        pid[particle_track.long()==0]=0
        pid = torch.abs(pid)
        bad_tracks = ((distance_track_cluster>0.24)+(angles<0.9998))*(pid.view(-1)!=13)
        bad_tracks = bad_tracks+((distance_track_cluster>0.5)+(angles<0.99))*(pid.view(-1)==13)
        index_bad_tracks = mask_hit_type_t2.nonzero().view(-1)[bad_tracks]
        
        g.ndata["particle_number"][index_bad_tracks]= 0 
    return g