import numpy as np
import torch
import dgl
from torch_scatter import scatter_add, scatter_sum
from sklearn.preprocessing import StandardScaler

from torch_scatter import scatter_sum


def get_number_hits(e_hits, part_idx):
    number_of_hits = scatter_sum(torch.ones_like(e_hits), part_idx.long(), dim=0)
    return (number_of_hits[1:].flatten()).tolist()


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


def create_inputs_from_table(output):
    number_hits = np.int32(np.sum(output["pf_mask"][0]))
    # print("number_hits", number_hits)
    number_part = np.int32(np.sum(output["pf_mask"][1]))
    #! idx of particle does not start at 1
    hit_particle_link = torch.tensor(output["pf_vectoronly"][0, 0:number_hits])
    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link)
    features_hits = torch.permute(
        torch.tensor(output["pf_features"][:, 0:number_hits]), (1, 0)
    )
    pos_hits = torch.permute(
        torch.tensor(output["pf_points"][:, 0:number_hits]), (1, 0)
    )
    hit_type = features_hits[:, -1].clone()
    print(hit_type)
    hit_type_one_hot = torch.nn.functional.one_hot(hit_type.long(), num_classes=2)
    # build the features (theta,phi,p)

    # features particles
    unique_list_particles = torch.Tensor(unique_list_particles).to(torch.int64)

    features_particles = torch.permute(
        torch.tensor(output["pf_vectors"][:, list(unique_list_particles)]),
        (1, 0),
    )

    # y_mass = features_particles[:, 3].view(-1).unsqueeze(1)
    # y_mom = features_particles[:, 2].view(-1).unsqueeze(1)
    # y_energy = torch.sqrt(y_mass**2 + y_mom**2)

    y_data_graph = features_particles

    assert len(y_data_graph) == len(unique_list_particles)

    result = [
        number_hits,
        number_part,
        y_data_graph,
        hit_type_one_hot,  # [no_tracks],
        cluster_id,
        hit_particle_link,
        features_hits,
    ]
    return result


def create_graph_tracking(
    output,
):

    (
        number_hits,
        number_part,
        y_data_graph,
        hit_type_one_hot,  # [no_tracks],
        cluster_id,
        hit_particle_link,
        features_hits,
    ) = create_inputs_from_table(output)

    if hit_type_one_hot.shape[0] > 0:
        graph_empty = False

        g = dgl.DGLGraph()
        g.add_nodes(hit_type_one_hot.shape[0])

        hit_features_graph = torch.cat(
            (features_hits, hit_type_one_hot), dim=1
        )  # dims = 9
        #! currently we are not doing the pid or mass regression
        g.ndata["h"] = hit_features_graph
        g.ndata["hit_type"] = hit_type_one_hot
        g.ndata["particle_number"] = cluster_id
        g.ndata["particle_number_nomap"] = hit_particle_link
        if len(y_data_graph) < 4:
            graph_empty = True
    else:
        graph_empty = True
        g = 0
        y_data_graph = 0
    if features_hits.shape[0] < 10:
        graph_empty = True

    return [g, y_data_graph], graph_empty
