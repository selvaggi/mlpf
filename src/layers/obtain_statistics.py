import torch
from torch_scatter import scatter_max, scatter_add, scatter_mean
import numpy as np
import dgl
import matplotlib.pyplot as plt
import os


def obtain_statistics_graph(stat_dict, y_all, g_all, pf=True):
    graphs = dgl.unbatch(g_all)
    batch_id = y_all[:, -1].view(-1)
    for i in range(0, len(graphs)):
        mask = batch_id == i
        y = y_all[mask]
        g = graphs[i]
        number_of_particles_event = len(y)
        if pf:
            energy_particles = y[:, 3]
        else:
            energy_particles = y[:, 3]

        # obtain stats about particles and energy of the particles
        stat_dict["freq_count_particles"][number_of_particles_event] = (
            stat_dict["freq_count_particles"][number_of_particles_event] + 1
        )
        stat_dict["freq_count_energy"] = stat_dict["freq_count_energy"] + torch.histc(
            energy_particles, bins=500, min=0.001, max=50
        )

        # obtain angle stats
        # if pf:
        #     cluster_space_coords = g.ndata["pos_hits_xyz"]
        #     object_index = g.ndata["particle_number"].view(-1)
        #     x_alpha_sum = scatter_mean(cluster_space_coords, object_index.long(), dim=0)
        #     nVs = x_alpha_sum[1:] / torch.norm(
        #         x_alpha_sum[1:], p=2, dim=-1, keepdim=True
        #     )
        #     # compute cosine of the angles using dot product
        #     cos_ij = torch.einsum("ij,pj->ip", nVs, nVs)
        #     min_cos_per_particle = torch.min(torch.abs(cos_ij), dim=0)[0]
        #     stat_dict["freq_count_angle"] = stat_dict["freq_count_angle"] + torch.histc(
        #         min_cos_per_particle, bins=10, min=0, max=1.1
        #     )
        # else:
        eta = y[:, 0]
        phi = y[:, 1]
        len_y = len(eta)
        dr_matrix = torch.sqrt(
            torch.square(
                torch.tile(eta.view(1, -1), (len_y, 1))
                - torch.tile(eta.view(-1, 1), (1, len_y))
            )
            + torch.square(
                torch.tile(phi.view(1, -1), (len_y, 1))
                - torch.tile(phi.view(-1, 1), (1, len_y))
            )
        )
        device = y.device
        dr_matrix = dr_matrix + torch.eye(len_y, len_y).to(device) * 10
        min_cos_per_particle = torch.min(dr_matrix, dim=1)[0]
        stat_dict["freq_count_angle"] = stat_dict["freq_count_angle"] + torch.histc(
            min_cos_per_particle, bins=40, min=0, max=4
        )
        return stat_dict


def create_stats_dict(device):
    bins_number_of_particles_event = torch.arange(0, 200, 1).to(device)
    freq_count_particles = torch.zeros_like(bins_number_of_particles_event)
    # the reason to not do log is that the histc only takes min, max, numbins and the other hist with bins is not supported in cuda
    energy_event = torch.arange(0.001, 50, 0.1).to(
        device
    )  # torch.exp(torch.arange(np.log(0.001), np.log(50), 0.1))
    freq_count_energy = torch.zeros(len(energy_event)).to(device)
    angle_distribution = torch.arange(0, 4 + 0.1, 0.1).to(device)
    freq_count_angle = torch.zeros(len(angle_distribution) - 1).to(device)
    stat_dict = {}
    stat_dict["bins_number_of_particles_event"] = bins_number_of_particles_event
    stat_dict["freq_count_particles"] = freq_count_particles
    stat_dict["energy_event"] = energy_event
    stat_dict["freq_count_energy"] = freq_count_energy
    stat_dict["angle_distribution"] = angle_distribution
    stat_dict["freq_count_angle"] = freq_count_angle
    return stat_dict


def save_stat_dict(stat_dict, path):
    path = path + "/stat_dict.pt"
    torch.save(stat_dict, path)

def stacked_hist_plot(lst, lst_pandora, path_store, title):
    # lst is a list of arrays. plot them in a stacked histogram with the same x-axis
    fig, ax = plt.subplots(len(lst), 1, sharex=True)
    bins = np.linspace(-0.1, 0.1, 200)
    if len(lst) == 1:
        ax = [ax]
    for i in range(len(lst)):
        ax[i].hist(lst[i], bins, histtype="step", label="ML", color="red")
        ax[i].hist(lst_pandora[i], bins, histtype="step", label="Pandora", color="blue")
        ax[i].legend()
        ax[i].grid()
        ax[i].set_yscale("log")
    ax[-1].set_xlabel("angle difference")
    fig.suptitle(title)
    fig.savefig(os.path.join(path_store, title + "_angle_distributions.pdf"))

def plot_distributions(stat_dict, PATH_store, pf=False):
    # energy per event
    print(PATH_store)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    b = stat_dict["freq_count_energy"] / torch.sum(stat_dict["freq_count_energy"])
    a = stat_dict["energy_event"]
    a = a.detach().cpu()
    b = b.detach().cpu()
    axs[0].bar(a, b, width=0.2)
    axs[0].set_title("Energy distribution")
    b = stat_dict["freq_count_angle"] / torch.sum(stat_dict["freq_count_angle"])
    a = stat_dict["angle_distribution"][:-1]
    a = a.detach().cpu()
    b = b.detach().cpu()
    axs[1].bar(a, b, width=0.02)
    axs[1].set_xlim([0, 1])
    axs[1].set_title("Angle distribution")
    # axs[1].set_ylim([0,1])
    b = stat_dict["freq_count_particles"] / torch.sum(stat_dict["freq_count_particles"])
    a = stat_dict["bins_number_of_particles_event"]
    a = a.detach().cpu()
    b = b.detach().cpu()
    axs[2].bar(a, b)
    axs[2].set_title("number of particles")
    # fig.suptitle('Stats event')
    fig.savefig(
        PATH_store + "/stats.png",
        bbox_inches="tight",
    )
