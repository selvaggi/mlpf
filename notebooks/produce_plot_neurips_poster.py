import torch
import sys
import os.path as osp
import os
import sys
import numpy as np

sys.path.append("/afs/cern.ch/work/m/mgarciam/private/mlpf/")
from src.dataset.dataset import SimpleIterDataset
from src.utils.utils import to_filelist
from torch.utils.data import DataLoader

# import dgl  # CPU only version for now
from tqdm import tqdm
from torch_scatter import scatter_sum
import matplotlib.pyplot as plt
import pickle
import numpy as np
import mplhep as hep
import plotly
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

hep.style.use("CMS")
import matplotlib

matplotlib.rc("font", size=40)


datasets = {
    "train": [
        "/eos/user/m/mgarciam/datasets_mlpf/all_energies_10_15/pf_tree_1.root",
        "/eos/user/m/mgarciam/datasets_mlpf/all_energies_10_15/pf_tree_2.root",
        "/eos/user/m/mgarciam/datasets_mlpf/all_energies_10_15/pf_tree_3.root"
        "/eos/user/m/mgarciam/datasets_mlpf/all_energies_10_15/pf_tree_4.root",
    ],
    "test": [
        "/eos/user/m/mgarciam/datasets_mlpf/all_energies_10_15/pf_tree_1.root",
        "/eos/user/m/mgarciam/datasets_mlpf/all_energies_10_15/pf_tree_2.root",
    ],
}

print(datasets["train"])


def get_ratios(g, y, corr_w_mass=False, return_pid_dict=False):
    part_idx = g.ndata["particle_number"]
    true_idx = np.arange(len(part_idx))
    part_idx = part_idx[true_idx]
    hit_types = g.ndata["hit_type"][true_idx]
    hit_idx = torch.where((hit_types[:, 2] == 1) | (hit_types[:, 3] == 1))[0]
    track_idx = torch.where((hit_types[:, 0] == 1) | (hit_types[:, 1] == 1))[0]
    hit_energies = g.ndata["e_hits"].flatten()[true_idx]  # [hit_idx]
    where_e_zero = hit_energies == 0
    hit_momenta = g.ndata["p_hits"].flatten()[true_idx]  # [track_idx]
    energy_from_showers = scatter_sum(hit_energies, part_idx.long(), dim=0)
    hits = torch.ones_like(hit_energies)
    hits_from_showers = scatter_sum(hits, part_idx.long(), dim=0)
    y_energy = y[:, 3]
    y_pid = y[:, -1].to(torch.long)
    energy_from_showers = energy_from_showers[1:]
    hits_from_showers = hits_from_showers[1:]
    assert len(energy_from_showers) > 0
    if return_pid_dict:
        pids = y_pid.unique().long()
        pid_dict = {
            int(pid): (
                energy_from_showers[y_pid == pid] / y_energy[y_pid == pid]
            ).tolist()
            for pid in pids
        }
        pid_dict_hits = {
            int(pid): (hits_from_showers[y_pid == pid]).tolist() for pid in pids
        }
        energy_total = {int(pid): (y_energy[y_pid == pid]).tolist() for pid in pids}
        pid_dict["ALL"] = (energy_from_showers / y_energy).tolist()
        return pid_dict, pid_dict_hits, energy_total
    return (energy_from_showers / y_energy).tolist()


class Args:
    def __init__(self, datasets):
        self.data_train = datasets
        self.data_val = datasets
        # self.data_train = files_train
        self.data_config = "/afs/cern.ch/work/m/mgarciam/private/mlpf/config_files/config_2_newlinks.yaml"
        self.extra_selection = None
        self.train_val_split = 1
        self.data_fraction = 1
        self.file_fraction = 1
        self.fetch_by_files = False
        self.fetch_step = 0.01
        self.steps_per_epoch = None
        self.in_memory = False
        self.local_rank = None
        self.copy_inputs = False
        self.no_remake_weights = False
        self.batch_size = 10
        self.num_workers = 0
        self.demo = False
        self.laplace = False
        self.diffs = False
        self.class_edges = False


args = {key: Args(value) for key, value in datasets.items()}

datas = {}
files_dict = {}
for key in datasets:
    train_range = (0, args[key].train_val_split)
    train_file_dict, train_files = to_filelist(args[key], "train")
    train_data = SimpleIterDataset(
        train_file_dict,
        args[key].data_config,
        for_training=True,
        extra_selection=args[key].extra_selection,
        remake_weights=True,
        load_range_and_fraction=(train_range, args[key].data_fraction),
        file_fraction=args[key].file_fraction,
        fetch_by_files=args[key].fetch_by_files,
        fetch_step=args[key].fetch_step,
        infinity_mode=False,
        in_memory=args[key].in_memory,
        async_load=False,
        name="train",
    )
    datas[key] = train_data
    files_dict[key] = train_files


iterators = {key: iter(val) for key, val in datas.items()}

from src.dataset.functions_graph import graph_batch_func

train_loaders = {
    key: DataLoader(
        datas[key],
        batch_size=1,
        drop_last=True,
        pin_memory=True,
        num_workers=min(
            args[key].num_workers, int(len(files_dict[key]) * args[key].file_fraction)
        ),
        collate_fn=graph_batch_func,
        persistent_workers=args[key].num_workers > 0
        and args[key].steps_per_epoch is not None,
    )
    for key in args
}

iterators = {key: iter(item) for key, item in train_loaders.items()}


from torch_scatter import scatter_sum

all_ratios = {}
all_number_hits = {}
all_energy = {}
num_particles = {}
num_hits = {}
particle_energy = {}
hits_energy_ecal = {}
hits_energy_hcal = {}


def upd_dict(d, small_dict):
    for k in small_dict:
        if k not in d:
            d[k] = []
        d[k] += small_dict[k]
    return d


hits_energy_ecal = []
hits_energy_hcal = []
for i in tqdm(range(99 * 4)):
    key = "train"
    g, y = next(iterators[key])
    num_part = y.shape[0]
    if key not in num_particles:
        num_particles[key] = []
        num_hits[key] = []
        particle_energy[key] = []
        # hits_energy_ecal[key] = []
        # hits_energy_hcal[key] = []
    num_particles[key].append(num_part)
    num_hits[key].append(g.ndata["particle_number"].shape[0])
    particle_energy[key] += y[:, 3].tolist()
    hits_energy = g.ndata["e_hits"].flatten().tolist()
    ecal_hits_filter = g.ndata["hit_type"][:, 2] == 1
    hits_energy_ecal.append(np.array(hits_energy)[ecal_hits_filter].tolist())
    hits_energy_hcal.append(np.array(hits_energy)[~ecal_hits_filter].tolist())

    # ratios, number_of_hits, energy_total = get_ratios(
    #     g, y, corr_w_mass=True, return_pid_dict=True
    # )
    # all_ratios = upd_dict(all_ratios, ratios)
    # all_number_hits = upd_dict(all_number_hits, number_of_hits)
    # all_energy = upd_dict(all_energy, energy_total)
    # TODOs tmrw: check particle number dist., how many are we throwing away
print("len of hgcal", len(hits_energy_hcal))
hits_energy_hcal = np.concatenate(hits_energy_hcal)
hits_energy_ecal = np.concatenate(hits_energy_ecal)
print("number of hits", len(hits_energy_hcal))
import matplotlib.pyplot as plt
import seaborn as sns

colors_list = ["#fff7bc", "#fec44f", "#d95f0e"]
fig = plt.figure()
sns.histplot(
    hits_energy_hcal,
    label="HCAL",
    element="step",
    color=colors_list[1],
    stat="percent",
    fill=False,
    binwidth=0.05,
    linewidth=3,
)
sns.histplot(
    hits_energy_ecal,
    label="ECAL",
    element="step",
    stat="percent",
    color=colors_list[2],
    fill=False,
    binwidth=0.05,
    linewidth=3,
)
plt.ylabel("count")
plt.xlabel("Hit Energy")
plt.yscale("log")
plt.legend(loc="upper right")
fig.tight_layout()
fig.savefig("/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/hit_energy.png")

# plot the total number of hits per particle
# import seaborn as sns

# all_number_hits = {str(k): v for k, v in all_number_hits.items()}
# keys = sorted(list(all_number_hits.keys()))
# fig, ax = plt.subplots(len(keys), figsize=(5, 13))

# for i in range(len(keys)):
#     sns.histplot(x=all_ratios[keys[i]], y=all_number_hits[keys[i]], ax=ax[i])
#     ax[i].set_xlabel("count")
#     ax[i].set_ylabel(r"$number hits$")
#     ax[i].set_title(f"{keys[i]}")

# fig.tight_layout()
# fig.savefig(
#     "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/220823_condor_spread_20_25_10k_number_hits.png"
# )


# all_energy = {str(k): v for k, v in all_energy.items()}
# keys = sorted(list(all_energy.keys()))
# fig, ax = plt.subplots(len(keys), figsize=(5, 13))

# for i in range(len(keys)):
#     e_reco = np.array(all_ratios[keys[i]]) * np.array(all_energy[keys[i]])
#     sns.histplot(x=all_energy[keys[i]], y=e_reco, ax=ax[i])
#     ax[i].set_xlabel("E gen")
#     ax[i].set_ylabel(r"$E reco$")
#     ax[i].set_title(f"{keys[i]}")

# fig.tight_layout()
# fig.savefig(
#     "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/220823_condor_spread_20_25_10k_Ereco_Egen.png"
# )


# fig, ax = plt.subplots(len(keys), figsize=(5, 13))

# for i in range(len(keys)):
#     e_reconstructed = np.array(all_ratios[keys[i]])
#     e_total = np.array(all_energy[keys[i]])
#     mask = e_total < 20
#     sns.histplot(e_reconstructed[mask], ax=ax[i])
#     ax[i].set_xlabel("E gen")
#     ax[i].set_ylabel(r"$E reco$")
#     ax[i].set_title(f"{keys[i]}")

# fig.tight_layout()
# fig.savefig(
#     "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/220823_condor_spread_20_25_10k_Epartition_small.png"
# )


# fig, ax = plt.subplots(len(keys), figsize=(5, 13))

# for i in range(len(keys)):
#     e_reconstructed = np.array(all_ratios[keys[i]])
#     e_total = np.array(all_energy[keys[i]])
#     mask = e_total > 20
#     sns.histplot(e_reconstructed[mask], ax=ax[i])
#     ax[i].set_xlabel("E reconstructed")
#     ax[i].set_ylabel(r"$counts$")
#     ax[i].set_title(f"{keys[i]}")

# fig.tight_layout()
# fig.savefig(
#     "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/220823_condor_spread_20_25_10k_Epartition_big.png"
# )
