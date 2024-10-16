import torch
import sys
import os.path as osp
import os
import sys
import numpy as np
import dgl

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

matplotlib.rc("font", size=13)

datasets = {
    "train": "/eos/user/m/mgarciam/datasets_mlpf/condor/condor_dataset_test_large_restrictedE_1/1/tree_1.root",
    "test": "/eos/user/m/mgarciam/datasets_mlpf/condor/condor_dataset_test_large_restrictedE_1/1/tree_1.root",
}


class Args:
    def __init__(self, datasets):
        self.data_train = [datasets]
        self.data_val = [datasets]
        # self.data_train = files_train
        self.data_config = "/afs/cern.ch/work/m/mgarciam/private/mlpf/config_files/config_2_newlinks.yaml"
        self.extra_selection = None
        self.train_val_split = 0.99
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
        batch_size=10,
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

from src.dataset.functions_graph import get_ratios, get_number_hits

itera = iter(train_loaders["train"])
g, y = next(itera)
print(len(dgl.unbatch(g)))
list_graphs = dgl.unbatch(g)
for l in list_graphs:
    print(l.number_of_nodes())
# print("hereee")
# for i in range(0, 5):
#     g_i = list_graphs[i]
#     mask = y[:, -1] == i
#     y_i = y[mask]
#     e_hits = g_i.ndata["e_hits"]

#     p_id = g_i.ndata["particle_number"]
#     print("_________________________event___________________", i)
#     print(e_hits[0:10])
#     ratios = get_ratios(e_hits, p_id, y_i)
#     print(ratios)
#     n_hits = get_number_hits(e_hits, p_id)
#     print(n_hits)
#     print(y_i[:, -3:])
#     print("particle energy", y_i[:, 3])
