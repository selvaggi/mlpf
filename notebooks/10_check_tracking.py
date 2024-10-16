import torch
import sys
import os.path as osp
import os
import sys
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px

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


hep.style.use("CMS")
import matplotlib

matplotlib.rc("font", size=13)

# This block is the same as 1_dataset.ipynb


def plot_event(g, y, i):
    from src.logger.plotting_tools import shuffle_truth_colors

    tidx = g.ndata["particle_number"]
    X = g.ndata["h"][:, 0:3]
    Edep = (g.ndata["h"][:, -2] + 2) * 0.5
    data = {
        "X": X[:, 0].view(-1, 1).detach().cpu().numpy(),
        "Y": X[:, 1].view(-1, 1).detach().cpu().numpy(),
        "Z": X[:, 2].view(-1, 1).detach().cpu().numpy(),
        "tIdx": tidx.view(-1, 1).detach().cpu().numpy(),
        "features": Edep.view(-1, 1).detach().cpu().numpy(),
    }
    hoverdict = {}
    # if hoverfeat is not None:
    #     for j in range(hoverfeat.shape[1]):
    #         hoverdict["f_" + str(j)] = hoverfeat[:, j : j + 1]
    #     data.update(hoverdict)

    # if nidx is not None:
    #     data.update({"av_same": av_same})

    df = pd.DataFrame(
        np.concatenate([data[k] for k in data], axis=1),
        columns=[k for k in data],
    )
    df["orig_tIdx"] = df["tIdx"]
    rdst = np.random.RandomState(1234567890)  # all the same
    shuffle_truth_colors(df, "tIdx", rdst)

    # hover_data = ["orig_tIdx", "idx"] + [k for k in hoverdict.keys()]
    # if nidx is not None:
    #     hover_data.append("av_same")
    fig = px.scatter_3d(
        df,
        x="X",
        y="Y",
        z="Z",
        color="tIdx",
        size="features",
        # hover_data=hover_data,
        template="plotly_dark",
        color_continuous_scale=px.colors.sequential.Rainbow,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.write_html("/eos/user/m/mgarciam/datasets_tracking/event" + str(i) + ".html")


def main():
    datasets = {
        "train": "/eos/user/m/mgarciam/datasets_tracking/eeTojj_IDEA_sim_edm4hep_small_processed.root",
        "test": "/eos/user/m/mgarciam/datasets_tracking/eeTojj_IDEA_sim_edm4hep_small_processed.root",
    }

    class Args:
        def __init__(self, datasets):
            self.data_train = [datasets]
            self.data_val = [datasets]
            # self.data_train = files_train
            self.data_config = "/afs/cern.ch/work/m/mgarciam/private/mlpf/config_files/config_tracking.yaml"
            self.extra_selection = None
            self.train_val_split = 0.8
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

    from src.dataset.functions_graph import graph_batch_func

    train_loaders = {
        key: DataLoader(
            datas[key],
            batch_size=1,
            drop_last=True,
            pin_memory=True,
            num_workers=min(
                args[key].num_workers,
                int(len(files_dict[key]) * args[key].file_fraction),
            ),
            collate_fn=graph_batch_func,
            persistent_workers=args[key].num_workers > 0
            and args[key].steps_per_epoch is not None,
        )
        for key in args
    }

    iterators = {key: iter(item) for key, item in train_loaders.items()}

    itera = iter(train_loaders["train"])

    for i in range(0, 5):
        g, y = next(itera)
        plot_event(g, y, i)


if __name__ == "__main__":
    main()
