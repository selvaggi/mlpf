# Similar to evaluate_mix, but plots a comparison between different ML methods on the same plot
import matplotlib
matplotlib.rc("font", size=35)
import os

from utils.inference.pandas_helpers import open_hgcal, open_mlpf_dataframe
from utils.inference.per_particle_metrics import (
    plot_per_energy_resolution2_multiple,
    plot_efficiency_all,
)
import matplotlib.pyplot as plt
import mplhep as hep
import torch
import pickle


hep.style.use("CMS")
colors_list = ["#deebf7", "#9ecae1", "#3182bd"]  # color list Jan
all_E = True
neutrals_only = False
log_scale = False
tracks = True
if all_E:
    PATH_store = (
        "/eos/user/g/gkrzmanc/eval_plots_EC/eval_gnn_3004_l1_training/eval_comparison"
    )
    if not os.path.exists(PATH_store):
        os.makedirs(PATH_store)
    plots_path = os.path.join(PATH_store, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    path_list = [
        "eval_gnn_3004_l1_training/showers_df_evaluation/0_0_None_hdbscan.pt",
        "eval_dnn_3004_l1_training_longereval/showers_df_evaluation/0_0_None_hdbscan.pt",
    ]
    path_pandora = "eval_gnn_3004_l1_training/showers_df_evaluation/0_0_None_pandora.pt"
    dir_top = "/eos/user/g/gkrzmanc/2024/"

labels = [
    "GNN+DNN",
    "DNN"
]

def main():
    df_list = []
    matched_all = {}
    for idx, i in enumerate(path_list):
        path_hgcal = os.path.join(dir_top, i)
        sd_hgb, matched_hgb = open_mlpf_dataframe(path_hgcal, neutrals_only)
        df_list.append(sd_hgb)
        matched_all[labels[idx]] = matched_hgb

    sd_pandora, matched_pandora = open_mlpf_dataframe(
        dir_top + path_pandora, neutrals_only
    )
    print("finished collection of data and started plotting")
    #plot_efficiency_all(sd_pandora, df_list, PATH_store, labels)
    plot_per_energy_resolution2_multiple(
        matched_pandora,
        matched_all,
        os.path.join(PATH_store, "plots"),
        tracks=tracks,
    )

if __name__ == "__main__":
    main()
def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di

