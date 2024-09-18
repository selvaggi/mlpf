# Similar to evaluate_mix, but plots a comparison between different ML methods on the same plot

import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
#matplotlib.rc("font", size=35)

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

import os

from utils.inference.pandas_helpers import open_hgcal, open_mlpf_dataframe
from utils.inference.per_particle_metrics import (
    plot_per_energy_resolution2_multiple, plot_confusion_matrix,
    plot_efficiency_all, calc_unit_circle_dist
)
import matplotlib.pyplot as plt
import mplhep as hep
import torch
import pickle

hep.style.use("CMS")
colors_list = ["#deebf7", "#9ecae1", "#d415bd"]  # color list Jan
all_E = True
neutrals_only = False
log_scale = False
tracks = True
perfect_pid = False # Pretend we got ideal PID and rescale the momentum vectors accordingly
mass_zero = False   # Set the mass to zero for all particles
ML_pid = True       # Use the PID from the ML classification head (electron/CH/NH/gamma)

if all_E:
    PATH_store = (
        "/eos/user/g/gkrzmanc/eval_plots_EC/eval_10_09_testset_300_files_avg_pos"
    )
    if not os.path.exists(PATH_store):
        os.makedirs(PATH_store)
    plots_path = os.path.join(PATH_store, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    path_list = [
        "results/eval_10_09_testset_300_files_avg_pos/showers_df_evaluation/0_0_None_hdbscan.pt"
    ]
    path_pandora = "results/eval_10_09_testset_300_files_avg_pos/showers_df_evaluation/0_0_None_pandora.pt"
    dir_top = "/eos/user/g/gkrzmanc/2024/"
    print(PATH_store)

labels = [
    "ML"
]

def filter_df(df):
    # quick filter to exclude problematic particles
    df = df[(df.pid != 11) & (df.pid != 22) ]
    return df

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
    # filter out photons with tracks in cluster
    #sd_pandora = sd_pandora[~((sd_pandora["pid"] == 22) & (sd_pandora["is_track_in_cluster"] == 1.0))]
    #sd_hgb = sd_hgb
    # _hgb["pid"] == 22) & (sd_hgb["is_track_in_cluster"] == 1.0))]
    plot_efficiency_all(sd_pandora, df_list, PATH_store, labels)
    plot_confusion_matrix(sd_hgb, PATH_store)
    plot_per_energy_resolution2_multiple(
        sd_pandora,
        {"ML": sd_hgb},
        os.path.join(PATH_store, "plots"),
        tracks=tracks,
        perfect_pid=perfect_pid,
        mass_zero=mass_zero,
        ML_pid=ML_pid,
    )
    print("Done")


if __name__ == "__main__":
    main()

def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di

