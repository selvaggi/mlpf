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
from src.utils.inference.event_Ks import get_decay_type
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
perfect_pid = False # pretend we got ideal PID and rescale the momentum vectors accordingly
mass_zero = False   # set the mass to zero for all particles
ML_pid = True       # Use the PID from the ML classification head (electron/CH/NH/gamma)

if all_E:
    PATH_store = (
        "/eos/user/g/gkrzmanc/eval_plots_EC/FixAvg_with_vertex_info_eval_Ks_13_09_FT_on_Ks/eval_Close_decays_oonly"
    )
    if not os.path.exists(PATH_store):
        os.makedirs(PATH_store)
    plots_path = os.path.join(PATH_store, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    path_list = [
        "FixAvg_with_vertex_info_eval_Ks_13_09_FT_on_Ks/showers_df_evaluation/0_0_None_hdbscan.pt"
    ]
    path_pandora = "FixAvg_with_vertex_info_eval_Ks_13_09_FT_on_Ks/showers_df_evaluation/0_0_None_pandora.pt"
    dir_top = "/eos/user/g/gkrzmanc/eval_plots_EC/"
    print(PATH_store)

labels = [
    "ML"
]

def renumber_batch_idx(df):
    # batch idx has missing numbers
    # renumber it to be like 0,1,2...
    batch_idx = df.number_batch
    unique_batch_idx = np.unique(batch_idx)
    new_to_old_batch_idx = {}
    new_batch_idx = np.zeros(len(batch_idx))
    for idx, i in enumerate(unique_batch_idx):
        new_batch_idx[batch_idx == i] = idx
        new_to_old_batch_idx[idx] = i
    df.number_batch = new_batch_idx
    return df


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
    decay_type = get_decay_type(sd_hgb)
    decay_type_pandora = get_decay_type(sd_pandora)
    print("!!! Filtering !!!")
    # Plot distance from (0,0,0) for the matched particles
    pandora_vertex = np.array(sd_pandora.vertex.values.tolist())
    # drop nan values
    mask_nan = np.isnan(pandora_vertex).any(axis=1)
    pandora_vertex = pandora_vertex[~mask_nan]
    hgb_vertex = np.array(sd_hgb.vertex.values.tolist())
    mask_nan_hgb = np.isnan(hgb_vertex).any(axis=1)
    hgb_vertex = hgb_vertex[~mask_nan_hgb]
    displacement_pandora = np.linalg.norm(pandora_vertex, axis=1)
    displacement_hgb = np.linalg.norm(hgb_vertex, axis=1)
    #sd_pandora = sd_pandora[decay_type_pandora == 1]
    #sd_hgb = sd_hgb[decay_type == 1]
    allowed_batch_idx = np.where(decay_type_pandora == 1)[0]
    allowed_batch_idx_pandora = np.where(decay_type == 1)[0]
    sd_pandora = sd_pandora[sd_pandora.number_batch.isin(allowed_batch_idx)]
    sd_hgb = sd_hgb[sd_hgb.number_batch.isin(allowed_batch_idx_pandora)]

    # filter the df based on where decay type is 0
    #allowed_batch_idx = np.where(displacement_hgb < 200)[0]
    #sd_hgb = sd_hgb[sd_hgb.number_batch.isin(allowed_batch_idx)]
    #allowed_batch_idx_pandora = np.where(displacement_pandora < 200)[0]
    #sd_pandora = sd_pandora[sd_pandora.number_batch.isin(allowed_batch_idx_pandora)]
    sd_pandora = renumber_batch_idx(sd_pandora)
    sd_hgb = renumber_batch_idx(sd_hgb)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    bins = np.linspace(0, 300, 200)
    ax.hist(displacement_pandora, bins=bins, histtype="step", label="Pandora", color="blue")
    ax.hist(displacement_hgb, bins=bins, histtype="step", label="ML", color="red")
    ax.legend()
    ax.set_yscale("log")
    fig.show()
    # Filter - only charged decays
    print("finished collection of data and started plotting")
    plot_efficiency_all(sd_pandora, df_list, PATH_store, labels)
    #plot_confusion_matrix(sd_hgb, PATH_store)
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

