import gzip
import pickle
import matplotlib

matplotlib.rc("font", size=35)
import numpy as np
import pandas as pd
import os
import numpy as np
from utils.inference.inference_metrics_hgcal import obtain_metrics_hgcal
from utils.inference.inference_metrics import obtain_metrics
from utils.inference.pandas_helpers import open_hgcal, open_mlpf_dataframe
from utils.inference.event_Ks import mass_Ks
from utils.inference.per_particle_metrics import (
    calc_unit_circle_dist, plot_per_energy_resolution2
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
    PATH_store = "/eos/user/g/gkrzmanc/eval_plots_EC/Ks_eval_reprod_3_6/eval_plots_2"
    # PATH_store = "/eos/user/g/gkrzmanc/eval_plots_EC/Ks_GATr_EP_regression_with_ML_model_clustering_1_without_normalization/results_3005"
    # PATH_store = "/eos/user/g/gkrzmanc/eval_plots_EC/Ks_GATr_EP_regression_with_ML_model_clustering_1_without_normalization_train_with_no_GT/results3005"
    #PATH_store = "/eos/user/g/gkrzmanc/eval_plots_EC/Ks_GATr_eval_3105_E_p_regression_fixbug/eval_3105_old_fn"
    if not os.path.exists(PATH_store):
        os.makedirs(PATH_store)
    plots_path = os.path.join(PATH_store, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    path_list = [
        "Ks_eval_reprod_3_6/showers_df_evaluation/0_0_None_hdbscan.pt",
    ]
    path_pandora = "Ks_eval_reprod_3_6/showers_df_evaluation/0_0_None_pandora.pt"
    dir_top = "/eos/user/g/gkrzmanc/eval_plots_EC/"

labels = [
    "gatr1_250324_E_cont",
]

def main():
    df_list = []
    for i in path_list:
        path_hgcal = dir_top + i
        sd_hgb, matched_hgb = open_mlpf_dataframe(path_hgcal, neutrals_only)
        df_list.append(sd_hgb)

    sd_pandora, matched_pandora = open_mlpf_dataframe(
        dir_top + path_pandora, neutrals_only
    )

    print("finished collection of data and started plotting")
    #mass_Ks(sd_pandora, sd_hgb, PATH_store)
    # plot_efficiency_all(sd_pandora, df_list, PATH_store, labels)
    # plot_per_energy_resolution2(
    #     sd_pandora,
    #     sd_hgb,
    #     matched_pandora,
    #     matched_hgb,
    #     os.path.join(PATH_store, "plots"),
    #     tracks=trackscy_all(sd_pandora, df_list, PATH_store, labels)
    plot_per_energy_resolution2(
         sd_pandora,
         sd_hgb,
         matched_pandora,
         matched_hgb,
         os.path.join(PATH_store, "plots"),
         tracks=tracks,
     )
    dist_pandora, pids = calc_unit_circle_dist(matched_pandora, pandora=True)
    dist_ml, pids_ml = calc_unit_circle_dist(matched_hgb, pandora=False)
    for pid in [22, -211, 211]:
        # plot histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        bins = np.linspace(0, 1, 100)
        ax.hist(
            dist_pandora[np.where(pids == pid)],
            bins=bins,
            histtype="step",
            label="Pandora",
            color="blue",
        )
        ax.hist(
            dist_ml[np.where(pids_ml == pid)],
            bins=bins,
            histtype="step",
            label="Model",
            color="red",
        )
        ax.set_xlabel("Distance to unit circle")
        ax.set_yscale("log")
        ax.legend()
        fig.savefig(os.path.join(PATH_store, f"unit_circle_dist_{pid}.pdf"))
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
