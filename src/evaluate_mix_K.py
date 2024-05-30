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
from utils.inference.plots import (
    plot_metrics,
    plot_histograms_energy,
    plot_correction,
    plot_for_jan,
)
from utils.inference.event_metrics import plot_per_event_metrics
from utils.inference.per_particle_metrics import (
    plot_per_energy_resolution2,
    plot_efficiency_all,
    calc_unit_circle_dist
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
        "/eos/user/g/gkrzmanc/eval_plots_EC/Ks_GATr_EP_regression_with_GT_clustering_1_without_normalization/eval_results_2905_vectors"
    )
    if not os.path.exists(PATH_store):
        os.makedirs(PATH_store)
    plots_path = os.path.join(PATH_store, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    path_list = [
        "Ks_GATr_EP_regression_with_GT_clustering_1_without_normalization/showers_df_evaluation/0_0_None_hdbscan.pt",
    ]
    path_pandora = "Ks_GATr_EP_regression_with_GT_clustering_1_without_normalization/showers_df_evaluation/0_0_None_pandora.pt"
    dir_top = "/eos/user/g/gkrzmanc/eval_plots_EC/"
    # path_list = [
    #     "030424_gatr_dr05/showers_df_evaluation/0_0_None_hdbscan.pt",
    # ]
    # path_pandora = "030424_gatr_dr05/showers_df_evaluation/" + "0_0_None_pandora.pt"

labels = [
    # "gatr1_200324_E",
    # "gatr1_210324_E_v1",
    # "gravnet_250324_1E",
    # "gatr1_250324_E",
    "gatr1_250324_E_cont",
    # "280324_gravnet"
    # "030424_gatr_dr05",
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
    #plot_efficiency_all(sd_pandora, df_list, PATH_store, labels)
    plot_per_energy_resolution2(
        sd_pandora,
        sd_hgb,
        matched_pandora,
        matched_hgb,
        os.path.join(PATH_store, "plots"),
        tracks=tracks,
    )
    dist_pandora, pids  = calc_unit_circle_dist(matched_pandora, pandora=True)
    dist_ml, pids_ml = calc_unit_circle_dist(matched_hgb, pandora=False)
    for pid in [22, -211, 211]:
        # plot histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        bins = np.linspace(0, 1, 100)
        ax.hist(dist_pandora[np.where(pids == pid)], bins=bins, histtype="step", label="Pandora", color="blue")
        ax.hist(dist_ml[np.where(pids_ml == pid)], bins=bins, histtype="step", label="Model", color="red")
        ax.set_xlabel("Distance to unit circle")
        ax.set_yscale("log")
        ax.legend()
        fig.savefig(os.path.join(PATH_store, f"unit_circle_dist_{pid}.pdf"))
    print("Done")
    # plot_metrics(
    #     neutrals_only,
    #     dic1,
    #     dic2,
    #     dic3,
    #     dict_1,
    #     dict_2,
    #     dict_3,
    #     colors_list,
    #     PATH_store=PATH_store,
    #     log_scale=log_scale,
    # )
    # # plot_for_jan(neutrals_only, dic1, dic2, dict_1, dict_2, dict_3, colors_list)
    # print("finished metrics")
    # plot_histograms_energy(dic1, dic2, dict_1, dict_2, dict_3, PATH_store=PATH_store)
    # # plot_correction(dic1, dic2, dict_1, dict_2, dict_3, PATH_store=PATH_store)
    # plot_per_event_metrics(sd_hgb, sd_pandora, PATH_store=PATH_store)



if __name__ == "__main__":
    main()
    # 16.01.24 "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/211223_v8/showers_df_evaluation/290124/"
    # "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/hgcal/logs_1015_1911/pandora/analysis/out.bin.gz"
    # evalutation on original test dataset
    # path_mlpf = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/mlpf/mlpf_all_energies_hgcal_loss/showers_df_evaluation/0_0_None.pt"
    # path_pandora = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/mlpf/mlpf_all_energies_hgcal_loss/showers_df_evaluation/0_0_None_pandora.pt"
    # eval removing particles that decay in tracker
    # path_mlpf = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/mlpf/mlpf_all_energies_hgcal_loss/showers_df_evaluation_all_E_notrackeri/0_0_None.pt"
    # path_pandora = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/mlpf/mlpf_all_energies_hgcal_loss/showers_df_evaluation_all_E_notrackeri/0_0_None_pandora.pt"
    # eval with calibration
    # the latest version of this is the v_0
    # path_mlpf = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/mlpf/mlpf_all_energies_hgcal_v3/showers_df_evaluation/0_0_None.pt"
    # path_pandora = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/mlpf/mlpf_all_energies_hgcal_v3/showers_df_evaluation/0_0_None_pandora.pt"


def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di
