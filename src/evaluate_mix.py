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
        "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/comparison_plots/"
    )
    path_list = [
        "gatr1_200324_E/showers_df_evaluation/0_0_None_hdbscan.pt",
    ]
    path_pandora = "gatr1_200324_E/showers_df_evaluation/" + "0_0_None_pandora.pt"
    dir_top = "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/"

labels = ["gatr1_200324_E"]


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
    plot_efficiency_all(sd_pandora, df_list, PATH_store, labels)
    # plot_per_energy_resolution2(
    #     sd_pandora,
    #     sd_hgb,
    #     matched_pandora,
    #     matched_hgb,
    #     PATH_store + "plots/",
    #     tracks=tracks,
    # )
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
