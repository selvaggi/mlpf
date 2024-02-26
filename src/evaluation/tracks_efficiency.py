import numpy as np
import matplotlib

matplotlib.rc("font", size=35)
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import concurrent.futures
import time

# TODO paralellize this script or make the data larger so that the binning needed is larger
from scipy.optimize import curve_fit


def plot_efficiency_all(sd_hgb, PATH_store, log):
    all_particles = create_eff_dic(sd_hgb, log)
    plot_eff("efficiency", all_particles, "Photons", PATH_store, log)


def create_eff_dic(matched_, log):
    df_id = matched_
    eff, energy_eff = calculate_eff(df_id, log)
    photons_dic = {}
    photons_dic["eff"] = eff
    photons_dic["energy_eff"] = energy_eff
    return photons_dic


def calculate_eff(sd, log_scale=False):
    if log_scale:
        bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    else:
        bins = np.arange(0, 51, 2)
    eff = []
    energy_eff = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = sd.true_showers_E.values <= bin_i1
        mask_below = sd.true_showers_E.values > bin_i
        mask = mask_below * mask_above
        number_of_non_reconstructed_showers = np.sum(
            np.isnan(sd.pred_showers_E.values)[mask]
        )
        total_showers = len(sd.pred_showers_E.values[mask])
        if total_showers > 0:
            eff.append(
                (total_showers - number_of_non_reconstructed_showers) / total_showers
            )
            energy_eff.append((bin_i1 + bin_i) / 2)

    return eff, energy_eff


def plot_eff(title, photons_dic, label1, PATH_store, log):
    colors_list = ["#FF0000", "#FF0000", "#0000FF"]
    fig = plt.figure()
    j = 0
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Efficiency [GeV]")
    # ax[row_i, j].set_xscale("log")
    plt.title(title)
    plt.grid()
    plt.scatter(
        photons_dic["energy_eff"],
        photons_dic["eff"],
        facecolors=colors_list[1],
        edgecolors=colors_list[1],
        label="ML",
        marker="o",
        s=50,
    )

    # plt.legend(loc="lower right")
    # if title == "Electromagnetic Shower Reconstruction Efficiency":
    #     plt.ylim([0.7, 1.1])
    # else:
    if log:
        log_ = "log"
        # plt.xscale("log")
    else:
        log_ = ""
    plt.ylim([0.5, 1.1])
    fig.savefig(
        PATH_store + title + log_ + ".png",
        bbox_inches="tight",
    )
