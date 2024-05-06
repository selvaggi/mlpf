import numpy as np
import matplotlib

matplotlib.rc("font", size=35)
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import concurrent.futures
import time
import mplhep as hep
import seaborn as sns

hep.style.use("CMS")
# TODO paralellize this script or make the data larger so that the binning needed is larger
from scipy.optimize import curve_fit


def plot_efficiency_all(list_dataframes, PATH_store, PATH_comparison, label_list, log):
    list_all_p = []
    for sd_hgb in list_dataframes:
        all_particles = create_eff_dic(sd_hgb, log)
        all_particles_0_10 = create_eff_dic(
            sd_hgb, log, num_hits_min=0, num_hits_max=10
        )
        all_particles_10_20 = create_eff_dic(
            sd_hgb, log, num_hits_min=10, num_hits_max=20
        )
        all_particles_20_30 = create_eff_dic(
            sd_hgb, log, num_hits_min=20, num_hits_max=30
        )
        all_particles_30_40 = create_eff_dic(
            sd_hgb, log, num_hits_min=30, num_hits_max=40
        )
        all_particles_40_50 = create_eff_dic(
            sd_hgb, log, num_hits_min=40, num_hits_max=50
        )
        all_particles_50_100 = create_eff_dic(
            sd_hgb, log, num_hits_min=50, num_hits_max=100
        )
        all_particles_100_all = create_eff_dic(
            sd_hgb, log, num_hits_min=100, num_hits_max=5000
        )
        list_ = [
            all_particles,
            all_particles_0_10,
            all_particles_10_20,
            all_particles_20_30,
            all_particles_30_40,
            all_particles_40_50,
            all_particles_50_100,
            all_particles_100_all,
        ]
        list_all_p.append(list_)

    plot_eff("efficiency", list_all_p, label_list, PATH_comparison, log)


def create_eff_dic(matched_, log, num_hits_min=0, num_hits_max=5000):
    mask1 = matched_["reco_showers_E"] > num_hits_min
    mask2 = matched_["reco_showers_E"] < num_hits_max
    matched_ = matched_[mask1 * mask2]
    df_id = matched_
    eff, energy_eff, total_showers_ = calculate_eff(df_id, log)
    photons_dic = {}
    photons_dic["eff"] = eff
    photons_dic["energy_eff"] = energy_eff
    photons_dic["total_showers_"] = total_showers_
    return photons_dic


def calculate_eff(sd, log_scale=False):
    if log_scale:
        bins = np.exp(np.arange(np.log(0.001), np.log(80), 0.5))
    else:
        bins = np.arange(0, 51, 2)
    eff = []
    energy_eff = []
    total_showers_ = []
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
            total_showers_.append(total_showers)
        print(
            "bin",
            bin_i1,
            bin_i,
            (total_showers - number_of_non_reconstructed_showers) / total_showers,
            total_showers,
        )
    return eff, energy_eff, total_showers_


def plot_eff(title, dataframe_list, label_list, PATH_store, log):
    list_nh = ["all", "0-10", "10-20", "20-30", "30-40", "40-50", "50-100", "100>"]
    markers = ["^", "*", "x", "."]
    fig, axs = plt.subplots(1, len(list_nh), figsize=(80, 15))
    for index, plot_title in enumerate(list_nh):
        colors_list = ["#FF0000", "#FF0000", "#0000FF"]

        j = 0
        axs[index].set_xlabel("p_T [GeV]", fontsize=35)
        axs[index].set_ylabel("Efficiency", fontsize=35)
        axs[index].grid()
        axs[index].title.set_text(plot_title)
        for i in range(0, len(dataframe_list)):
            axs[index].scatter(
                dataframe_list[i][index]["energy_eff"],
                dataframe_list[i][index]["eff"],
                label="Model " + label_list[i],
                marker=markers[i],
                s=100,
            )
        if log:
            log_ = "log"
            axs[index].set_xscale("log")
        else:
            log_ = ""
        axs[index].set_ylim([0.5, 1.1])
        axs[index].legend(loc="lower left")
        axs[index].axvline(x=0.6)
        axs[index].axvline(x=0.1)
        axs[index].set_yticks(ticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1])
        axs[index].set_xticks(ticks=[0.001, 0.01, 0.1, 1, 10, 100])
    fig.savefig(
        PATH_store + "comparison" + log_ + ".png",
        bbox_inches="tight",
    )

    # fig = plt.figure()
    # plt.xlabel("p_T [GeV]", fontsize=35)
    # plt.ylabel("Number of total showers", fontsize=35)
    # plt.grid()

    # df_l = []
    # for i in range(1, len(photons_dic)):
    #     p = len(photons_dic[i]["energy_eff"])
    #     df = {
    #         "energy_eff": np.arange(0, p),
    #         "total_s": np.array(photons_dic[i]["total_showers_"]),
    #         "n": np.zeros((len(photons_dic[i]["total_showers_"]))) + i,
    #     }
    #     df = pd.DataFrame(data=df)
    #     df_l.append(df)
    # df_l = pd.concat(df_l)

    # sns.barplot(df_l, x="energy_eff", y="total_s", hue="n")
    # # if log:
    # #     log_ = "log"
    # #     plt.xscale("log")
    # # else:
    # #     log_ = ""
    # plt.xticks(fontsize=25)
    # plt.yticks(fontsize=25)
    # plt.legend()
    # fig.savefig(
    #     PATH_store + title + log_ + "bar_plot_totalshowers" + ".png",
    #     bbox_inches="tight",
    # )
