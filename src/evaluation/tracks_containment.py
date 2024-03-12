import numpy as np
import matplotlib

matplotlib.rc("font", size=35)
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

colors_list = ["#c91023", "#9ecae1", "#3182bd"]


def plot_metrics(
    dict_1,
    PATH_store,
):
    marker_size = 2
    idx_start = 0
    fig, ax = plt.subplots(2, 1, figsize=(8, 16))
    # efficiency plot

    ax = plot_containment(ax, dict_1, 0, idx_start)
    ax = plot_purity(ax, dict_1, 1, idx_start)

    fig.savefig(
        PATH_store + "testeq_rec_comp_MVP_68_calibrated_compare_to_V8.png",
        bbox_inches="tight",
    )


def plot_containment(ax, dict_1, i, idx_start):
    marker_size = 2
    ax[i].errorbar(
        np.array(dict_1["energy_ms"])[idx_start:],
        np.array(dict_1["fce_energy"])[idx_start:],
        np.array(dict_1["fce_var_energy"])[idx_start:],
        marker="o",
        mec=colors_list[0],
        mfc=colors_list[0],
        ecolor=colors_list[0],
        ms=marker_size,
        mew=4,
        linestyle="",
    )

    ax[i].set_xlabel("Energy [GeV]")
    ax[i].set_ylabel("Containment")
    ax[i].grid()
    ax[i].set_ylim([0.8, 1.1])
    # ax[i].set_xticks(fontsize=25)
    # ax[i].set_yticks(fontsize=25)
    ax[i].set_yticks(ticks=[0.8, 0.9, 1, 1.1])
    # ax[3, 0].set_yscale("log")
    return ax


def plot_purity(ax, dict_1, i, idx_start):
    marker_size = 2

    ax[i].errorbar(
        np.array(dict_1["energy_ms"])[idx_start:],
        np.array(dict_1["purity_energy"])[idx_start:],
        np.array(dict_1["purity_var_energy"])[idx_start:],
        marker="o",
        mec=colors_list[0],
        mfc=colors_list[0],
        ecolor=colors_list[0],
        ms=marker_size,
        mew=4,
        linestyle="",
    )

    ax[i].set_xlabel("Energy [GeV]")
    ax[i].set_ylabel("Purity")
    ax[i].grid()
    ax[i].set_ylim([0.8, 1.1])
    # ax[i].set_xticks(fontsize=25)
    # ax[i].set_yticks(fontsize=25)
    ax[i].set_yticks([0.8, 0.9, 1, 1.1])
    return ax


def obtain_metrics(matched):

    (
        fce_energy,
        fce_var_energy,
        energy_ms,
        purity_energy,
        purity_var_energy,
    ) = calculate_purity_containment(matched)

    dict = {
        "fce_energy": fce_energy,
        "fce_var_energy": fce_var_energy,
        "energy_ms": energy_ms,
        "purity_energy": purity_energy,
        "purity_var_energy": purity_var_energy,
    }
    return dict


def calculate_purity_containment(
    matched,
):
    bins = np.arange(0, 51, 2)
    fce_energy = []
    fce_var_energy = []
    energy_ms = []

    purity_energy = []
    purity_var_energy = []
    fce = matched["e_pred_and_truth"] / matched["reco_showers_E"]
    purity = matched["e_pred_and_truth"] / matched["pred_showers_E"]
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = matched["true_showers_E"] <= bin_i1
        mask_below = matched["true_showers_E"] > bin_i
        mask_check = matched["pred_showers_E"] > 0
        mask = mask_below * mask_above * mask_check
        fce_e = np.mean(fce[mask])
        fce_var = np.var(fce[mask])
        purity_e = np.mean(purity[mask])
        purity_var = np.var(purity[mask])
        if np.sum(mask) > 0:
            fce_energy.append(fce_e)
            fce_var_energy.append(fce_var)
            energy_ms.append((bin_i1 + bin_i) / 2)
            purity_energy.append(purity_e)
            purity_var_energy.append(purity_var)
    return (
        fce_energy,
        fce_var_energy,
        energy_ms,
        purity_energy,
        purity_var_energy,
    )
