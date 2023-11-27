import matplotlib

matplotlib.rc("font", size=20)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import seaborn as sns

hep.style.use("CMS")
colors_list = ["#fff7bc", "#fec44f", "#d95f0e"]


def plot_metrics(neutrals_only, dic1, dic2, dict_1, dict_2, dict_3, colors_list):
    fig, ax = plt.subplots(4, 2, figsize=(18, 8 * 4))
    # efficiency plot
    if dic1:
        ax[0, 0].scatter(
            dict_1["energy_eff"],
            dict_1["eff"],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            label="Hgcal",
        )
    if dic2:
        ax[0, 0].scatter(
            dict_2["energy_eff"],
            dict_2["eff"],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML_Pytorch",
        )
        ax[0, 0].scatter(
            dict_3["energy_eff"],
            dict_3["eff"],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
        )
    ax[0, 0].set_xlabel("True Energy [GeV]")
    ax[0, 0].set_ylabel("Efficiency")
    ax[0, 0].grid()
    ax[0, 0].legend(loc="lower right")

    # fake rates
    if dic1:
        ax[0, 1].scatter(
            dict_1["energy_fakes"],
            dict_1["fake_rate"],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            label="Hgcal",
        )
    if dic2:
        ax[0, 1].scatter(
            dict_2["energy_fakes"],
            dict_2["fake_rate"],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML_Pytorch",
        )
        ax[0, 1].scatter(
            dict_3["energy_fakes"],
            dict_3["fake_rate"],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
        )
    ax[0, 1].set_xlabel("Reconstructed Energy [GeV]")
    ax[0, 1].set_ylabel("Fake rate")
    ax[0, 1].grid()
    ax[0, 1].set_yscale("log")
    ax[0, 1].legend(loc="upper right")

    # resolution
    if dic1:
        ax[1, 0].scatter(
            dict_1["energy_resolutions_reco"],
            dict_1["mean_true_rec"],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            label="Hgcal",
        )
    if dic2:
        ax[1, 0].scatter(
            dict_2["energy_resolutions_reco"],
            dict_2["mean_true_rec"],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML_Pytorch",
        )
        ax[1, 0].scatter(
            dict_3["energy_resolutions_reco"],
            dict_3["mean_true_rec"],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
        )
    ax[1, 0].set_xlabel("Reco Energy [GeV]")
    ax[1, 0].set_ylabel("Response")
    ax[1, 0].grid()
    ax[1, 0].legend(loc="lower right")
    ax[1, 0].set_ylim([0, 1.5])

    # response
    if dic1:
        ax[1, 1].scatter(
            dict_1["energy_resolutions_reco"],
            dict_1["variance_om_true_rec"],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            label="Hgcal",
        )
    if dic2:
        ax[1, 1].scatter(
            dict_2["energy_resolutions_reco"],
            dict_2["variance_om_true_rec"],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML_Pytorch",
        )
        ax[1, 1].scatter(
            dict_3["energy_resolutions_reco"],
            dict_3["variance_om_true_rec"],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
        )
    ax[1, 1].set_xlabel("Reco Energy [GeV]")
    ax[1, 1].set_ylabel("Resolution")
    ax[1, 1].grid()
    ax[1, 1].set_yscale("log")
    ax[1, 1].legend(loc="upper right")
    ax[1, 1].set_ylim([0, 1])

    if dic1:
        ax[2, 0].scatter(
            dict_1["energy_resolutions"],
            dict_1["mean"],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            label="Hgcal",
        )
    if dic2:
        ax[2, 0].scatter(
            dict_2["energy_resolutions"],
            dict_2["mean"],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML_Pytorch",
        )
        ax[2, 0].scatter(
            dict_3["energy_resolutions"],
            dict_3["mean"],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
        )
    ax[2, 0].set_xlabel("True Energy [GeV]")
    ax[2, 0].set_ylabel("Response")
    ax[2, 0].grid()
    ax[2, 0].legend(loc="lower right")
    ax[2, 0].set_ylim([0, 1.5])

    # response
    if dic1:
        ax[2, 1].scatter(
            dict_1["energy_resolutions"],
            dict_1["variance_om"],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            label="Hgcal",
        )
    if dic2:
        ax[2, 1].scatter(
            dict_2["energy_resolutions"],
            dict_2["variance_om"],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML_Pytorch",
        )
        ax[2, 1].scatter(
            dict_3["energy_resolutions"],
            dict_3["variance_om"],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
        )
    ax[2, 1].set_xlabel("True Energy [GeV]")
    ax[2, 1].set_ylabel("Resolution")
    ax[2, 1].set_yscale("log")
    ax[2, 1].grid()
    ax[2, 1].legend(loc="upper right")
    ax[2, 1].set_ylim([0, 1])

    # purity
    if dic1:
        ax[3, 0].errorbar(
            np.array(dict_1["energy_ms"]),
            np.array(dict_1["fce_energy"]),
            np.array(dict_1["fce_var_energy"]),
            marker="o",
            mec=colors_list[0],
            ecolor=colors_list[0],
            ms=5,
            mew=4,
            linestyle="",
        )
    if dic2:
        ax[3, 0].errorbar(
            np.array(dict_2["energy_ms"]),
            np.array(dict_2["fce_energy"]),
            np.array(dict_2["fce_var_energy"]),
            marker="o",
            mec=colors_list[1],
            ecolor=colors_list[1],
            ms=5,
            mew=4,
            linestyle="",
        )
        ax[3, 0].errorbar(
            np.array(dict_3["energy_ms"]),
            np.array(dict_3["fce_energy"]),
            np.array(dict_3["fce_var_energy"]),
            marker="^",
            mec=colors_list[2],
            ecolor=colors_list[2],
            ms=5,
            mew=4,
            linestyle="",
        )

    ax[3, 0].set_xlabel("Reco Energy [GeV]")
    ax[3, 0].set_ylabel("Containment")
    ax[3, 0].grid()

    if dic1:
        ax[3, 1].errorbar(
            np.array(dict_1["energy_ms"]),
            np.array(dict_1["purity_energy"]),
            np.array(dict_1["purity_var_energy"]),
            marker=".",
            mec=colors_list[0],
            ecolor=colors_list[0],
            ms=5,
            mew=4,
            linestyle="",
        )
    if dic2:
        ax[3, 1].errorbar(
            np.array(dict_2["energy_ms"]),
            np.array(dict_2["purity_energy"]),
            np.array(dict_2["purity_var_energy"]),
            marker=".",
            mec=colors_list[1],
            ecolor=colors_list[1],
            ms=5,
            mew=4,
            linestyle="",
        )
        ax[3, 1].errorbar(
            np.array(dict_3["energy_ms"]),
            np.array(dict_3["purity_energy"]),
            np.array(dict_3["purity_var_energy"]),
            marker="^",
            mec=colors_list[2],
            ecolor=colors_list[2],
            ms=5,
            mew=4,
            linestyle="",
        )

    ax[3, 1].set_xlabel("Reco Energy [GeV]")
    ax[3, 1].set_ylabel("Purity")
    ax[3, 1].grid()
    ax[3, 1].set_ylim([0, 1.5])
    if neutrals_only:
        fig.savefig(
            "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/testeq_rec_comp_MVP_68_calibrated_neutrals.png",
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/testeq_rec_comp_MVP_68_calibrated.png",
            bbox_inches="tight",
        )


def plot_histograms_energy(dic1, dic2, dict_1, dict_2, dict_3):
    bins_plot_histogram = [0, 5, 10, 20]
    fig, ax = plt.subplots(4, 2, figsize=(18, 25))

    for i in range(0, 4):
        bin_name = bins_plot_histogram[i]
        sns.histplot(
            dict_2["dic_histograms"][str(bin_name) + "reco"],
            label="MLPF",
            stat="percent",
            color=colors_list[1],
            element="step",
            fill=False,
            ax=ax[i, 0],
        )
        sns.histplot(
            dict_3["dic_histograms"][str(bin_name) + "reco"],
            label="Pandora",
            stat="percent",
            color=colors_list[2],
            element="step",
            fill=False,
            ax=ax[i, 0],
        )
        sns.histplot(
            dict_2["dic_histograms"][str(bin_name) + "true"],
            label="MLPF",
            stat="percent",
            color=colors_list[1],
            element="step",
            fill=False,
            ax=ax[i, 1],
        )
        sns.histplot(
            dict_3["dic_histograms"][str(bin_name) + "true"],
            label="Pandora",
            stat="percent",
            color=colors_list[2],
            element="step",
            fill=False,
            ax=ax[i, 1],
        )
        ax[i, 1].set_xlabel("E pred / True Energy [GeV]")
        ax[i, 1].set_xlim([0, 2])
        ax[i, 1].grid()
        ax[i, 1].legend(loc="upper right")
        ax[i, 1].set_title(
            "[" + str(bin_name * 2) + ", " + str(bin_name * 2 + 2) + "]" + " GeV"
        )
        ax[i, 0].set_xlabel("E pred / Reco Energy [GeV]")
        ax[i, 0].grid()
        ax[i, 0].set_xlim([0, 2])
        ax[i, 0].legend(loc="upper right")
        ax[i, 0].set_title(
            "[" + str(bin_name * 2) + ", " + str(bin_name * 2 + 2) + "]" + " GeV"
        )
        # ax[i, 0].set_yscale("log")
    fig.tight_layout(pad=2.0)
    fig.savefig(
        "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/histograms_energy.png",
        bbox_inches="tight",
    )
