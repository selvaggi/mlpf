import numpy as np
import matplotlib

matplotlib.rc("font", size=35)
import pandas as pd
import matplotlib.pyplot as plt

from utils.inference.inference_metrics import obtain_MPV_and_68

#TODO paralellize this script or make the data larger so that the binning needed is larger 
def plot_per_energy_resolution(matched_pandora, matched_, PATH_store):
    colors_list = ["#fde0dd", "#c994c7", "#dd1c77"]  # color list poster neurips
    unique_particle_IDS = np.unique(matched_pandora["pid"].values)
    pids_pandora = matched_pandora["pid"].values
    pids = matched_["pid"].values
    marker_size = 15
    log_scale = True
    fig, ax = plt.subplots(
        len(unique_particle_IDS), 4, figsize=(len(unique_particle_IDS) * 4, 20 * 4)
    )
    for row_i, id in enumerate(unique_particle_IDS):
        mask_id = pids_pandora == id
        df_id_pandora = matched_pandora[mask_id]
        mask_id = pids == id
        df_id = matched_[mask_id]
        (
            mean_p,
            variance_om_p,
            mean_true_rec_p,
            variance_om_true_rec_p,
            energy_resolutions_p,
            energy_resolutions_reco_p,
        ) = calculate_response(df_id_pandora, True, False)
        (
            mean,
            variance_om,
            mean_true_rec,
            variance_om_true_rec,
            energy_resolutions,
            energy_resolutions_reco,
        ) = calculate_response(df_id, False, False)

        ax[row_i, 0].scatter(
            energy_resolutions,
            mean,
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML",
            s=50,
        )
        ax[row_i, 0].scatter(
            energy_resolutions_p,
            mean_p,
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            s=50,
        )
        j = 0
        ax[row_i, j].set_xlabel("Log Energy [GeV]")
        # ax[row_i, j].set_xscale("log")
        ax[row_i, j].set_ylabel("Response")
        ax[row_i, j].set_title(str(id))
        ax[row_i, j].grid()
        ax[row_i, j].legend(loc="lower right")
        ax[row_i, 1].scatter(
            energy_resolutions,
            variance_om,
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML",
            s=50,
        )
        ax[row_i, 1].scatter(
            energy_resolutions_p,
            variance_om_p,
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            s=50,
        )
        j = 1
        ax[row_i, j].set_xlabel("Log Energy [GeV]")
        ax[row_i, j].set_xscale("log")
        ax[row_i, j].set_ylabel("Resoltution")
        ax[row_i, j].grid()
        ax[row_i, j].set_title(str(id))
        ax[row_i, j].legend(loc="lower right")
        ax[row_i, 2].scatter(
            energy_resolutions_reco,
            mean_true_rec,
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML",
            s=50,
        )
        ax[row_i, 2].scatter(
            energy_resolutions_reco_p,
            mean_true_rec_p,
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            s=50,
        )
        j = 2
        ax[row_i, j].set_xlabel("Log Reco Energy [GeV]")
        # ax[row_i, j].set_xscale("log")
        ax[row_i, j].set_ylabel("Response")
        ax[row_i, j].grid()
        ax[row_i, j].set_title(str(id))
        ax[row_i, j].legend(loc="lower right")
        ax[row_i, 3].scatter(
            energy_resolutions_reco,
            variance_om_true_rec,
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML",
            s=50,
        )
        ax[row_i, 3].scatter(
            energy_resolutions_reco_p,
            variance_om_true_rec_p,
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            s=50,
        )
        j = 3
        ax[row_i, j].set_xlabel("Log Reco Energy [GeV]")
        ax[row_i, j].set_xscale("log")
        ax[row_i, j].set_title(str(id))
        ax[row_i, j].set_ylabel("Resolution")
        ax[row_i, j].grid()
        ax[row_i, j].legend(loc="lower right")

    fig.savefig(
        PATH_store + "per_particle_metrics.png",
        bbox_inches="tight",
    )


def calculate_response(matched, pandora, log_scale=False):
    if log_scale:
        bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    else:
        bins = np.arange(0, 51, 2)
    binning = 1e-4
    if pandora:
        bins_per_binned_E = np.arange(0, 3, binning)
    else:
        bins_per_binned_E = np.arange(0, 3, binning)
    mean = []
    variance_om = []
    mean_true_rec = []
    variance_om_true_rec = []
    energy_resolutions = []
    energy_resolutions_reco = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = (
            matched["reco_showers_E"] <= bin_i1
        )  # true_showers_E, reco_showers_E
        mask_below = matched["reco_showers_E"] > bin_i
        mask_check = matched["pred_showers_E"] > 0
        mask = mask_below * mask_above * mask_check

        pred_e = matched.pred_showers_E[mask]
        true_rec = matched.reco_showers_E[mask]
        true_e = matched.true_showers_E[mask]
        if pandora:
            pred_e_corrected = matched.pandora_calibrated_E[mask]
        else:
            pred_e_corrected = matched.calibrated_E[mask]
        if np.sum(mask) > 0:  # if the bin is not empty
            e_over_rec = pred_e / true_rec

            mean_predtored, variance_om_true_rec_ = obtain_MPV_and_68(
                e_over_rec, bins_per_binned_E
            )
            print("variance", len(e_over_rec), bin_i, i, pandora, variance_om_true_rec_)
            # mean_predtored = np.mean(e_over_rec)
            # variance_om_true_rec_ = np.var(e_over_rec) / mean_predtored
            mean_true_rec.append(mean_predtored)
            variance_om_true_rec.append(variance_om_true_rec_)
            energy_resolutions_reco.append((bin_i1 + bin_i) / 2)

    if pandora:
        bins_per_binned_E = np.arange(0, 3, binning)
    else:
        bins_per_binned_E = np.arange(0, 3, binning)
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = matched["true_showers_E"] <= bin_i1
        mask_below = matched["true_showers_E"] > bin_i
        mask_check = matched["pred_showers_E"] > 0
        mask = mask_below * mask_above * mask_check
        true_e = matched.true_showers_E[mask]
        true_rec = matched.reco_showers_E[mask]
        if pandora:
            pred_e = matched.pandora_calibrated_E[mask]
        else:
            pred_e = matched.calibrated_E[mask]
        if np.sum(mask) > 0:  # if the bin is not empty
            e_over_true = pred_e / true_e
            e_rec_over_true = true_rec / true_e
            mean_predtotrue, var_predtotrue = obtain_MPV_and_68(
                e_over_true, bins_per_binned_E
            )

            # mean_predtotrue = np.mean(e_over_true)
            # var_predtotrue = np.var(e_over_true) / mean_predtotrue

            mean.append(mean_predtotrue)
            variance_om.append(var_predtotrue)
            energy_resolutions.append((bin_i1 + bin_i) / 2)

    return (
        mean,
        variance_om,
        mean_true_rec,
        variance_om_true_rec,
        energy_resolutions,
        energy_resolutions_reco,
    )
