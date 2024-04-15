import numpy as np
import matplotlib

matplotlib.rc("font", size=35)
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from utils.inference.inference_metrics import obtain_MPV_and_68
import concurrent.futures
import time
from utils.inference.inference_metrics import calculate_eff

# TODO paralellize this script or make the data larger so that the binning needed is larger
from scipy.optimize import curve_fit
from utils.inference.inference_metrics import get_sigma_gaussian


def get_mask_id(id, pids_pandora):
    mask_id = np.full((len(pids_pandora)), False, dtype=bool)
    for i in id:
        mask_i = pids_pandora == i
        mask_id = mask_id + mask_i
    mask_id = mask_id.astype(bool)
    return mask_id


def get_response_for_id_i(id, matched_pandora, matched_, tracks=False):
    pids_pandora = np.abs(matched_pandora["pid"].values)
    mask_id = get_mask_id(id, pids_pandora)
    df_id_pandora = matched_pandora[mask_id]

    pids = np.abs(matched_["pid"].values)
    mask_id = get_mask_id(id, pids)
    df_id = matched_[mask_id]

    (
        mean_p,
        variance_om_p,
        mean_true_rec_p,
        variance_om_true_rec_p,
        energy_resolutions_p,
        energy_resolutions_reco_p,
        mean_baseline,
        variance_om_baseline,
    ) = calculate_response(df_id_pandora, True, False, tracks=tracks)
    (
        mean,
        variance_om,
        mean_true_rec,
        variance_om_true_rec,
        energy_resolutions,
        energy_resolutions_reco,
        mean_baseline,
        variance_om_baseline,
    ) = calculate_response(df_id, False, False, tracks=tracks)
    print(variance_om_p)
    print(variance_om)
    print("recoooo")
    print(variance_om_true_rec_p)
    print(variance_om_true_rec)
    dic = {}
    dic["mean_p"] = mean_p
    dic["variance_om_p"] = variance_om_p
    dic["variance_om"] = variance_om
    dic["mean"] = mean
    dic["energy_resolutions"] = energy_resolutions
    dic["energy_resolutions_p"] = energy_resolutions_p
    dic["mean_p_reco"] = mean_true_rec_p
    dic["variance_om_p_reco"] = variance_om_true_rec_p
    dic["energy_resolutions_p_reco"] = energy_resolutions_reco_p
    dic["mean_reco"] = mean_true_rec
    dic["variance_om_reco"] = variance_om_true_rec
    dic["energy_resolutions_reco"] = energy_resolutions_reco
    dic["mean_baseline"] = mean_baseline
    dic["variance_om_baseline"] = variance_om_baseline
    return dic


def plot_X(
    title,
    photons_dic,
    electrons_dic,
    y_axis,
    PATH_store,
    label1,
    label2,
    reco,
    plot_label1=False,
    plot_label2=False,
):
    colors_list = ["#fde0dd", "#c994c7", "#dd1c77"]  # color list poster neurips
    colors_list = ["#FF0000", "#FF0000", "#0000FF"]
    fig = plt.figure()
    j = 0
    plt.xlabel("Energy [GeV]", fontsize=40)
    # ax[row_i, j].set_xscale("log")
    plt.title(title, fontsize=40)
    plt.grid()
    if plot_label1:
        plt.scatter(
            photons_dic["energy_resolutions" + reco],
            photons_dic[y_axis + reco],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML " + label1,
            marker="x",
            s=50,
        )
        plt.scatter(
            photons_dic["energy_resolutions_p" + reco],
            photons_dic[y_axis + "_p" + reco],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora " + label1,
            marker="x",
            s=50,
        )
    if plot_label2:
        plt.scatter(
            electrons_dic["energy_resolutions" + reco],
            electrons_dic[y_axis + reco],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            marker="o",
            label="ML " + label2,
            s=50,
        )
        plt.scatter(
            electrons_dic["energy_resolutions_p" + reco],
            electrons_dic[y_axis + "_p" + reco],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            marker="o",
            label="Pandora " + label2,
            s=50,
        )
    if title == "Electromagnetic Resolution" or title == "Hadronic Resolution":
        if reco == "":
            if plot_label2:
                plt.scatter(
                    electrons_dic["energy_resolutions"],
                    electrons_dic["variance_om_baseline"],
                    facecolors="black",
                    edgecolors="black",
                    marker=".",
                    label="Baseline " + label2,
                    s=50,
                )
            if plot_label1:
                plt.scatter(
                    photons_dic["energy_resolutions"],
                    photons_dic["variance_om_baseline"],
                    facecolors="black",
                    edgecolors="black",
                    marker=".",
                    label="Baseline " + label1,
                    s=50,
                )
            dic0_fit = get_fit(
                photons_dic["energy_resolutions"], photons_dic["variance_om_baseline"]
            )
            dic01_fit = get_fit(
                electrons_dic["energy_resolutions"],
                electrons_dic["variance_om_baseline"],
            )
        dic1_fit = get_fit(
            photons_dic["energy_resolutions" + reco], photons_dic[y_axis + reco]
        )
        dic1_fit_pandora = get_fit(
            photons_dic["energy_resolutions_p" + reco],
            photons_dic[y_axis + "_p" + reco],
        )
        dic2_fit = get_fit(
            electrons_dic["energy_resolutions" + reco], electrons_dic[y_axis + reco]
        )
        dic2_fit_pandora = get_fit(
            electrons_dic["energy_resolutions_p" + reco],
            electrons_dic[y_axis + "_p" + reco],
        )
        if reco == "":
            fits_l1 = [
                dic0_fit,
                dic1_fit,
                dic1_fit_pandora,
            ]
            fits_l2 = [
                dic01_fit,
                dic2_fit,
                dic2_fit_pandora,
            ]
            color_list_fits_l1 = [
                "black",
                colors_list[1],
                colors_list[2],
            ]
            color_list_fits_l2 = [
                "black",
                colors_list[1],
                colors_list[2],
            ]
            line_type_fits_l1 = ["-", "-", "-."]
            line_type_fits_l2 = ["-", "-", "-."]
        else:
            fits = [dic1_fit, dic1_fit_pandora, dic2_fit, dic2_fit_pandora]
            color_list_fits = [
                colors_list[1],
                colors_list[2],
                colors_list[1],
                colors_list[2],
            ]
            line_type_fits = ["-", "-", "-.", "-."]
        if plot_label1:
            plot_fit(fits_l1, line_type_fits_l1, color_list_fits_l1)
        if plot_label2:
            plot_fit(fits_l2, line_type_fits_l2, color_list_fits_l2)
        if reco == "_reco":
            plt.yscale("log")
        else:
            if title == "Electromagnetic Resolution":
                ymax = 0.3
            else:
                ymax = 0.5
            plt.ylim([0, ymax])
        plt.xlim([0, 55])
        ylabel = r"$\frac{\sigma_{E_{reco}}}{\langle E_{reco} \rangle}$"
        plt.ylabel(ylabel, fontsize=40)

    else:
        ylabel = r"$\langle E_{reco} \rangle / E_{true}$"
        plt.ylabel(ylabel, fontsize=40)
    # loc="upper right",
    plt.tick_params(axis="both", which="major", labelsize=40)
    if title == "Electromagnetic Response" or title == "Hadronic Response":
        plt.ylim([0.6, 1.4])
    plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc="upper left")
    if plot_label1:
        label = label1
    if plot_label2:
        label = label2
    fig.savefig(PATH_store + title + reco + label + "_v0.png", bbox_inches="tight")


def plot_fit(fits, line_type_fits, color_list_fits):
    fitlabel1 = r"$\frac{\sigma_E}{\langle E \rangle} = \sqrt{\frac{a^2}{E} + \frac{b^2}{E^2} + c^2}$"
    # fitlabel1 = r"$\frac{\sigma_E}{\langle E \rangle} = \sqrt{\frac{a^2}{E} + c^2}$"
    fitlabel2 = ""
    for id_fix, fit in enumerate(fits):
        if id_fix == 0:
            fitlabel = fitlabel1
        else:
            fitlabel = fitlabel2
        fit_a = f"{np.abs(fit[1][0]):.2f}"
        fit_b = f"{np.abs(fit[1][1]):.2f}"
        fit_c = f"{np.abs(fit[1][2]):.2f}"
        plt.plot(
            fit[0],
            resolution(fit[0], *fit[1]),
            line_type_fits[id_fix],
            c=color_list_fits[id_fix],
            label=fitlabel
            + "\nFit: a = "
            + fit_a
            + "; b = "
            + fit_b
            + "; c = "
            + fit_c,
        )


def get_fit(energies, errors):
    energies = energies
    errors = errors
    popt, pcov = curve_fit(resolution, energies, errors)
    xdata = np.arange(0, 51, 0.1)
    return [xdata, popt]


def resolution(E, a, b, c):
    return (a**2 / E + c**2 + b**2 / E**2) ** 0.5
    # return (a**2 / E + c**2) ** 0.5


def plot_per_energy_resolution2(
    sd_pandora, sd_hgb, matched_pandora, matched_, PATH_store, tracks=False
):
    mask = matched_["calibration_factor"] > 0
    matched_ = matched_[mask]
    if tracks:
        tracks_label = "tracks"
    else:
        tracks_label = ""
    plot_response = True
    if plot_response:
        list_plots = ["", "_reco"]  # "","_reco"
        photons_dic = get_response_for_id_i(
            [22], matched_pandora, matched_, tracks=tracks
        )
        electrons_dic = get_response_for_id_i(
            [11], matched_pandora, matched_, tracks=tracks
        )
        hadrons_dic = get_response_for_id_i(
            [130], matched_pandora, matched_, tracks=tracks
        )
        hadrons_dic2 = get_response_for_id_i(
            [211], matched_pandora, matched_, tracks=tracks
        )
        for el in list_plots:

            plot_one_label(
                "Electromagnetic Resolution",
                photons_dic,
                "variance_om",
                PATH_store,
                "Photons",
                el,
                tracks=tracks_label,
            )
            plot_one_label(
                "Electromagnetic Response",
                photons_dic,
                "mean",
                PATH_store,
                "Photons",
                el,
                tracks=tracks_label,
            )
            plot_one_label(
                "Electromagnetic Response",
                electrons_dic,
                "mean",
                PATH_store,
                "Electrons",
                el,
                tracks=tracks_label,
            )
            plot_one_label(
                "Electromagnetic Resolution",
                electrons_dic,
                "variance_om",
                PATH_store,
                "Electrons",
                el,
                tracks=tracks_label,
            )

            plot_one_label(
                "Hadronic Resolution",
                hadrons_dic,
                "variance_om",
                PATH_store,
                "KL",
                el,
                tracks=tracks_label,
            )
            plot_one_label(
                "Hadronic Response",
                hadrons_dic,
                "mean",
                PATH_store,
                "KL",
                el,
                tracks=tracks_label,
            )

            plot_one_label(
                "Hadronic Resolution",
                hadrons_dic2,
                "variance_om",
                PATH_store,
                "Pions",
                el,
                tracks=tracks_label,
            )
            plot_one_label(
                "Hadronic Response",
                hadrons_dic2,
                "mean",
                PATH_store,
                "Pions",
                el,
                tracks=tracks_label,
            )


def plot_per_energy_resolution(
    sd_pandora, sd_hgb, matched_pandora, matched_, PATH_store, tracks=False
):
    plot_response = True
    if plot_response:
        list_plots = ["_reco"]  # "","_reco"
        for el in list_plots:
            colors_list = ["#fde0dd", "#c994c7", "#dd1c77"]  # color list poster neurips
            marker_size = 15
            log_scale = True
            photons_dic = get_response_for_id_i(22, matched_pandora, matched_)
            electrons_dic = get_response_for_id_i(11, matched_pandora, matched_)
            plot_X(
                "Electromagnetic Response",
                photons_dic,
                electrons_dic,
                "mean",
                PATH_store,
                "Photons",
                "Electrons",
                el,
                plot_label1=True,
            )
            plot_X(
                "Electromagnetic Resolution",
                photons_dic,
                electrons_dic,
                "variance_om",
                PATH_store,
                "Photons",
                "Electrons",
                el,
                plot_label1=True,
            )
            plot_X(
                "Electromagnetic Resolution",
                photons_dic,
                electrons_dic,
                "variance_om",
                PATH_store,
                "Photons",
                "Electrons",
                el,
                plot_label2=True,
            )
            pions_dic = get_response_for_id_i(211.0, matched_pandora, matched_)
            kaons_dic = get_response_for_id_i(130.0, matched_pandora, matched_)
            plot_X(
                "Hadronic Response",
                pions_dic,
                kaons_dic,
                "mean",
                PATH_store,
                "Pions",
                "Kaons",
                el,
                plot_label1=True,
            )
            plot_X(
                "Hadronic Resolution",
                pions_dic,
                kaons_dic,
                "variance_om",
                PATH_store,
                "Pions",
                "Kaons",
                el,
                plot_label1=True,
            )
            plot_X(
                "Hadronic Resolution",
                pions_dic,
                kaons_dic,
                "variance_om",
                PATH_store,
                "Pions",
                "Kaons",
                el,
                plot_label2=True,
            )


def plot_efficiency_all(sd_pandora, df_list, PATH_store, labels):

    photons_dic = create_eff_dic_pandora(sd_pandora, 22)
    electrons_dic = create_eff_dic_pandora(sd_pandora, 11)
    pions_dic = create_eff_dic_pandora(sd_pandora, 211)
    kaons_dic = create_eff_dic_pandora(sd_pandora, 130)

    for var_i, sd_hgb in enumerate(df_list):
        photons_dic = create_eff_dic(photons_dic, sd_hgb, 22, var_i=var_i)
        electrons_dic = create_eff_dic(electrons_dic, sd_hgb, 11, var_i=var_i)
        pions_dic = create_eff_dic(pions_dic, sd_hgb, 211, var_i=var_i)
        kaons_dic = create_eff_dic(kaons_dic, sd_hgb, 130, var_i=var_i)
    plot_eff(
        "Electromagnetic",
        photons_dic,
        "Photons",
        PATH_store,
        labels,
    )
    plot_eff(
        "Electromagnetic",
        electrons_dic,
        "Electrons",
        PATH_store,
        labels,
    )
    plot_eff(
        "Hadronic",
        pions_dic,
        "Pions",
        PATH_store,
        labels,
    )
    plot_eff(
        "Hadronic",
        kaons_dic,
        "Kaons",
        PATH_store,
        labels,
    )


def create_eff_dic_pandora(matched_pandora, id):
    pids_pandora = np.abs(matched_pandora["pid"].values)
    mask_id = pids_pandora == id
    df_id_pandora = matched_pandora[mask_id]
    eff_p, energy_eff_p = calculate_eff(df_id_pandora, False)
    photons_dic = {}
    photons_dic["eff_p"] = eff_p
    photons_dic["energy_eff_p"] = energy_eff_p
    return photons_dic


def create_eff_dic(photons_dic, matched_, id, var_i):
    pids = np.abs(matched_["pid"].values)
    mask_id = pids == id
    df_id = matched_[mask_id]

    eff, energy_eff = calculate_eff(df_id, False)
    photons_dic["eff_" + str(var_i)] = eff
    photons_dic["energy_eff_" + str(var_i)] = energy_eff
    return photons_dic


def plot_eff(title, photons_dic, label1, PATH_store, labels):
    colors_list = ["#FF0000", "#FF0000", "#0000FF"]
    markers = ["^", "*", "x", "d", ".", "s"]
    fig = plt.figure()
    j = 0
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Efficiency [GeV]")
    # ax[row_i, j].set_xscale("log")
    plt.title(title)
    plt.grid()
    for i in range(0, len(labels)):
        plt.scatter(
            photons_dic["energy_eff_" + str(i)],
            photons_dic["eff_" + str(i)],
            label="ML " + labels[i],
            marker=markers[i],
            s=50,
        )
    plt.scatter(
        photons_dic["energy_eff_p"],
        photons_dic["eff_p"],
        facecolors=colors_list[2],
        edgecolors=colors_list[2],
        label="Pandora " + label1,
        marker="x",
        s=50,
    )

    plt.legend(loc="lower right")
    if title == "Electromagnetic":
        plt.ylim([0.5, 1.1])
    else:
        plt.ylim([0.5, 1.1])
    fig.savefig(
        PATH_store + title + label1 + ".png",
        bbox_inches="tight",
    )


def calculate_response(matched, pandora, log_scale=False, tracks=False):
    if log_scale:
        bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    else:
        bins = np.arange(0, 51, 6)
    binning = 1e-4
    if pandora:
        bins_per_binned_E = np.arange(0.5, 1.5, binning)
    else:
        bins_per_binned_E = np.arange(0.5, 1.5, binning)
    mean = []
    variance_om = []
    mean_baseline = []
    variance_om_baseline = []
    mean_true_rec = []
    variance_om_true_rec = []
    energy_resolutions = []
    energy_resolutions_reco = []

    # tic = time.time()
    # vector = range(len(bins) - 1)
    # output_results = parallel_process(vector, bins, matched, pandora, bins_per_binned_E)
    # mean_true_rec = [r[0] for ind, r in enumerate(output_results)]
    # variance_om_true_rec = [r[1] for ind, r in enumerate(output_results)]
    # energy_resolutions_reco = [r[2] for ind, r in enumerate(output_results)]
    # toc = time.time()
    # print("time with paralel version", toc - tic)
    print("START PANDORA")
    binning = 1e-3
    if pandora:
        bins_per_binned_E = np.arange(0, 2, binning)
    else:
        bins_per_binned_E = np.arange(0, 2, binning)
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
            if tracks:
                pred_e = matched.pandora_calibrated_pfo[mask]
            else:
                pred_e = matched.pandora_calibrated_E[mask]
        else:
            pred_e = matched.calibrated_E[mask]
        pred_e_nocor = matched.pred_showers_E[mask]
        if np.sum(mask) > 0:  # if the bin is not empty
            e_over_true = pred_e / true_e
            e_over_reco = true_rec / true_e
            e_over_reco_ML = pred_e_nocor / true_rec
            # mean_predtotrue, var_predtotrue = obtain_MPV_and_68(
            #     e_over_true, bins_per_binned_E
            # )

            mean_predtotrue, var_predtotrue = get_sigma_gaussian(
                e_over_true, bins_per_binned_E
            )

            mean_reco_true, var_reco_true = get_sigma_gaussian(
                e_over_reco, bins_per_binned_E
            )

            mean_reco_ML, var_reco_ML = get_sigma_gaussian(
                e_over_reco_ML, bins_per_binned_E
            )
            # mean_reco_true, var_reco_true = obtain_MPV_and_68(
            #     e_over_reco, bins_per_binned_E
            # )

            # mean_predtotrue = np.mean(e_over_true)
            # var_predtotrue = np.var(e_over_true) / mean_predtotrue
            mean_true_rec.append(mean_reco_ML)
            variance_om_true_rec.append(np.abs(var_reco_ML))
            mean_baseline.append(mean_reco_true)
            variance_om_baseline.append(np.abs(var_reco_true))
            mean.append(mean_predtotrue)
            variance_om.append(np.abs(var_predtotrue))
            energy_resolutions.append((bin_i1 + bin_i) / 2)
            energy_resolutions_reco.append((bin_i1 + bin_i) / 2)

    return (
        mean,
        variance_om,
        mean_true_rec,
        variance_om_true_rec,
        energy_resolutions,
        energy_resolutions_reco,
        mean_baseline,
        variance_om_baseline,
    )


def process_element(i, bins, matched, pandora, bins_per_binned_E):
    # Your processing logic here
    bin_i = bins[i]
    bin_i1 = bins[i + 1]
    print(i)
    mask_above = matched["reco_showers_E"] <= bin_i1
    mask_below = matched["reco_showers_E"] > bin_i
    mask_check = matched["pred_showers_E"] > 0
    mask = mask_below * mask_above * mask_check

    pred_e = matched.pred_showers_E[mask]
    true_rec = matched.reco_showers_E[mask]
    if np.sum(mask) > 0:  # if the bin is not empty
        e_over_rec = pred_e / true_rec

        mean_predtored, variance_om_true_rec_ = obtain_MPV_and_68(
            e_over_rec, bins_per_binned_E
        )

    return mean_predtored, variance_om_true_rec_, (bin_i1 + bin_i) / 2


def parallel_process(
    vector, fixed_arg1, fixed_arg2, fixed_arg3, fixed_arg4, num_workers=16
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use executor.map to parallelize the processing of vector elements
        results1 = list(
            executor.map(
                process_element,
                vector,
                [fixed_arg1] * len(vector),
                [fixed_arg2] * len(vector),
                [fixed_arg3] * len(vector),
                [fixed_arg4] * len(vector),
            )
        )

    return results1


def plot_one_label(title, photons_dic, y_axis, PATH_store, label1, reco, tracks=""):
    if reco == "":
        label_add = " raw"
        label_add_pandora = " corrected"
    else:
        label_add = " raw"
        label_add_pandora = " raw"

    colors_list = ["#FF0000", "#FF0000", "#0000FF"]
    fig = plt.figure()
    j = 0
    plt.xlabel("Energy [GeV]", fontsize=40)
    # ax[row_i, j].set_xscale("log")
    plt.title(title, fontsize=40)
    plt.grid()
    plt.scatter(
        photons_dic["energy_resolutions" + reco],
        photons_dic[y_axis + reco],
        facecolors=colors_list[1],
        edgecolors=colors_list[1],
        label="ML " + label1 + label_add,
        marker="x",
        s=50,
    )
    plt.scatter(
        photons_dic["energy_resolutions_p" + reco],
        photons_dic[y_axis + "_p" + reco],
        facecolors=colors_list[2],
        edgecolors=colors_list[2],
        label="Pandora " + label1 + label_add_pandora,
        marker="x",
        s=50,
    )

    if title == "Electromagnetic Resolution" or title == "Hadronic Resolution":
        if reco == "":
            plt.scatter(
                photons_dic["energy_resolutions"],
                photons_dic["variance_om_baseline"],
                facecolors="black",
                edgecolors="black",
                marker=".",
                label="Baseline " + label1 + " raw",
                s=50,
            )
            dic0_fit = get_fit(
                photons_dic["energy_resolutions"], photons_dic["variance_om_baseline"]
            )

        dic1_fit = get_fit(
            photons_dic["energy_resolutions" + reco], photons_dic[y_axis + reco]
        )
        dic1_fit_pandora = get_fit(
            photons_dic["energy_resolutions_p" + reco],
            photons_dic[y_axis + "_p" + reco],
        )
        if reco == "":
            fits_l1 = [
                dic0_fit,
                dic1_fit,
                dic1_fit_pandora,
            ]

            color_list_fits_l1 = [
                "black",
                colors_list[1],
                colors_list[2],
            ]

            line_type_fits_l1 = ["-", "-", "-."]
            plot_fit(fits_l1, line_type_fits_l1, color_list_fits_l1)
        else:
            fits = [dic1_fit, dic1_fit_pandora]
            color_list_fits = [
                colors_list[1],
                colors_list[2],
            ]
            line_type_fits = ["-", "-."]

            plot_fit(fits, line_type_fits, color_list_fits)

        if reco == "_reco":
            plt.yscale("log")
        else:
            if title == "Electromagnetic Resolution":
                ymax = 0.3
            else:
                ymax = 0.6
            plt.ylim([0, ymax])
        plt.xlim([0, 55])
        ylabel = r"$\frac{\sigma_{E_{reco}}}{\langle E_{reco} \rangle}$"
        plt.ylabel(ylabel, fontsize=40)

    else:
        ylabel = r"$\langle E_{reco} \rangle / E_{true}$"
        plt.ylabel(ylabel, fontsize=40)
    # loc="upper right",
    plt.tick_params(axis="both", which="major", labelsize=40)
    if title == "Electromagnetic Response" or title == "Hadronic Response":
        plt.ylim([0.6, 1.4])
    plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc="upper left")
    label = label1

    fig.savefig(
        PATH_store + title + reco + label + tracks + "_v1.pdf", bbox_inches="tight"
    )
