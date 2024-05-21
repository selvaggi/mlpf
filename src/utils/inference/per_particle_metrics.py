import numpy as np
import matplotlib
import os
matplotlib.rc("font", size=35)
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from src.utils.inference.inference_metrics import obtain_MPV_and_68
import concurrent.futures
import time
from src.utils.inference.inference_metrics import calculate_eff
import torch
import plotly
import plotly.graph_objs as go
import plotly.express as px


# TODO paralellize this script or make the data larger so that the binning needed is larger
from scipy.optimize import curve_fit
from src.utils.inference.inference_metrics import get_sigma_gaussian
from torch_scatter import scatter_sum


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
        e_over_e_distr_pandora,
        mean_errors_p,
        variance_errors_p
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
        e_over_e_distr_model,
        mean_errors,
        variance_errors
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
    dic["mean_errors"] = mean_errors
    dic["variance_errors"] = variance_errors
    dic["variance_errors_p"] = variance_errors_p
    dic["mean_errors_p"] = mean_errors_p
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
    dic["distributions_pandora"] = e_over_e_distr_pandora
    dic["distributions_model"] = e_over_e_distr_model
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
    plt.xlabel("Energy [GeV]", fontsize=30)
    # ax[row_i, j].set_xscale("log")
    plt.title(title, fontsize=30)
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
        plt.ylabel(ylabel, fontsize=30)

    else:
        ylabel = r"$\langle E_{reco} \rangle / E_{true}$"
        plt.ylabel(ylabel, fontsize=30)
    # loc="upper right",
    plt.tick_params(axis="both", which="major", labelsize=40)
    if title == "Electromagnetic Response" or title == "Hadronic Response":
        plt.ylim([0.6, 1.4])
    plt.legend(fontsize=30, bbox_to_anchor=(1.05, 1), loc="upper left")
    if plot_label1:
        label = label1
    if plot_label2:
        label = label2
    fig.savefig(PATH_store + title + reco + label + "_v0.png", bbox_inches="tight")


def plot_fit(fits, line_type_fits, color_list_fits, ax=None):
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
        if ax is None:
            a = plt
        else:
            a = ax
        a.plot(
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
    return [xdata, popt, np.sqrt(np.diag(pcov))]


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
        event_numbers = [0, 1, 2, 3]
        for event_number in event_numbers:
            filename = os.path.join(PATH_store, f"event_{event_number}_pandora.html")
            # plot_event(matched_, pandora=False, output_dir=filename)
            plot_event(matched_pandora[matched_pandora.number_batch == event_number], pandora=True, output_dir=filename)
        list_plots = [""] #  "", "_reco"
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
        neutrons = get_response_for_id_i(
            [2112], matched_pandora, matched_, tracks=tracks
        )
        protons = get_response_for_id_i(
            [2212], matched_pandora, matched_, tracks=tracks
        )
        event_res_dic = get_response_for_event_energy(matched_pandora, matched_)
        plot_one_label(
            "Event Energy Resolution",
            event_res_dic,
            "variance_om",
            PATH_store,
            "ML",
            "",
            tracks="",
            plot_baseline=True
        )
        plot_per_particle = True
        if plot_per_particle:
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
                '''plot_one_label(
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
                )'''
                '''plot_one_label(
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
                )'''
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
                '''# plot the neutrons and protons
                plot_one_label(
                    "Hadronic Resolution",
                    neutrons,
                    "variance_om",
                    PATH_store,
                    "Neutrons",
                    el,
                    tracks=tracks_label,
                )
                plot_one_label(
                    "Hadronic Response",
                    neutrons,
                    "mean",
                    PATH_store,
                    "Neutrons",
                    el,
                    tracks=tracks_label,
                )
                plot_one_label(
                    "Hadronic Resolution",
                    protons,
                    "variance_om",
                    PATH_store,
                    "Protons",
                    el,
                    tracks=tracks_label,
                )
                plot_one_label(
                    "Hadronic Response",
                    protons,
                    "mean",
                    PATH_store,
                    "Protons",
                    el,
                    tracks=tracks_label,
                )'''

def plot_per_energy_resolution2_multiple(
    matched_pandora, matched_all, PATH_store, tracks=False
):
    # matched_all: label -> matched df
    figs, axs = {}, {} # resolution
    figs_r, axs_r = {}, {} # response
    colors = {
        "DNN": "green",
        "GNN+DNN": "purple",
        "DNN w/o FT": "blue"
    }
    colors = {
        "DNN ~3 epochs": "green",
        "GNN+DNN ~3 epochs": "purple",
        "GNN+DNN ~13 epochs": "blue"
    }
    plot_pandora, plot_baseline = True, True
    for pid in [22, 11, 130, 211, 2112, 2212]:
        figs[pid], axs[pid] = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
        figs_r[pid], axs_r[pid] = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
    event_res_dic = {}
    fig_event_res, ax_event_res = plt.subplots(1, 1, figsize=(15, 10), sharex=False)
    for key in matched_all:
        matched_ = matched_all[key]
        mask = matched_["calibration_factor"] > 0
        matched_ = matched_[mask]
        if tracks:
            tracks_label = "tracks"
        else:
            tracks_label = ""
        plot_response = True
        if plot_response:
            list_plots = [""]  # "","_reco"
            event_res_dic[key] = get_response_for_event_energy(matched_pandora, matched_)
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
            neutrons = get_response_for_id_i(
                [2112], matched_pandora, matched_, tracks=tracks
            )
            protons = get_response_for_id_i(
                [2212], matched_pandora, matched_, tracks=tracks
            )
            plot_histograms("Event Energy Resolution", event_res_dic[key], fig_event_res, ax_event_res, plot_pandora, prefix=key + " ", color=colors[key])
            '''plot_one_label(
                "Event Energy Resolution",
                event_res_dic,
                "variance_om",
                PATH_store,
                "ML",
                "",
                tracks="",
                plot_baseline=True,
                fig=fig_event_energy, ax=ax_event_energy, plot_pandora=False, plot_baseline=False
            )'''

            for el in list_plots:
                plot_one_label(
                    "Electromagnetic Resolution",
                    photons_dic,
                    "variance_om",
                    PATH_store,
                    "Photons " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs[22], ax=axs[22], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                plot_one_label(
                    "Electromagnetic Response",
                    photons_dic,
                    "mean",
                    PATH_store,
                    "Photons " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs_r[22], ax=axs_r[22], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                plot_one_label(
                    "Electromagnetic Response",
                    electrons_dic,
                    "mean",
                    PATH_store,
                    "Electrons " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs_r[11], ax=axs_r[11], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                plot_one_label(
                    "Electromagnetic Resolution",
                    electrons_dic,
                    "variance_om",
                    PATH_store,
                    "Electrons " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs[11], ax=axs[11], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                plot_one_label(
                    "Hadronic Resolution",
                    hadrons_dic,
                    "variance_om",
                    PATH_store,
                    "KL " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs[130], ax=axs[130], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                plot_one_label(
                    "Hadronic Response",
                    hadrons_dic,
                    "mean",
                    PATH_store,
                    "KL " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs_r[130], ax=axs_r[130], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                plot_one_label(
                    "Hadronic Resolution",
                    hadrons_dic2,
                    "variance_om",
                    PATH_store,
                    "Pions " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs[211], ax=axs[211], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                plot_one_label(
                    "Hadronic Response",
                    hadrons_dic2,
                    "mean",
                    PATH_store,
                    "Pions " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs_r[211], ax=axs_r[211], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                # plot the neutrons and protons
                plot_one_label(
                    "Hadronic Resolution",
                    neutrons,
                    "variance_om",
                    PATH_store,
                    "Neutrons " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs[2112], ax=axs[2112], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                plot_one_label(
                    "Hadronic Response",
                    neutrons,
                    "mean",
                    PATH_store,
                    "Neutrons " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs_r[2112], ax=axs_r[2112], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                plot_one_label(
                    "Hadronic Resolution",
                    protons,
                    "variance_om",
                    PATH_store,
                    "Protons " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs[2212], ax=axs[2212], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                plot_one_label(
                    "Hadronic Response",
                    neutrons,
                    "mean",
                    PATH_store,
                    "Neutrons " + key,
                    el,
                    tracks=tracks_label,
                    fig=figs_r[2212], ax=axs_r[2212], save=False,
                    plot_pandora=plot_pandora, plot_baseline=plot_baseline, color=colors[key]
                )
                plot_pandora = False
                plot_baseline = False
    for key in figs:
        for a in axs[key]:
            a.grid()
        for a in axs_r[key]:
            a.grid()
        figs[key].tight_layout()
        figs[key].savefig(os.path.join(PATH_store, f"comparison_resolution_{key}.pdf"), bbox_inches="tight")
        figs_r[key].tight_layout()
        figs_r[key].savefig(os.path.join(PATH_store, f"comparison_response_{key}.pdf"), bbox_inches="tight")
    fig_event_res.savefig(os.path.join(PATH_store, "event_resolution.pdf"), bbox_inches="tight")
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


def calculate_phi(x, y, z=None):
    return torch.arctan2(y, x)
def calculate_eta(x, y, z):
    theta = torch.arctan2(torch.sqrt(x ** 2 + y ** 2), z)
    return -torch.log(torch.tan(theta / 2))


def plot_event(df, pandora=True, output_dir="", graph=None):
    # plot the event with plotly. Compare ML and pandora reconstructed with truth
    # also plotst the graph is specified
    # also plot eta-phi (a bit easier debugging)
    import plotly
    import plotly.graph_objs as go
    import plotly.express as px
    # arrows from 0,0,0 to df.true_pos and the hover text should be the true energy (df.true_showers_E)
    fig = go.Figure()
    scale = 1. # size of the direction vector, to make it easier to see it with the hits
    # list of 20 random colors
    # color_list = px.colors.qualitative.Plotly # this is only 10, not enough
    color_list = px.colors.qualitative.Light24
    if graph is not None:
        sum_hits = scatter_sum(graph.ndata["e_hits"].flatten(), graph.ndata["particle_number"].long())
        hit_pos = graph.ndata["pos_hits_xyz"].numpy()
        #fig.add_trace(go.Scatter3d(x=hit_pos[:, 0], y=hit_pos[:, 1], z=hit_pos[:, 2], mode='markers', color=graph.ndata["particle_number"], name='hits'))
        # fix this: color by particle number (it is an array of size [0,0,0,0,1,1,1,2,2,2...]
        ht = [f"part. {int(i)}, sum_hits={sum_hits[i]:.2f}" for i in graph.ndata["particle_number"].long().tolist()]
        fig.add_trace(go.Scatter3d(x=hit_pos[:, 0], y=hit_pos[:, 1], z=hit_pos[:, 2], mode='markers', marker=dict(size=4, color=[color_list[int(i.item())] for i in graph.ndata["particle_number"]]), name='hits', hovertext=ht, hoverinfo="text"))
        # set scale to avg hig_pos
        scale = np.mean(np.linalg.norm(hit_pos, axis=1))

        if "pos_pxpypz" in graph.ndata.keys():
            vectors = graph.ndata["pos_pxpypz"].numpy()
            trks = graph.ndata["pos_hits_xyz"].numpy()
            #filt = (vectors[:, 0] != 0) & (vectors[:, 1] != 0) & (vectors[:, 2] != 0)
            filt = graph.ndata["h"][:, 7 ] > 0
            vf = vectors[filt]
            vectors = vectors[filt]#
            trks = trks[filt] # track positions
            # normalize 3-comp vectors / np.linalg.norm(vectors, axis=1) * scale # remove zero vectors
            vectors = vectors / np.linalg.norm(vectors, axis=1).reshape(-1, 1) * scale
            track_p = graph.ndata["h"][:, 7].numpy()[filt]
            # plot these vectors
            for i in range(len(vectors)):
                def plot_single_arrow(fig, vec, hovertext="", init_pt=[0,0,0]):
                    # init_pt: initial point of the vector
                    fig.add_trace(go.Scatter3d(x=[init_pt[0], vec[0] + init_pt[0]], y=[init_pt[1], init_pt[1] + vec[1]], z=[init_pt[2], init_pt[2] + vec[2]], mode='lines', line=dict(color='black', width=1)))
                    fig.add_trace(go.Scatter3d(x=[vec[0] + init_pt[0]], y=[vec[1] + init_pt[1]], z=[vec[2] + init_pt[2]], mode='markers', marker=dict(size=4, color='black'), hovertext=hovertext))
                plot_single_arrow(fig, vectors[i] / 5, hovertext=f"track {track_p[i]} , pxpypz={vf[i]}", init_pt=trks[i])  # a bit smaller
        # color this by graph.ndata["particle_number"]
    truepos = np.array(df.true_pos.values.tolist()) * scale
    pids = [str(x) for x in df.pid.values]
    col = np.arange(1, len(truepos) + 1)
    true_E = df.true_showers_E.values
    true_P = np.array(df.true_pos.values.tolist()) * true_E.reshape(-1, 1)
    ht = [f"GT E={true_E[i]:.2f}, PID={pids[i]} , p={true_P[i]}" for i in range(len(true_E))]
    fig.add_trace(go.Scatter3d(x=truepos[:, 0], y=truepos[:, 1], z=truepos[:, 2], mode='markers',  marker=dict(size=4, color=[color_list[c] for c in col]), name='ground truth', hovertext=ht, hoverinfo="text"))
    # add lines from 0,0,0 to these points
    for i in range(len(truepos)):
        fig.add_trace(go.Scatter3d(
            x=[0, truepos[i, 0]], y=[0, truepos[i, 1]], z=[0, truepos[i, 2]],
            mode='lines', line=dict(color='blue', width=1)
        ))
    if pandora:
        pandorapos = np.array(df.pandora_calibrated_pos.values.tolist()) * scale

        fig.add_trace(go.Scatter3d(x=pandorapos[:, 0], y=pandorapos[:, 1], z=pandorapos[:, 2], mode='markers', marker=dict(size=4, color='green'), name='Pandora', hovertext=df.pandora_calibrated_E.values, hoverinfo="text")
                      )
        # also add lines here
        for i in range(len(pandorapos)):
            fig.add_trace(go.Scatter3d(
                x=[0, pandorapos[i, 0]], y=[0, pandorapos[i, 1]], z=[0, pandorapos[i, 2]],
                mode='lines', line=dict(color='green', width=1))
            )
    else:
        predpos = np.array(df.pred_pos_matched.values.tolist()) * scale
        fig.add_trace(
            go.Scatter3d(x=predpos[:, 0], y=predpos[:, 1], z=predpos[:, 2],
                         mode='markers', marker=dict(size=4, color='red'), name='ML', hovertext=df.calibrated_E.values))
        # add lines
        for i in range(len(predpos)):
            fig.add_trace(go.Scatter3d
                            (x=[0, predpos[i, 0]], y=[0, predpos[i, 1]], z=[0, predpos[i, 2]],
                             mode='lines', line=dict(color='red', width=1))
                            )
    #fig.show()
    assert output_dir != ""
    plotly.offline.plot(fig, filename=output_dir + "event.html")


def calculate_event_energy_resolution(df, pandora=False, full_vector=False):
    bins = [0, 700]
    #if pandora and "pandora_calibrated_pos" in df.columns:
    #    full_vector = True
    #else:
    #    full_vector = False
    if full_vector and pandora:
        assert "pandora_calibrated_pos" in df.columns
    bins = [0, 700]
    binsx = []
    mean = []
    variance = []
    distributions = []
    distr_baseline = []
    mean_baseline = []
    variance_baseline = []
    binning = 1e-2
    bins_per_binned_E = np.arange(0, 2, binning)
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        binsx.append(0.5 * (bin_i + bin_i1))
        true_e = df.true_showers_E.values
        batch_idx = df.number_batch
        if pandora:
            pred_e = df.pandora_calibrated_E.values
            pred_e1 = torch.tensor(pred_e).unsqueeze(1).repeat(1, 3)
            if full_vector:
                pred_vect = np.array(df.pandora_calibrated_pos.values.tolist()) * pred_e1.numpy()
                true_vect = np.array(df.true_pos.values.tolist())*torch.tensor(true_e).unsqueeze(1).repeat(1, 3).numpy()
                pred_vect = torch.tensor(pred_vect)
                true_vect = torch.tensor(true_vect)
        else:
            pred_e = df.calibrated_E.values
            pred_e1 = torch.tensor(pred_e).unsqueeze(1).repeat(1, 3)
            if full_vector:
                pred_vect = np.array(df.pred_pos_matched.values.tolist()) * pred_e1.numpy()
                true_vect = np.array(df.true_pos.values.tolist())*torch.tensor(true_e).unsqueeze(1).repeat(1, 3).numpy()
                pred_vect = torch.tensor(pred_vect)
                true_vect = torch.tensor(true_vect)
        true_rec = df.reco_showers_E
        # pred_e_nocor = df.pred_showers_E[mask]
        true_e = torch.tensor(true_e)
        batch_idx = torch.tensor(batch_idx.values).long()
        pred_e = torch.tensor(pred_e)
        true_rec = torch.tensor(true_rec.values)
        if full_vector:
            # vector scatter_sum of pred_vect and true_vect
            #true_e = scatter_sum(true_e, batch_idx)
            #pred_e = scatter_sum(pred_e, batch_idx)
            #true_e_1 = scatter_sum(true_vect[:, 0], batch_idx)
            #true_e_2 = scatter_sum(true_vect[:, 1], batch_idx)
            #true_e_3 = scatter_sum(true_vect[:, 2], batch_idx) # for some reason scatter doens't work with more axes?
            true_e = scatter_sum(true_vect, batch_idx, dim=0)
            pred_e = scatter_sum(pred_vect, batch_idx, dim=0)
            true_e = torch.norm(true_e, dim=1)
            pred_e = torch.norm(pred_e, dim=1)
        else:
            true_e = scatter_sum(true_e, batch_idx)
            pred_e = scatter_sum(pred_e, batch_idx)
        true_rec = scatter_sum(true_rec, batch_idx)
        mask_above = true_e <= bin_i1
        mask_below = true_e > bin_i
        mask_check = true_e > 0
        mask = mask_below * mask_above * mask_check
        true_e = true_e[mask]
        true_rec = true_rec[mask]
        pred_e = pred_e[mask]
        if torch.sum(mask) > 0:  # if the bin is not empty
            e_over_true = pred_e / true_e
            e_over_reco = true_rec / true_e
            distributions.append(e_over_true)
            distr_baseline.append(e_over_reco)
            mean_predtotrue, var_predtotrue, err_mean_predtotrue, err_var_predtotrue = get_sigma_gaussian(
                e_over_true, bins_per_binned_E
            )
            mean_reco_true, var_reco_true, err_mean_reco_true, err_var_reco_true = get_sigma_gaussian(
                e_over_reco, bins_per_binned_E
            )
            mean.append(mean_predtotrue)
            variance.append(np.abs(var_predtotrue))
            mean_baseline.append(mean_reco_true)
            variance_baseline.append(np.abs(var_reco_true))
    return (
       mean, variance, distributions, binsx, mean_baseline, variance_baseline, distr_baseline
    )

def get_response_for_event_energy(matched_pandora, matched_):
    mean_p, variance_om_p, distr_p, x_p, _ , _ , _ = calculate_event_energy_resolution(matched_pandora, True, True)
    mean, variance_om, distr, x, mean_baseline, variance_om_baseline, _ = calculate_event_energy_resolution(matched_, False, True)
    dic = {}
    dic["mean_p"] = mean_p
    dic["variance_om_p"] = variance_om_p
    dic["variance_om"] = variance_om
    dic["mean"] = mean
    dic["energy_resolutions"] = x
    dic["energy_resolutions_p"] = x_p
    dic["mean_baseline"] = mean_baseline
    dic["variance_om_baseline"] = variance_om_baseline
    dic["distributions_pandora"] = distr_p
    dic["distributions_model"] = distr
    return dic

def calculate_response(matched, pandora, log_scale=False, tracks=False):
    if log_scale:
        bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    else:
        #bins = np.arange(0, 51, 6)
        bins = [0, 5, 15, 35, 50]
    mean = []
    variance_om = []
    mean_baseline = []
    variance_om_baseline = []
    mean_true_rec = []
    variance_om_true_rec = []
    mean_errors = []
    variance_om_errors = []
    energy_resolutions = []
    energy_resolutions_reco = []
    distributions = [] # distributions of E/Etrue for plotting later
    # tic = time.time()
    # vector = range(len(bins) - 1)
    # output_results = parallel_process(vector, bins, matched, pandora, bins_per_binned_E)
    # mean_true_rec = [r[0] for ind, r in enumerate(output_results)]
    # variance_om_true_rec = [r[1] for ind, r in enumerate(output_results)]
    # energy_resolutions_reco = [r[2] for ind, r in enumerate(output_results)]
    # toc = time.time()
    # print("time with paralel version", toc - tic)
    print("START PANDORA")
    binning = 1e-2
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
            distributions.append(e_over_true)
            mean_predtotrue, var_predtotrue, err_mean_predtotrue, err_var_predtotrue = get_sigma_gaussian(
                e_over_true, bins_per_binned_E
            )
            print("Pandora", pandora, "err_var", err_var_predtotrue)
            mean_reco_true, var_reco_true, err_mean_reco_true, err_var_reco_true = get_sigma_gaussian(
                e_over_reco, bins_per_binned_E
            )
            mean_reco_ML, var_reco_ML, err_mean_reco_ML, err_mean_var_reco_ML = get_sigma_gaussian(
                e_over_reco_ML, bins_per_binned_E
            )
            # raise err if mean_reco_ML is nan
            if np.isnan(mean_reco_ML):
                raise ValueError("mean_reco_ML is nan")
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
            mean_errors.append(err_mean_predtotrue)
            variance_om_errors.append(err_var_predtotrue)

    return (
        mean,
        variance_om,
        mean_true_rec,
        variance_om_true_rec,
        energy_resolutions,
        energy_resolutions_reco,
        mean_baseline,
        variance_om_baseline,
        distributions,
        mean_errors,
        variance_om_errors,
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


def plot_one_label(title, photons_dic, y_axis, PATH_store, label1, reco, tracks="", fig=None, ax=None, save=True, plot_pandora=True, plot_baseline=True, color=None):
    if reco == "":
        label_add = " raw"
        label_add_pandora = " corrected"
    else:
        label_add = " raw"
        label_add_pandora = " raw"
    colors_list = ["#FF0000", "#FF0000", "#0000FF"]
    if color is not None:
        colors_list[1] = color
    fig_distr, ax_distr = plt.subplots(len(photons_dic["energy_resolutions" + reco]), 1, figsize=(14, 18), sharex=True)
    if title == "Event Energy Resolution":
        fig_distr, ax_distr = plt.subplots(len(photons_dic["energy_resolutions" + reco]), 1, figsize=(14, 10), sharex=True)
    if not type(ax_distr) == list and not type(ax_distr) == np.ndarray:
        ax_distr = [ax_distr]
    for i in range(len(photons_dic["energy_resolutions" + reco])):
        distr_model = photons_dic["distributions_model"][i]
        distr_pandora = photons_dic["distributions_pandora"][i]
        if type(distr_model) == torch.Tensor:
            distr_model = distr_model.numpy()
            distr_pandora = distr_pandora.numpy()
        else:
            distr_model = distr_model.values
            distr_pandora = distr_pandora.values
        max_distr_model = np.max(distr_model)
        max_distr_pandora = np.max(distr_pandora)
        # remove everything higher than 2.0 and note the fraction of such events
        mask = distr_model < 2.0
        distr_model = distr_model[mask]
        frac_model_dropped = int((1 - len(distr_model) / len(photons_dic["distributions_model"][i]))*1000)
        mask = distr_pandora < 2.0
        distr_pandora = distr_pandora[mask]
        frac_pandora_dropped = int((1 - len(distr_pandora) / len(photons_dic["distributions_pandora"][i]))*1000)
        mu = photons_dic["mean"][i]
        sigma = (photons_dic["variance_om"][i]) * mu
        mu_pandora = photons_dic["mean_p"][i]
        sigma_pandora = (photons_dic["variance_om_p"][i]) * mu
        ax_distr[i].hist(distr_model, bins=np.arange(0, 2, 1e-2), color="blue", label="ML μ={} σ={}".format(round(mu, 2), round(sigma, 2)), alpha=0.5, histtype="step")
        ax_distr[i].hist(distr_pandora, bins=np.arange(0, 2, 1e-2), color="red", label="Pandora μ={} σ={}".format(round(mu_pandora, 2), round(sigma_pandora, 2)), alpha=0.5, histtype="step")
        # ALSO PLOT MU AND SIGMA #
        ax_distr[i].axvline(mu, color="blue", linestyle="-", ymin=0.95, ymax=1.0)
        ax_distr[i].axvline(mu + sigma, color="blue", linestyle="--", ymin=0.95, ymax=1.0)
        ax_distr[i].axvline(mu - sigma, color="blue", linestyle="--", ymin=0.95, ymax=1.0)
        ax_distr[i].axvline(mu_pandora, color="red", linestyle="-", ymin=0.95, ymax=1.0)
        ax_distr[i].axvline(mu_pandora + sigma_pandora, color="red", linestyle="--", ymin=0.95, ymax=1.0)
        ax_distr[i].axvline(mu_pandora - sigma_pandora, color="red", linestyle="--", ymin=0.95, ymax=1.0)
        # variance_om
        ax_distr[i].set_xlabel("E/Etrue")
        ax_distr[i].set_xlim([0, 2])
        ax_distr[i].set_title(f"{title} {photons_dic['energy_resolutions' + reco][i]:.2f} GeV / max model: " + str(max_distr_model) + " / max pandora: " + str(max_distr_pandora))
        ax_distr[i].legend()
        ax_distr[i].set_yscale("log")
    fig_distr.tight_layout()
    if fig is None or ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    j = 0
    ax[1].set_xlabel("Energy [GeV]", fontsize=30)
    ax[0].set_xlabel("Energy [GeV]", fontsize=30)
    # ax[row_i, j].set_xscale("log")
    ax[0].set_title(title, fontsize=30)
    ax[0].grid()
    ax[1].grid()
    ax[1].set_yscale("log")
    '''f y_axis == "mean":
        # error is the mean error
        errors = photons_dic["mean_errors"]
        pandora_errors = photons_dic["mean_errors_p"]
    else:
        errors = photons_dic["variance_errors"]
        pandora_errors = photons_dic["variance_errors_p"]'''
    for a in ax:
        a.errorbar(
            photons_dic["energy_resolutions" + reco],
            photons_dic[y_axis + reco],
            #yerr=errors,
            color=colors_list[1],
            #edgecolors=colors_list[1],
            label=label1,
            marker="x",
            markersize=8,
            linestyle="None",
            # error color
            ecolor=colors_list[1],
            capsize=5,
        )
        if plot_pandora:
            a.errorbar(
                photons_dic["energy_resolutions_p" + reco],
                photons_dic[y_axis + "_p" + reco],
                #yerr=pandora_errors,
                color=colors_list[2],
                #edgecolors=colors_list[2],
                label="Pandora",
                marker="x",
                markersize=8,
                capsize=5,
                ecolor=colors_list[2],
                linestyle="None"
            )

    if title == "Electromagnetic Resolution" or title == "Hadronic Resolution":
        if reco == "":
            for a in ax:
                if plot_baseline:
                    a.scatter(
                        photons_dic["energy_resolutions"],
                        photons_dic["variance_om_baseline"],
                        facecolors="black",
                        edgecolors="black",
                        marker=".",
                        label="Baseline",
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
                #dic0_fit,
                dic1_fit,
                #dic1_fit_pandora,
            ]
            color_list_fits_l1 = [
                #"black",
                colors_list[1],
                #colors_list[2],
            ]
            line_type_fits_l1 = ["-"]#, "-", "-."]
            if plot_baseline:
                fits_l1.append(dic0_fit)
                color_list_fits_l1.append("black")
                line_type_fits_l1.append("-")
            if plot_pandora:
                fits_l1.append(dic1_fit_pandora)
                color_list_fits_l1.append(colors_list[2])
                line_type_fits_l1.append("-.")
            for a in ax:
                plot_fit(fits_l1, line_type_fits_l1, color_list_fits_l1, ax=a)
        else:
            raise NotImplementedError
            #line_type_fits = ["-", "-."]
            #for a in ax:
            #    plot_fit(fits, line_type_fits, color_list_fits, ax=a)
        if reco == "_reco":
            plt.yscale("log")
        else:
            if title == "Electromagnetic Resolution":
                ymax = 0.3
            else:
                ymax = 0.6
            ax[0].set_ylim([0, ymax])
            ax[1].set_ylim([0, ymax])
        ax[0].set_xlim([0, 55])
        ax[1].set_xlim([0, 55])
        ylabel = r"$\frac{\sigma_{E_{reco}}}{\langle E_{reco} \rangle}$"
        ax[0].set_ylabel(ylabel, fontsize=30)
        ax[1].set_ylabel(ylabel, fontsize=30)
    else:
        ylabel = r"$\langle E_{reco} \rangle / E_{true}$"
        ax[0].set_ylabel(ylabel, fontsize=30)
        ax[1].set_ylabel(ylabel, fontsize=30)
    # loc="upper right",
    #plt.tick_params(axis="both", which="major", labelsize=40)
    ax[0].tick_params(axis="both", which="major", labelsize=30)
    ax[1].tick_params(axis="both", which="major", labelsize=30)
    if title == "Electromagnetic Response" or title == "Hadronic Response":
        ax[0].set_ylim([0.6, 1.4])
        ax[1].set_ylim([0.6, 1.4])
    ax[0].legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc="upper left")
    label = label1
    if save:
        fig.tight_layout()
        fig.savefig(
            PATH_store + title + reco + label + tracks + "_v1.pdf", bbox_inches="tight"
        )
        fig_distr.savefig(PATH_store + title + reco + label + tracks + "_v1_distributions.pdf", bbox_inches="tight")


def plot_histograms(title, photons_dic, fig_distr, ax_distr, plot_pandora, prefix="ML ", color="blue", normalize=True):
    assert title == "Event Energy Resolution" # fix
    #if title == "Event Energy Resolution":
    #    fig_distr, ax_distr = plt.subplots(len(photons_dic["energy_resolutions"]), 1, figsize=(14, 10), sharex=True)
    #if not type(ax_distr) == list and not type(ax_distr) == np.ndarray:
    #    ax_distr = [ax_distr]
    distr_model = photons_dic["distributions_model"][0]
    distr_pandora = photons_dic["distributions_pandora"][0]
    if type(distr_model) == torch.Tensor:
        distr_model = distr_model.numpy()
        distr_pandora = distr_pandora.numpy()
    else:
        distr_model = distr_model.values
        distr_pandora = distr_pandora.values
    # max_distr_model = np.max(distr_model)
    # max_distr_pandora = np.max(distr_pandora)
    # remove everything higher than 2.0 and note the fraction of such events
    mask = distr_model < 2.0
    distr_model = distr_model[mask]
    mask = distr_pandora < 2.0
    distr_pandora = distr_pandora[mask]
    mu = photons_dic["mean"][0]
    sigma = (photons_dic["variance_om"][0]) * mu
    mu_pandora = photons_dic["mean_p"][0]
    sigma_pandora = (photons_dic["variance_om_p"][0]) * mu
    ax_distr.hist(distr_model, bins=np.arange(0, 2, 1e-2), color=color, label=prefix + "μ={} σ={}".format(round(mu, 2), round(sigma, 2)), alpha=0.5, histtype="step", density=normalize)
    if plot_pandora:
        ax_distr.hist(distr_pandora, bins=np.arange(0, 2, 1e-2), color="red", label="Pandora μ={} σ={}".format(round(mu_pandora, 2), round(sigma_pandora, 2)), alpha=0.5, histtype="step", density=normalize)
    # ALSO PLOT MU AND SIGMA #
    ax_distr.axvline(mu, color=color, linestyle="-", ymin=0.95, ymax=1.0)
    ax_distr.axvline(mu + sigma, color=color, linestyle="--", ymin=0.95, ymax=1.0)
    ax_distr.axvline(mu - sigma, color=color, linestyle="--", ymin=0.95, ymax=1.0)
    ax_distr.axvline(mu_pandora, color="red", linestyle="-", ymin=0.95, ymax=1.0)
    ax_distr.axvline(mu_pandora + sigma_pandora, color="red", linestyle="--", ymin=0.95, ymax=1.0)
    ax_distr.axvline(mu_pandora - sigma_pandora, color="red", linestyle="--", ymin=0.95, ymax=1.0)
    # variance_om
    ax_distr.set_xlabel("$E_{reco} / E_{true}$")
    ax_distr.set_xlim([0, 2])
    ax_distr.set_title(f"{title}")
    ax_distr.legend()
    ax_distr.set_yscale("log")
    fig_distr.tight_layout()
