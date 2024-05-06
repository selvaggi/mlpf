import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.rc("font", size=35)
import matplotlib.pyplot as plt


def plot_per_event_metrics(sd, sd_pandora, PATH_store=None):
    (
        calibrated_list,
        calibrated_list_pandora,
        reco_list,
        reco_list_pandora,
    ) = calculate_energy_per_event(sd, sd_pandora)
    plot_per_event_energy_distribution(
        calibrated_list,
        calibrated_list_pandora,
        reco_list,
        reco_list_pandora,
        PATH_store,
    )


def calculate_energy_per_event(
    sd,
    sd_pandora,
):
    sd = sd.reset_index(drop=True)
    sd_pandora = sd_pandora.reset_index(drop=True)
    corrected_list = []
    reco_list = []
    reco_list_pandora = []
    corrected_list_pandora = []
    for i in range(0, int(np.max(sd.number_batch))):
        mask = sd.number_batch == i
        event_E_total_reco = np.nansum(sd.reco_showers_E[mask])
        event_E_total_reco_corrected = np.nansum(sd.calibrated_E[mask])
        event_ML_total_reco = np.nansum(sd.pred_showers_E[mask])
        mask_p = sd_pandora.number_batch == i
        event_E_total_reco_p = np.nansum(sd_pandora.reco_showers_E[mask_p])
        event_ML_total_reco_p = np.nansum(sd_pandora.reco_showers_E[mask_p])
        event_ML_total_reco_p_corrected = np.nansum(
            sd_pandora.pandora_calibrated_pfo[mask_p]
        )

        reco_list.append(event_ML_total_reco / event_E_total_reco)
        corrected_list.append(event_E_total_reco_corrected / event_E_total_reco)
        reco_list_pandora.append(event_ML_total_reco_p / event_E_total_reco_p)
        corrected_list_pandora.append(
            event_ML_total_reco_p_corrected / event_E_total_reco_p
        )
    return corrected_list, corrected_list_pandora, reco_list, reco_list_pandora


def plot_per_event_energy_distribution(
    calibrated_list, calibrated_list_pandora, reco_list, reco_list_pandora, PATH_store
):
    fig = plt.figure(figsize=(8, 8))
    sns.histplot(
        data=np.array(calibrated_list) + 1 - np.mean(calibrated_list),
        stat="percent",
        binwidth=0.1,
        label="MLPF",
        element="step",
        fill=False,
        color="red",
        linewidth=2,
    )
    sns.histplot(
        data=calibrated_list_pandora,
        stat="percent",
        color="blue",
        binwidth=0.1,
        label="Pandora",
        element="step",
        fill=False,
        linewidth=2,
    )
    plt.ylabel("Percent of events")
    plt.xlabel("$E_{corrected}/E_{total}$")
    plt.yscale("log")
    plt.legend()
    plt.xlim([0, 2])
    fig.savefig(
        PATH_store + "per_event_E.png",
        bbox_inches="tight",
    )
    fig = plt.figure(figsize=(8, 8))
    sns.histplot(data=reco_list, stat="percent", binwidth=0.05, label="MLPF")
    sns.histplot(
        data=reco_list_pandora,
        stat="percent",
        color="orange",
        binwidth=0.05,
        label="Pandora",
    )
    plt.ylabel("Percent of events")
    plt.xlabel("$E_{recoML}/E_{reco}$")
    plt.legend()
    plt.xlim([0.5, 1.5])
    plt.yscale("log")
    fig.savefig(
        PATH_store + "per_event_E_reco.png",
        bbox_inches="tight",
    )
