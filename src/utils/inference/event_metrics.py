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
    calibrated_list = []
    reco_list = []

    calibrated_list_pandora = []
    reco_list_pandora = []
    number_of_showers = len(sd["number_batch"].values)
    counter_i = 0
    counter_total = 0
    while number_of_showers > counter_total + 20:
        sum_by = np.argmax(sd["number_batch"].values != sd["number_batch"].values[0])
        print("sum_by", sum_by)
        if sum_by < 1:
            print("sum by not enough", counter_i)
            print(sd)
            break
        temp_sd = sd[0:sum_by]
        counter_total = counter_total + sum_by
        total_e_event = np.nansum(temp_sd["true_showers_E"].values)
        total_e_reco = np.nansum(temp_sd["reco_showers_E"].values)
        total_e_ML_cali = np.nansum(temp_sd["calibrated_E"].values)
        total_e_reco_ML = np.nansum(temp_sd["pred_showers_E"].values)
        calibrated_list.append(total_e_ML_cali / total_e_event)
        reco_list.append(total_e_reco_ML / total_e_reco)
        sd = sd.drop(np.arange(0, sum_by))
        counter_i = np.mod(counter_i + 1, 4)
        sd = sd.reset_index(drop=True)
        print("1", number_of_showers, counter_total, number_of_showers - counter_total)

    number_of_showers = len(sd_pandora["number_batch"].values)
    counter_i = 0
    counter_total = 0

    while number_of_showers > counter_total + 20:
        sum_by = np.argmax(sd_pandora["number_batch"].values != counter_i)
        temp_sd = sd_pandora[0:sum_by]
        mask = (temp_sd["pred_showers_E"] > 0.6) * (
            np.isnan(temp_sd["true_showers_E"])
        ) + (~np.isnan(temp_sd["true_showers_E"]))
        temp_sd = temp_sd[mask]
        counter_total = counter_total + sum_by
        total_e_event = np.nansum(temp_sd["true_showers_E"].values)
        total_e_reco = np.nansum(temp_sd["reco_showers_E"].values)
        total_e_ML_cali = np.nansum(temp_sd["pandora_calibrated_pfo"].values)
        total_e_reco_ML = np.nansum(temp_sd["pred_showers_E"].values)
        calibrated_list_pandora.append(total_e_ML_cali / total_e_event)
        reco_list_pandora.append(total_e_reco_ML / total_e_reco)
        sd_pandora = sd_pandora.drop(np.arange(0, sum_by))
        counter_i = np.mod(counter_i + 1, 4)
        sd_pandora = sd_pandora.reset_index(drop=True)
        print("2", number_of_showers, counter_total, number_of_showers - counter_total)

    return calibrated_list, calibrated_list_pandora, reco_list, reco_list_pandora


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
