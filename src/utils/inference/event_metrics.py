import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.rc("font", size=35)
import matplotlib.pyplot as plt
import torch
from src.utils.inference.inference_metrics import get_sigma_gaussian
from torch_scatter import scatter_sum, scatter_mean

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
        event_E_total_true = np.nansum(sd.true_showers_E[mask])
        event_E_total_reco_corrected = np.nansum(sd.calibrated_E[mask])
        event_ML_total_reco = np.nansum(sd.pred_showers_E[mask])
        mask_p = sd_pandora.number_batch == i
        event_E_total_reco_p = np.nansum(sd_pandora.reco_showers_E[mask_p])
        event_E_total_true_p = np.nansum(sd_pandora.true_showers_E[mask_p])
        event_ML_total_reco_p = np.nansum(sd_pandora.pred_showers_E[mask_p])
        event_ML_total_reco_p_corrected = np.nansum(
            sd_pandora.pandora_calibrated_pfo[mask_p]
        )

        reco_list.append(event_ML_total_reco / event_E_total_reco)
        corrected_list.append(event_E_total_reco_corrected / event_E_total_true)
        reco_list_pandora.append(event_ML_total_reco_p / event_E_total_reco_p)
        corrected_list_pandora.append(
            event_ML_total_reco_p_corrected / event_E_total_true_p
        )
    return corrected_list, corrected_list_pandora, reco_list, reco_list_pandora


def plot_per_event_energy_distribution(
    calibrated_list, calibrated_list_pandora, reco_list, reco_list_pandora, PATH_store
):
    fig = plt.figure(figsize=(8, 8))
    sns.histplot(
        data=np.array(calibrated_list),  # + 1 - np.mean(calibrated_list)
        stat="percent",
        binwidth=0.01,
        label="MLPF",
        # element="step",
        # fill=False,
        color="red",
        # linewidth=2,
    )
    sns.histplot(
        data=calibrated_list_pandora,
        stat="percent",
        color="blue",
        binwidth=0.01,
        label="Pandora",
        # element="step",
        # fill=False,
        # linewidth=2,
    )
    plt.ylabel("Percent of events")
    plt.xlabel("$E_{corrected}/E_{total}$")
    # plt.yscale("log")
    plt.legend()
    plt.xlim([0, 2])
    fig.savefig(
        PATH_store + "per_event_E.png",
        bbox_inches="tight",
    )
    fig = plt.figure(figsize=(8, 8))
    sns.histplot(data=reco_list, stat="percent", binwidth=0.01, label="MLPF")
    sns.histplot(
        data=reco_list_pandora,
        stat="percent",
        color="orange",
        binwidth=0.01,
        label="Pandora",
    )
    plt.ylabel("Percent of events")
    plt.xlabel("$E_{recoML}/E_{reco}$")
    plt.legend()
    plt.xlim([0.5, 1.5])
    # plt.yscale("log")
    fig.savefig(
        PATH_store + "per_event_E_reco.png",
        bbox_inches="tight",
    )

particle_masses = {0: 0, 22: 0, 11: 0.00511, 211: 0.13957, 130: 0.493677, 2212: 0.938272, 2112: 0.939565}
particle_masses_4_class = {0: 0.00511, 1: 0.13957, 2: 0.939565, 3: 0.0} # electron, CH, NH, photon

def safeint(x, default_val=0):
    if np.isnan(x):
        return default_val
    return int(x)


def calculate_event_mass_resolution(df, pandora, perfect_pid=False, mass_zero=False, ML_pid=False):
    true_e = torch.Tensor(df.true_showers_E.values)
    mask_nan_true = np.isnan(df.true_showers_E.values)
    true_e[mask_nan_true] = 0
    batch_idx = df.number_batch
    if pandora:
        pred_E = df.pandora_calibrated_pfo.values
        nan_mask = np.isnan(df.pandora_calibrated_pfo.values)
        pred_E[nan_mask] = 0
        pred_e1 = torch.tensor(pred_E).unsqueeze(1).repeat(1, 3)
        pred_vect = torch.tensor(np.array(df.pandora_calibrated_pos.values.tolist()))
        nan_mask_p = torch.isnan(pred_vect).any(dim=1)
        pred_vect[nan_mask_p] = 0
        true_vect = torch.tensor(np.array(df.true_pos.values.tolist()))
        mask_nan_p = torch.isnan(true_vect).any(dim=1)
        true_vect[mask_nan_true] = 0
    else:
        pred_E = df.calibrated_E.values
        nan_mask = np.isnan(df.calibrated_E.values)
        print(np.sum(nan_mask))
        pred_E[nan_mask] = 0
        pred_e1 = torch.tensor(pred_E).unsqueeze(1).repeat(1, 3)
        pred_vect = torch.tensor(
            np.array(df.pred_pos_matched.values.tolist())
        )
        pred_vect[nan_mask] = 0
        true_vect = torch.tensor(
            np.array(df.true_pos.values.tolist())
        )
        true_vect[mask_nan_true] = 0
    if perfect_pid or mass_zero or ML_pid:
        pred_vect /= np.linalg.norm(pred_vect, axis=1).reshape(-1, 1)
        pred_vect[np.isnan(pred_vect)] = 0
        if ML_pid:
            #assert pandora is False
            if pandora:
                print("perfect PID for Pandora")
                m = np.array([particle_masses.get(abs(safeint(i)), 0) for i in df.pid])
            else:
                m = np.array([particle_masses_4_class.get(safeint(i), 0) for i in df.pred_pid_matched.values])
        else:
            m = np.array([particle_masses.get(abs(safeint(i)), 0) for i in df.pid])
        if mass_zero:
            m = np.array([0 for _ in m])
        p_squared = (pred_E ** 2 - m ** 2)
        p_squared[p_squared < 0] = 0 # they are always like of order -1e-8
        pred_vect = np.sqrt(p_squared).reshape(-1, 1) * np.array(pred_vect)
    batch_idx = torch.tensor(batch_idx.values).long()
    pred_E = torch.tensor(pred_E)
    true_jet_vect = scatter_sum(true_vect, batch_idx, dim=0)
    pred_jet_vect = scatter_sum(torch.tensor(pred_vect), batch_idx, dim=0)
    true_E_jet = scatter_sum(torch.tensor(true_e), batch_idx)
    pred_E_jet = scatter_sum(torch.tensor(pred_E), batch_idx)
    true_jet_p = torch.norm(true_jet_vect, dim=1)  # This is actually momentum resolution
    pred_jet_p = torch.norm(pred_jet_vect, dim=1)
    mass_true = torch.sqrt(torch.abs(true_E_jet ** 2) - true_jet_p ** 2)
    mass_pred_p = torch.sqrt(
        torch.abs(pred_E_jet ** 2) - pred_jet_p ** 2)  ## TODO: fix the nan values in pred_jet_p!!!!!
    # replace nans in these with 0
    mass_over_true_p = mass_pred_p / mass_true
    E_over_true = pred_E_jet / true_E_jet
    p_over_true = pred_jet_p / true_jet_p
    p_jet_pandora = pred_jet_p
    (
        mean_mass,
        var_mass,
        _,
        _,
    ) = get_sigma_gaussian(mass_over_true_p, np.linspace(0, 4, 300))
    return mean_mass, var_mass, mass_over_true_p, mass_true, p_over_true, true_jet_p, E_over_true


def calculate_event_energy_resolution(df, pandora=False, full_vector=False):
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
    mass_list = []
    binning = 1e-2
    bins_per_binned_E = np.arange(0, 2, binning)
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        binsx.append(0.5 * (bin_i + bin_i1))
        true_e = df.true_showers_E.values
        batch_idx = df.number_batch
        if pandora:
            pred_e = df.pandora_calibrated_pfo.values
            pred_e1 = torch.tensor(pred_e).unsqueeze(1).repeat(1, 3)
            if full_vector:
                pred_vect = (
                    np.array(df.pandora_calibrated_pos.values.tolist())
                    # * pred_e1.numpy()
                )
                true_vect = (
                    np.array(df.true_pos.values.tolist())
                    # * torch.tensor(true_e).unsqueeze(1).repeat(1, 3).numpy()
                )
                pred_vect = torch.tensor(pred_vect)
                true_vect = torch.tensor(true_vect)
        else:
            pred_e = df.calibrated_E.values
            pred_e1 = torch.tensor(pred_e).unsqueeze(1).repeat(1, 3)
            if full_vector:
                pred_vect = (
                    np.array(df.pred_pos_matched.values.tolist()) * pred_e1.numpy()
                )
                true_vect = (
                    np.array(df.true_pos.values.tolist())
                    # * torch.tensor(true_e).unsqueeze(1).repeat(1, 3).numpy()
                )
                pred_vect = torch.tensor(pred_vect)
                true_vect = torch.tensor(true_vect)
        true_rec = df.reco_showers_E
        # pred_e_nocor = df.pred_showers_E[mask]
        true_e = torch.tensor(true_e)
        batch_idx = torch.tensor(batch_idx.values).long()
        pred_e = torch.tensor(pred_e)
        true_rec = torch.tensor(true_rec.values)
        if full_vector:
            true_p_vect = scatter_sum(true_vect, batch_idx, dim=0)
            pred_p_vect = scatter_sum(pred_vect, batch_idx, dim=0)
            true_e1 = scatter_sum(torch.tensor(true_e), batch_idx)
            pred_e1 = scatter_sum(torch.tensor(pred_e), batch_idx)
            true_e = torch.norm(
                true_p_vect, dim=1
            )  # This is actually momentum resolution
            pred_e = torch.norm(pred_p_vect, dim=1)
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
            (
                mean_predtotrue,
                var_predtotrue,
                err_mean_predtotrue,
                err_var_predtotrue,
            ) = get_sigma_gaussian(e_over_true, bins_per_binned_E)
            if full_vector:
                mass_true = torch.sqrt(true_e1[mask] ** 2 - true_e**2)
                mass_pred = torch.sqrt(pred_e1[mask] ** 2 - pred_e**2)
                print(pandora, len(mass_true), len(mass_pred))
                mass_over_true = mass_pred / mass_true

                (
                    mean_mass,
                    var_mass,
                    _,
                    _,
                ) = get_sigma_gaussian(mass_over_true, bins_per_binned_E)
                mass_list.append(mass_over_true)
            (
                mean_reco_true,
                var_reco_true,
                err_mean_reco_true,
                err_var_reco_true,
            ) = get_sigma_gaussian(e_over_reco, bins_per_binned_E)
            mean.append(mean_predtotrue)
            variance.append(np.abs(var_predtotrue))
            mean_baseline.append(mean_reco_true)
            variance_baseline.append(np.abs(var_reco_true))
    if full_vector:
        mass_list = torch.cat(mass_list)
    ret = [
        mean,
        variance,
        distributions,
        binsx,
        mean_baseline,
        variance_baseline,
        distr_baseline,
    ]
    if full_vector:
        ret += [mass_list]
    else:
        ret += [None]
    return ret


def get_response_for_event_energy(matched_pandora, matched_, perfect_pid=False, mass_zero=False, ML_pid=False):
    (
        mean_p,
        variance_om_p,
        distr_p,
        x_p,
        _,
        _,
        _,
        mass_over_true_pandora,
    ) = calculate_event_energy_resolution(matched_pandora, True, False)
    (
        mean,
        variance_om,
        distr,
        x,
        mean_baseline,
        variance_om_baseline,
        _,
        mass_over_true_model,
    ) = calculate_event_energy_resolution(matched_, False, False)

    mean_mass_p, var_mass_p, distr_mass_p, mass_true_p, _, _, E_over_true_pandora = calculate_event_mass_resolution(matched_pandora, True, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid)
    mean_mass, var_mass, distr_mass, mass_true, _, _, E_over_true = calculate_event_mass_resolution(matched_, False, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid)

    (
        mean_energy_over_true,
        var_energy_over_true,
        _,
        _,
    ) = get_sigma_gaussian(E_over_true, np.linspace(0, 4, 300))
    (
        mean_energy_over_true_pandora,
        var_energy_over_true_pandora,
        _,
        _,
    ) = get_sigma_gaussian(E_over_true_pandora, np.linspace(0, 4, 300))
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
    dic["mass_over_true_model"] = distr_mass
    dic["mass_over_true_pandora"] = distr_mass_p
    dic["mass_model"] = distr_mass * mass_true
    dic["mass_pandora"] = distr_mass_p * mass_true_p
    dic["mean_mass_model"] = mean_mass
    dic["mean_mass_pandora"] = mean_mass_p
    dic["var_mass_model"] = var_mass
    dic["var_mass_pandora"] = var_mass_p
    dic["energy_over_true"] = E_over_true
    dic["energy_over_true_pandora"] = E_over_true_pandora
    dic["mean_energy_over_true"] = mean_energy_over_true
    dic["mean_energy_over_true_pandora"] = mean_energy_over_true_pandora
    dic["var_energy_over_true"] = var_energy_over_true
    dic["var_energy_over_true_pandora"] = var_energy_over_true_pandora
    return dic


def plot_mass_resolution(event_res_dic, PATH_store):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlabel(r"$M_{pred}/M_{true}$")
    bins = np.linspace(0, 3, 100)
    ax.hist(
        event_res_dic["mass_over_true_model"],
        bins=bins,
        histtype="step",
        label="ML $\mu$={} $\sigma/\mu$={}".format(
            round((event_res_dic["mean_mass_model"]), 2),
            round((event_res_dic["var_mass_model"]), 2),
        ),
        color="red",
        density=True,
    )
    ax.hist(
        event_res_dic["mass_over_true_pandora"],
        bins=bins,
        histtype="step",
        label="Pandora $\mu$={} $\sigma/\mu$={}".format(
            round((event_res_dic["mean_mass_pandora"]), 2),
            round((event_res_dic["var_mass_pandora"]), 2),
        ),
        color="blue",
        density=True,
    )
    ax.grid()
    ax.legend()
    #ax.set_xlim([0, 10])
    fig.tight_layout()
    print("Saving mass resolution")
    import os
    fig.savefig(os.path.join(PATH_store, "mass_resolution.pdf"), bbox_inches="tight")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlabel(r"$M_{reco}$")
    bins = np.linspace(0, 3, 100)
    ax.hist(
        event_res_dic["mass_model"],
        bins=bins,
        histtype="step",
        label="ML",
        color="red",
        density=True,
    )
    ax.hist(
        event_res_dic["mass_pandora"],
        bins=bins,
        histtype="step",
        label="Pandora",
        color="blue",
        density=True,
    )
    ax.grid()
    ax.legend()
    #ax.set_xlim([0, 10])
    fig.tight_layout()
    fig.savefig(os.path.join(PATH_store, "mass_reco_absolute.pdf"), bbox_inches="tight")

