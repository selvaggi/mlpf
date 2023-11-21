import gzip
import pickle
import mplhep as hep

hep.style.use("CMS")
import matplotlib

matplotlib.rc("font", size=25)
import numpy as np
import pandas as pd

from evaluation_plots import obtain_metrics

# "/eos/user/m/mgarciam/datasets_mlpf/models_trained/logs_10_15_allp_karolina1/large_eval/analysis/out.bin.gz",
# /eos/user/m/mgarciam/datasets_mlpf/models_trained/logs_10_15_allp_karolina/training_evaluation_test2309/analysis/out_matchedshowers.bin.gz
def main():
    with gzip.open(
        "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/hgcal/logs_1015_1911/pandora/analysis/out.bin.gz",
        "rb",
    ) as f:
        data = pickle.load(f)
    sd = data["showers_dataframe"]
    matched = sd.dropna()
    ms = data["matched_showers"]
    print(ms.head())
    dict_1 = obtain_metrics_hgcal(sd, matched, ms)

    dic2 = True
    if dic2:
        data = pd.read_pickle(
            "/eos/user/m/mgarciam/datasets_mlpf/models_trained/2309/mlpf/mlpf_v3/showers_df_evaluation/0_0_None_pandora.pt"
        )
        sd = data
        matched = sd.dropna()
        dict_2 = obtain_metrics(sd, matched)

        # with gzip.open(
        # "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/hgcal/logs_1015_1911_reduced_100_v0/pandora/analysis_logs_1015_1911_reduced_100/out.bin.gz",
        # "rb",
        # ) as f:
        #     data = pickle.load(f)
        # sd = data["showers_dataframe"]
        # matched = sd.dropna()
        # ms = data["matched_showers"]
        # print(ms.head())
        # dict_3 = obtain_metrics_hgcal(sd, matched, ms)
        data = pd.read_pickle(
            "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/mlpf/mlpf_all_energies_hgcal_loss/showers_df_evaluation/0_0_None.pt"
        )
        sd = data
        matched = sd.dropna()
        dict_3 = obtain_metrics(sd, matched)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 2, figsize=(18, 12))
    # efficiency plot
    ax[0, 0].scatter(
        dict_1["energy_eff"], dict_1["eff"], facecolors="none", edgecolors="b"
    )
    if dic2:
        ax[0, 0].scatter(
            dict_2["energy_eff"], dict_2["eff"], facecolors="none", edgecolors="r"
        )
        ax[0, 0].scatter(
            dict_3["energy_eff"], dict_3["eff"], facecolors="none", edgecolors="g"
        )
    ax[0, 0].set_xlabel("True Energy [GeV]")
    ax[0, 0].set_ylabel("Efficiency")
    ax[0, 0].grid()

    # fake rates
    ax[0, 1].scatter(
        dict_1["energy_fakes"], dict_1["fake_rate"], facecolors="none", edgecolors="b"
    )
    if dic2:
        ax[0, 1].scatter(
            dict_2["energy_fakes"],
            dict_2["fake_rate"],
            facecolors="none",
            edgecolors="r",
        )
        ax[0, 1].scatter(
            dict_3["energy_fakes"],
            dict_3["fake_rate"],
            facecolors="none",
            edgecolors="g",
        )
    ax[0, 1].set_xlabel("Reconstructed Energy [GeV]")
    ax[0, 1].set_ylabel("Fake rate")
    ax[0, 1].grid()
    ax[0, 1].set_yscale("log")

    # resolution
    ax[1, 0].scatter(
        dict_1["energy_resolutions"],
        dict_1["mean_true_rec"],
        facecolors="none",
        edgecolors="b",
    )
    if dic2:
        ax[1, 0].scatter(
            dict_2["energy_resolutions"],
            dict_2["mean_true_rec"],
            facecolors="none",
            edgecolors="r",
        )
        ax[1, 0].scatter(
            dict_3["energy_resolutions"],
            dict_3["mean_true_rec"],
            facecolors="none",
            edgecolors="g",
        )
    ax[1, 0].set_xlabel("Reco Energy [GeV]")
    ax[1, 0].set_ylabel("Response")
    ax[1, 0].grid()

    # response
    ax[1, 1].scatter(
        dict_1["energy_resolutions"],
        dict_1["variance_om_true_rec"],
        facecolors="none",
        edgecolors="b",
    )
    if dic2:
        ax[1, 1].scatter(
            dict_2["energy_resolutions"],
            dict_2["variance_om_true_rec"],
            facecolors="none",
            edgecolors="r",
        )
        ax[1, 1].scatter(
            dict_3["energy_resolutions"],
            dict_3["variance_om_true_rec"],
            facecolors="none",
            edgecolors="g",
        )
    ax[1, 1].set_xlabel("Reco Energy [GeV]")
    ax[1, 1].set_ylabel("Resolution")
    ax[1, 1].grid()

    # purity
    ax[2, 0].errorbar(
        np.array(dict_1["energy_ms"]),
        np.array(dict_1["fce_energy"]),
        np.array(dict_1["fce_var_energy"]),
        marker="o",
        mec="blue",
        ms=5,
        mew=4,
        linestyle="",
    )
    if dic2:
        ax[2, 0].errorbar(
            np.array(dict_2["energy_ms"]),
            np.array(dict_2["fce_energy"]),
            np.array(dict_2["fce_var_energy"]),
            marker="o",
            mec="red",
            ms=5,
            mew=4,
            linestyle="",
        )
        ax[2, 0].errorbar(
            np.array(dict_3["energy_ms"]),
            np.array(dict_3["fce_energy"]),
            np.array(dict_3["fce_var_energy"]),
            marker="o",
            mec="green",
            ms=5,
            mew=4,
            linestyle="",
        )

    ax[2, 0].set_xlabel("Reco Energy [GeV]")
    ax[2, 0].set_ylabel("Containment")
    ax[2, 0].grid()

    ax[2, 1].errorbar(
        np.array(dict_1["energy_ms"]),
        np.array(dict_1["purity_energy"]),
        np.array(dict_1["purity_var_energy"]),
        marker=".",
        mec="blue",
        ms=5,
        mew=4,
        linestyle="",
    )
    if dic2:
        ax[2, 1].errorbar(
            np.array(dict_2["energy_ms"]),
            np.array(dict_2["purity_energy"]),
            np.array(dict_2["purity_var_energy"]),
            marker=".",
            mec="red",
            ms=5,
            mew=4,
            linestyle="",
        )
        ax[2, 1].errorbar(
            np.array(dict_3["energy_ms"]),
            np.array(dict_3["purity_energy"]),
            np.array(dict_3["purity_var_energy"]),
            marker=".",
            mec="green",
            ms=5,
            mew=4,
            linestyle="",
        )

    ax[2, 1].set_xlabel("Reco Energy [GeV]")
    ax[2, 1].set_ylabel("Purity")
    ax[2, 1].grid()

    fig.savefig(
        "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/testeq_rec.png",
        bbox_inches="tight",
    )

    # fig = plt.figure(figsize=(18 / 2, 7))
    # # efficiency plot
    # plt.scatter(dict_1["energy_eff"], dict_1["eff"])
    # # ax[0, 0].scatter(dict_1['energy_eff_eq'], dict_1['eff_eq'])
    # plt.xlabel("True Energy [GeV]")
    # plt.ylabel("Efficiency")
    # plt.grid()
    # fig.savefig(
    #     "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/Efficiency.png",
    #     bbox_inches="tight",
    # )

    # fig = plt.figure(figsize=(18 / 2, 7))
    # plt.scatter(dict_1["energy_fakes"], dict_1["fake_rate"])
    # # ax[0, 1].scatter(dict_1['energy_fakes_eq'], dict_1['fake_rate_eq'])
    # plt.xlabel("Reconstructed Energy [GeV]")
    # plt.ylabel("Fake rate")
    # plt.grid()
    # plt.yscale("log")
    # fig.savefig(
    #     "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/Fake.png",
    #     bbox_inches="tight",
    # )

    # fig = plt.figure(figsize=(18 / 2, 7))
    # plt.scatter(dict_1["energy_resolutions"], dict_1["mean_true_rec"])
    # # ax[1, 0].scatter(dict_1['energy_eq'], dict_1['mean_true_rec_eq'])
    # plt.xlabel("Reconstructed Energy [GeV]")
    # plt.ylabel("Response")
    # plt.grid()
    # fig.savefig(
    #     "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/Response.png",
    #     bbox_inches="tight",
    # )

    # fig = plt.figure(figsize=(18 / 2, 7))
    # plt.scatter(dict_1["energy_resolutions"], dict_1["variance_om_true_rec"])
    # # ax[1, 1].scatter(dict_1['energy_eq'], dict_1['variance_om_true_rec_eq'])
    # plt.xlabel("Reconstructed Energy [GeV]")
    # plt.ylabel("Resolution")
    # plt.grid()

    # fig.savefig(
    #     "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/Resolution.png",
    #     bbox_inches="tight",
    # )

    # fig = plt.figure(figsize=(18 / 2, 7))
    # plt.errorbar(
    #     np.array(dict_1["energy_ms"]),
    #     np.array(dict_1["fce_energy"]),
    #     np.array(dict_1["fce_var_energy"]),
    #     marker=".",
    #     mec="blue",
    #     ms=5,
    #     mew=4,
    #     linestyle="",
    # )

    # plt.xlabel("Reconstructed Energy [GeV]")
    # plt.ylabel("Containing")
    # plt.grid()
    # fig.savefig(
    #     "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/Containing.png",
    #     bbox_inches="tight",
    # )

    # fig = plt.figure(figsize=(18 / 2, 7))
    # plt.errorbar(
    #     np.array(dict_1["energy_ms"]),
    #     np.array(dict_1["purity_energy"]),
    #     np.array(dict_1["purity_var_energy"]),
    #     marker=".",
    #     mec="blue",
    #     ms=5,
    #     mew=4,
    #     linestyle="",
    # )

    # plt.xlabel("Reconstructed Energy [GeV]")
    # plt.ylabel("Purity")
    # plt.grid()

    # fig.savefig(
    #     "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/Purity.png",
    #     bbox_inches="tight",
    # )


def obtain_metrics_hgcal(sd, matched, ms):
    true_e = matched.truthHitAssignedEnergies
    bins = np.arange(0, 51, 2)
    eff = []
    fake_rate = []
    energy_eff = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = sd.truthHitAssignedEnergies.values <= bin_i1
        mask_below = sd.truthHitAssignedEnergies.values > bin_i
        mask = mask_below * mask_above
        number_of_non_reconstructed_showers = np.sum(
            np.isnan(sd.pred_energy_hits_raw.values)[mask]
        )
        total_showers = len(sd.t_rec_energy.values[mask])
        if total_showers > 0:
            eff.append(
                (total_showers - number_of_non_reconstructed_showers) / total_showers
            )
            energy_eff.append((bin_i1 + bin_i) / 2)
    # fake rate per energy with a binning of 1
    true_e = matched.truthHitAssignedEnergies
    bins_fakes = np.arange(0, 51, 2)
    fake_rate = []
    energy_fakes = []
    total_true_showers = np.sum(
        ~np.isnan(sd.truthHitAssignedEnergies.values)
    )  # the ones where truthHitAssignedEnergies is not nan
    for i in range(len(bins_fakes) - 1):
        bin_i = bins_fakes[i]
        bin_i1 = bins_fakes[i + 1]
        mask_above = sd.pred_energy_hits_raw.values <= bin_i1
        mask_below = sd.pred_energy_hits_raw.values > bin_i
        mask = mask_below * mask_above
        fakes = np.sum(np.isnan(sd.truthHitAssignedEnergies)[mask])
        total_showers = len(sd.pred_energy_hits_raw.values[mask])

        if total_showers > 0:
            # print(fakes, np.mean(sd.pred_energy_hits_raw[mask]))
            fake_rate.append((fakes) / total_true_showers)
            energy_fakes.append((bin_i1 + bin_i) / 2)

    # plot 2 for each energy bin calculate the mean and the variance of the distribution
    mean = []
    variance_om = []
    mean_true_rec = []
    variance_om_true_rec = []
    energy_resolutions = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = ms["e_truth"] <= bin_i1
        mask_below = ms["e_truth"] > bin_i
        mask = mask_below * mask_above
        pred_e = matched.pred_energy_hits_raw[mask]
        true_e = matched.truthHitAssignedEnergies[mask]
        true_rec = ms.e_truth[mask]

        if np.sum(mask) > 0:
            mean_predtotrue = np.mean(pred_e / true_e)
            mean_predtored = np.mean(pred_e / true_rec)
            var_predtotrue = np.var(pred_e / true_e) / mean_predtotrue
            variance_om_true_rec_ = np.var(pred_e / true_rec) / mean_predtored
            mean.append(mean_predtotrue)
            mean_true_rec.append(mean_predtored)
            variance_om.append(var_predtotrue)
            variance_om_true_rec.append(variance_om_true_rec_)
            energy_resolutions.append((bin_i1 + bin_i) / 2)

    bins = np.arange(0, 51, 2)
    fce_energy = []
    fce_var_energy = []
    energy_ms = []

    purity_energy = []
    purity_var_energy = []
    fce = ms["e_pred_and_truth"] / ms["e_truth"]
    purity = ms["e_pred_and_truth"] / ms["e_pred"]
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = ms["e_truth"] <= bin_i1
        mask_below = ms["e_truth"] > bin_i
        mask = mask_below * mask_above
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

    dict = {
        "energy_eff": energy_eff,
        "eff": eff,
        "energy_fakes": energy_fakes,
        "fake_rate": fake_rate,
        "mean_true_rec": mean,
        "variance_om_true_rec": variance_om,
        "fce_energy": fce_energy,
        "fce_var_energy": fce_var_energy,
        "energy_ms": energy_ms,
        "purity_energy": purity_energy,
        "purity_var_energy": purity_var_energy,
        "energy_resolutions": energy_resolutions,
    }
    return dict


if __name__ == "__main__":
    main()
