import gzip
import pickle
import mplhep as hep
import pandas as pd

hep.style.use("CMS")
import matplotlib

matplotlib.rc("font", size=25)
import numpy as np
import numpy as np



def main():

    neutrals_only = True
    # "/eos/user/m/mgarciam/datasets_mlpf/models_trained/logs_10_15_allp_karolina1/large_eval/analysis/out.bin.gz",
    # /eos/user/m/mgarciam/datasets_mlpf/models_trained/logs_10_15_allp_karolina/training_evaluation_test2309/analysis/out_matchedshowers.bin.gz
    data = pd.read_pickle(
        "/eos/user/m/mgarciam/datasets_mlpf/models_trained/mlpf_all_energies/showers_df_evaluation/0_0_None.pt"
    )
    if neutrals_only:
        sd = pd.concat([data[data["pid"] == 130], data[data["pid"] == 2112]])
    else:
        sd = data
    matched = sd.dropna()
    dict_1 = obtain_metrics(sd, matched)

    dic2 = True
    if dic2:
        data = pd.read_pickle(
            "/eos/user/m/mgarciam/datasets_mlpf/models_trained/mlpf_all_energies/showers_df_evaluation/0_0_None_pandora.pt"
        )
        if neutrals_only:
            sd = pd.concat([data[data["pid"] == 130], data[data["pid"] == 2112]])
        else:
            sd = data
        matched = sd.dropna()
        dict_2 = obtain_metrics(sd, matched)

    # with gzip.open(
    #     "/eos/user/m/mgarciam/datasets_mlpf/models_trained/2309_eq_3/training_evaluation_test2309/analysis/out.bin.gz",
    #     "rb",
    # ) as f:
    #     data2 = pickle.load(f)
    # sd_eq = data2["showers_dataframe"]
    # matched_eq = sd_eq.dropna()
    # (
    #     energy_eq,
    #     energy_eff_eq,
    #     eff_eq,
    #     energy_fakes_eq,
    #     fake_rate_eq,
    #     mean_true_rec_eq,
    #     variance_om_true_rec_eq,
    # ) = obtain_metrics(sd_eq, matched_eq)

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

    ax[2, 1].set_xlabel("Reco Energy [GeV]")
    ax[2, 1].set_ylabel("Purity")
    ax[2, 1].grid()
    if neutrals_only:
        fig.savefig(
            "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora/testeq_rec_neutrals.png",
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora/testeq_rec.png",
            bbox_inches="tight",
        )

    fig = plt.figure(figsize=(18 / 2, 7))
    # efficiency plot
    plt.scatter(dict_1["energy_eff"], dict_1["eff"])
    # ax[0, 0].scatter(dict_1['energy_eff_eq'], dict_1['eff_eq'])
    plt.xlabel("True Energy [GeV]")
    plt.ylabel("Efficiency")
    plt.grid()
    fig.savefig(
        "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora/Efficiency.png",
        bbox_inches="tight",
    )

    fig = plt.figure(figsize=(18 / 2, 7))
    plt.scatter(dict_1["energy_fakes"], dict_1["fake_rate"])
    # ax[0, 1].scatter(dict_1['energy_fakes_eq'], dict_1['fake_rate_eq'])
    plt.xlabel("Reconstructed Energy [GeV]")
    plt.ylabel("Fake rate")
    plt.grid()
    plt.yscale("log")
    fig.savefig(
        "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora/Fake.png",
        bbox_inches="tight",
    )

    fig = plt.figure(figsize=(18 / 2, 7))
    plt.scatter(dict_1["energy_resolutions"], dict_1["mean_true_rec"])
    # ax[1, 0].scatter(dict_1['energy_eq'], dict_1['mean_true_rec_eq'])
    plt.xlabel("Reconstructed Energy [GeV]")
    plt.ylabel("Response")
    plt.grid()
    fig.savefig(
        "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora/Response.png",
        bbox_inches="tight",
    )

    fig = plt.figure(figsize=(18 / 2, 7))
    plt.scatter(dict_1["energy_resolutions"], dict_1["variance_om_true_rec"])
    # ax[1, 1].scatter(dict_1['energy_eq'], dict_1['variance_om_true_rec_eq'])
    plt.xlabel("Reconstructed Energy [GeV]")
    plt.ylabel("Resolution")
    plt.grid()

    fig.savefig(
        "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora/Resolution.png",
        bbox_inches="tight",
    )

    fig = plt.figure(figsize=(18 / 2, 7))
    plt.errorbar(
        np.array(dict_1["energy_ms"]),
        np.array(dict_1["fce_energy"]),
        np.array(dict_1["fce_var_energy"]),
        marker=".",
        mec="blue",
        ms=5,
        mew=4,
        linestyle="",
    )

    plt.xlabel("Reconstructed Energy [GeV]")
    plt.ylabel("Containing")
    plt.grid()
    fig.savefig(
        "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora/Containing.png",
        bbox_inches="tight",
    )

    fig = plt.figure(figsize=(18 / 2, 7))
    plt.errorbar(
        np.array(dict_1["energy_ms"]),
        np.array(dict_1["purity_energy"]),
        np.array(dict_1["purity_var_energy"]),
        marker=".",
        mec="blue",
        ms=5,
        mew=4,
        linestyle="",
    )

    plt.xlabel("Reconstructed Energy [GeV]")
    plt.ylabel("Purity")
    plt.grid()

    fig.savefig(
        "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora/Purity.png",
        bbox_inches="tight",
    )
