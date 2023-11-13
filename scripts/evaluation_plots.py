import gzip
import pickle
import mplhep as hep
import pandas as pd

hep.style.use("CMS")
import matplotlib

matplotlib.rc("font", size=25)
import numpy as np
import numpy as np


def obtain_metrics(sd, matched):
    bins = np.arange(0, 51, 2)
    eff = []
    fake_rate = []
    energy_eff = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = sd.true_showers_E.values <= bin_i1
        mask_below = sd.true_showers_E.values > bin_i
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
    true_e = matched.true_showers_E
    bins_fakes = np.arange(0, 51, 2)
    fake_rate = []
    energy_fakes = []
    total_true_showers = np.sum(
        ~np.isnan(sd.true_showers_E.values)
    )  # the ones where truthHitAssignedEnergies is not nan
    for i in range(len(bins_fakes) - 1):
        bin_i = bins_fakes[i]
        bin_i1 = bins_fakes[i + 1]
        mask_above = sd.pred_showers_E.values <= bin_i1
        mask_below = sd.pred_showers_E.values > bin_i
        mask = mask_below * mask_above
        fakes = np.sum(np.isnan(sd.true_showers_E)[mask])
        total_showers = len(sd.pred_showers_E.values[mask])

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
        mask_above = matched["true_showers_E"] <= bin_i1
        mask_below = matched["true_showers_E"] > bin_i
        mask = mask_below * mask_above
        pred_e = matched.pred_showers_E[mask]
        true_e = matched.true_showers_E[mask]
        true_rec = matched.true_showers_E[mask]

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
    fce = matched["e_pred_and_truth"] / matched["true_showers_E"]
    purity = matched["e_pred_and_truth"] / matched["pred_showers_E"]
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = matched["true_showers_E"] <= bin_i1
        mask_below = matched["true_showers_E"] > bin_i
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
        "mean_true_rec": mean_true_rec,
        "variance_om_true_rec": variance_om_true_rec,
        "fce_energy": fce_energy,
        "fce_var_energy": fce_var_energy,
        "energy_ms": energy_ms,
        "purity_energy": purity_energy,
        "purity_var_energy": purity_var_energy,
        "energy_resolutions": energy_resolutions,
    }
    return dict


# "/eos/user/m/mgarciam/datasets_mlpf/models_trained/logs_10_15_allp_karolina1/large_eval/analysis/out.bin.gz",
# /eos/user/m/mgarciam/datasets_mlpf/models_trained/logs_10_15_allp_karolina/training_evaluation_test2309/analysis/out_matchedshowers.bin.gz
data = pd.read_pickle("./dummy.pkl")
sd = data
matched = sd.dropna()
dict_1 = obtain_metrics(sd, matched)

dic2 = True
if dic2:
    data = pd.read_pickle("./dummy.pkl")
    sd = data["showers_dataframe"]
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
ax[0, 0].scatter(dict_1["energy_eff"], dict_1["eff"])
if dic2:
    ax[0, 0].scatter(dict_2["energy_eff"], dict_2["eff"])
ax[0, 0].set_xlabel("True Energy [GeV]")
ax[0, 0].set_ylabel("Efficiency")
ax[0, 0].grid()


# fake rates
ax[0, 1].scatter(dict_1["energy_fakes"], dict_1["fake_rate"])
if dic2:
    ax[0, 1].scatter(dict_2["energy_fakes"], dict_2["fake_rate"])
ax[0, 1].set_xlabel("Reconstructed Energy [GeV]")
ax[0, 1].set_ylabel("Fake rate")
ax[0, 1].grid()
ax[0, 1].set_yscale("log")


# resolution
ax[1, 0].scatter(dict_1["energy_resolutions"], dict_1["mean_true_rec"])
if dic2:
    ax[1, 0].scatter(dict_2["energy_resolutions"], dict_2["mean_true_rec"])
ax[1, 0].set_xlabel("Reco Energy [GeV]")
ax[1, 0].set_ylabel("Response")
ax[1, 0].grid()

# response
ax[1, 1].scatter(dict_1["energy_resolutions"], dict_1["variance_om_true_rec"])
if dic2:
    ax[1, 1].scatter(dict_2["energy_resolutions"], dict_2["variance_om_true_rec"])
ax[1, 1].set_xlabel("Reco Energy [GeV]")
ax[1, 1].set_ylabel("Resolution")
ax[1, 1].grid()

# purity
ax[2, 0].errorbar(
    np.array(dict_1["energy_ms"]),
    np.array(dict_1["fce_energy"]),
    np.array(dict_1["fce_var_energy"]),
    marker=".",
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
        marker=".",
        mec="blue",
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
        mec="blue",
        ms=5,
        mew=4,
        linestyle="",
    )

ax[2, 1].set_xlabel("Reco Energy [GeV]")
ax[2, 1].set_ylabel("Purity")
ax[2, 1].grid()
fig.savefig(
    "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/testeq_rec.png",
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
    "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Efficiency.png",
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
    "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Fake.png",
    bbox_inches="tight",
)

fig = plt.figure(figsize=(18 / 2, 7))
plt.scatter(dict_1["energy_resolutions"], dict_1["mean_true_rec"])
# ax[1, 0].scatter(dict_1['energy_eq'], dict_1['mean_true_rec_eq'])
plt.xlabel("Reconstructed Energy [GeV]")
plt.ylabel("Response")
plt.grid()
fig.savefig(
    "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Response.png",
    bbox_inches="tight",
)

fig = plt.figure(figsize=(18 / 2, 7))
plt.scatter(dict_1["energy_resolutions"], dict_1["variance_om_true_rec"])
# ax[1, 1].scatter(dict_1['energy_eq'], dict_1['variance_om_true_rec_eq'])
plt.xlabel("Reconstructed Energy [GeV]")
plt.ylabel("Resolution")
plt.grid()
fig.savefig(
    "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Resolution.png",
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
    "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Containing.png",
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
    "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Purity.png",
    bbox_inches="tight",
)
