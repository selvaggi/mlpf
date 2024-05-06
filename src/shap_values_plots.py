import matplotlib

matplotlib.rc("font", size=35)

import shap
import matplotlib.pyplot as plt
import mplhep as hep
import pickle
import numpy as np
import PIL
from pdf2image import convert_from_path
from PIL import Image

def pdf_to_pil(pdf_path, page=0):
    # Convert the PDF to a list of PIL images
    images = convert_from_path(pdf_path, first_page=page + 1, last_page=page + 1)

    # Return the first image (assuming only one page)
    return images[0]



hep.style.use("CMS")
colors_list = ["#deebf7", "#9ecae1", "#3182bd"]  # color list Jan


'''input_filename = "/eos/user/g/gkrzmanc/eval_plots_EC/eval_EC_shap_testing/shap_vals.pkl"
shap_vals = torch.load(open(input_filename, "rb"))
feature_names =  ["ecal_E", "hcal_E", "num_hits", "track_p", "ecal_dispersion", "hcal_dispersion", "sum_e", "num_tracks", "track_p_chis", "hit_x_avg", "hit_y_avg", "hit_z_avg", "eta", "phi"]
feature_names += ["GAT" + str(i) for i in range(32)]

# make a plot of the shap values - for each feature, plot the mean and std of the shap values
fig, ax = plt.subplots(figsize=(15, 25))
shap_vals_mean = np.mean(shap_vals, axis=0)
shap_vals_std = np.std(shap_vals, axis=0)
ax.barh(feature_names, shap_vals_mean, xerr=shap_vals_std, color=colors_list[0])
ax.set_xlabel("SHAP value")
ax.set_ylabel("Feature")
fig.show()'''

#path = "/eos/user/g/gkrzmanc/eval_plots_EC/eval_EC_shap_df/showers_df_evaluation/0_0_None_hdbscan.pt"
path = "/eos/user/g/gkrzmanc/2024/eval_dnn_3004_l1_training_longereval/showers_df_evaluation/0_0_None_hdbscan.pt" # clusters 30.4., eval. of the 29.4. GNN GAT run, with 40 eval files
df = pickle.load(open(path, "rb"))
print(len(df))
feature_names =  ["ecal_E", "hcal_E", "num_hits", "track_p", "ecal_dispersion", "hcal_dispersion", "sum_e", "num_tracks", "track_p_chis", "hit_x_avg", "hit_y_avg", "hit_z_avg", "eta", "phi"]
feature_names += ["GAT" + str(i) for i in range(32)]
PIDs = list(df.pid.unique())
print(PIDs)
energy_ranges = [0, 6, 12, 18, 24, 30, 36, 42, 48]
print(df.columns)

'''regenerate_plots = False
if regenerate_plots:
    for pid in PIDs:
        for i in range(len(energy_ranges) - 1):
            df_energy_range = df[(df["true_showers_E"] > energy_ranges[i]) & (df["true_showers_E"] < energy_ranges[i + 1]) & (df["pid"] == pid)]
            if len(df_energy_range) > 0:
                features = np.stack([np.array(x) for x in df_energy_range.ec_x.values])
                shap_vals = np.stack([np.array(x) for x in df_energy_range.shap_values.values])
                shap.summary_plot(shap_vals, features, show=False, feature_names=feature_names)
                f = plt.gcf()
                f.suptitle(f"PID {int(pid)}, E in [{energy_ranges[i]}, {energy_ranges[i+1]}]")
                f.tight_layout()
                #f.show()
                f.savefig(f"/eos/user/g/gkrzmanc/eval_plots_EC/eval_EC_shap_df/showers_df_evaluation/shap_plots/pid_{int(pid)}_energy_range_{energy_ranges[i]}_{energy_ranges[i+1]}.png", dpi=500)
                plt.clf()

# Same as above, but with !!all!! plots inside one figure. For each PID, do a row of energy ranges.
fig, axs = plt.subplots(len(PIDs), len(energy_ranges) - 1, figsize=(50, 50))
for i, pid in enumerate(PIDs):
    for j in range(len(energy_ranges) - 1):
        df_energy_range = df[(df["true_showers_E"] > energy_ranges[j]) & (df["true_showers_E"] < energy_ranges[j + 1]) & (df["pid"] == pid)]
        if len(df_energy_range) > 0:
            #features = np.stack([np.array(x) for x in df_energy_range.ec_x.values])
            #shap_vals = np.stack([np.array(x) for x in df_energy_range.shap_values.values])
            filename = f"/eos/user/g/gkrzmanc/eval_plots_EC/eval_EC_shap_df/showers_df_evaluation/shap_plots/pid_{int(pid)}_energy_range_{energy_ranges[j]}_{energy_ranges[j+1]}.png"
            #shap_vals = pickle.load(open(filename, "rb"))
            #features = np.stack([np.array(x) for x in df_energy_range.ec_x.values])
            #shap.summary_plot(shap_vals, features, show=False, feature_names=feature_names, ax=axs[i][j])
            axs[i][j].set_title(f"PID {int(pid)}, E in [{energy_ranges[j]}, {energy_ranges[j+1]}]")
            axs[i][j].imshow(PIL.Image.open(filename))

fig.tight_layout()
fig.savefig("/eos/user/g/gkrzmanc/eval_plots_EC/eval_EC_shap_df/showers_df_evaluation/shap_plots/summary.pdf")
'''
# remove nan from PIDs
PIDs = [pid for pid in PIDs if not np.isnan(pid)]

'''# mean absolute SHAP value histograms, for each PID one row of energy ranges.
fig, axs = plt.subplots(len(PIDs), len(energy_ranges) - 1, figsize=(50, 50))
for i, pid in enumerate(PIDs):
    for j in range(len(energy_ranges) - 1):
        df_energy_range = df[(df["true_showers_E"] > energy_ranges[j]) & (df["true_showers_E"] < energy_ranges[j + 1]) & (df["pid"] == pid)]

        if len(df_energy_range) > 0:
            shap_vals = np.stack([np.array(x) for x in df_energy_range.shap_values.values])
            # remove nan items from shap_vals
            shap_vals = shap_vals[~np.isnan(shap_vals).any(axis=1)]

            shap_vals_mean = np.mean(np.abs(shap_vals), axis=0)
            shap_vals_std = np.std(np.abs(shap_vals), axis=0)
            print("pid", pid, "energy range", energy_ranges[j], energy_ranges[j+1], "mean", shap_vals_mean)
            # if is nan, print the shap_vals
            if np.isnan(shap_vals_mean).any():
                print(shap_vals)
                raise Exception
            axs[i][j].barh(feature_names, shap_vals_mean, color=colors_list[0], align="edge")
            # just display a giant number with len(df_energy_range)
            #axs[i][j].text(0.5, 0.5, f"{len(df_energy_range)}", fontsize=35, ha='center', va='center')
            axs[i][j].set_title(f"PID {int(pid)}, E in [{energy_ranges[j]}, {energy_ranges[j+1]}]")
fig.tight_layout()
fig.savefig("/eos/user/g/gkrzmanc/eval_plots_EC/eval_EC_shap_df/showers_df_evaluation/shap_plots/summary_mean_abs.pdf")
'''

# now plot just all shap vals for all pids averaged
shap_vals = np.stack([np.array(x) for x in df.shap_values.values])
shap_vals = shap_vals[~np.isnan(shap_vals).any(axis=1)]
shap_vals_mean = np.mean(np.abs(shap_vals), axis=0)
shap_vals_std = np.std(np.abs(shap_vals), axis=0)
shap_vals_mean_gnn = shap_vals_mean
fig, ax = plt.subplots(figsize=(10,18))
ax.barh(feature_names, shap_vals_mean, color="blue")
ax.grid()
ax.set_xlabel("Mean absolute SHAP value (feature importance)")
ax.set_xscale("log")
fig.tight_layout()
fig.show()
fig.savefig("/eos/user/g/gkrzmanc/2024/eval_dnn_3004_l1_training_longereval/showers_df_evaluation/shap_plots/summary_mean_abs_log.pdf")

###################

#path = "/eos/user/g/gkrzmanc/eval_plots_EC/eval_EC_shap_df/showersc_df_evaluation/0_0_None_hdbscan.pt"
path = "/eos/user/g/gkrzmanc/2024/eval_gnn_3004_l1_training/showers_df_evaluation/0_0_None_hdbscan.pt" # clusters 30.4., eval. of the 29.4. GNN GAT run, with 40 eval files
df = pickle.load(open(path, "rb"))
print(len(df))
feature_names =  ["ecal_E", "hcal_E", "num_hits", "track_p", "ecal_dispersion", "hcal_dispersion", "sum_e", "num_tracks", "track_p_chis", "hit_x_avg", "hit_y_avg", "hit_z_avg", "eta", "phi"]
feature_names += ["GAT" + str(i) for i in range(32)]
PIDs = list(df.pid.unique())
print(PIDs)
energy_ranges = [0, 6, 12, 18, 24, 30, 36, 42, 48]
print(df.columns)

PIDs = [pid for pid in PIDs if not np.isnan(pid)]
# now plot just all shap vals for all pids averaged
shap_vals = np.stack([np.array(x) for x in df.shap_values.values])
shap_vals = shap_vals[~np.isnan(shap_vals).any(axis=1)]
shap_vals_mean = np.mean(np.abs(shap_vals), axis=0)
shap_vals_std = np.std(np.abs(shap_vals), axis=0)
shap_vals_mean_gnn = shap_vals_mean
fig, ax = plt.subplots(figsize=(10,18))
ax.barh(feature_names, shap_vals_mean, color="blue")
ax.grid()
ax.set_xlabel("Mean absolute SHAP value (feature importance)")
ax.set_xscale("log")
fig.tight_layout()
fig.show()
fig.savefig("/eos/user/g/gkrzmanc/2024/eval_gnn_3004_l1_training/showers_df_evaluation/shap_plots/summary_mean_abs_log.pdf")

###################

