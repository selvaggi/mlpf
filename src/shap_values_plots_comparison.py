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

path = "/eos/user/g/gkrzmanc/2024/eval_dnn_3004_l1_training_longereval/showers_df_evaluation/0_0_None_hdbscan.pt" # clusters 30.4., eval. of the 29.4. GNN GAT run, with 40 eval files
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
shap_vals_mean_dnn = np.mean(np.abs(shap_vals), axis=0)

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
shap_vals_mean_gnn = np.mean(np.abs(shap_vals), axis=0)
shap_vals_std = np.std(np.abs(shap_vals), axis=0)

fig, ax = plt.subplots(figsize=(12, 25))

# plot both GNN and DNN
ind = np.arange(len(feature_names))
width = 0.4

fig, ax = plt.subplots(figsize=(12, 15))
ax.barh(ind, shap_vals_mean_gnn, width, color='blue', label='DNN+GNN')
ax.barh(ind + width, shap_vals_mean_dnn, width, color='red', label='DNN')

ax.set(yticks=ind + width, yticklabels=feature_names, ylim=[2*width - 1, len(feature_names)])
ax.set_xlabel("Mean absolute SHAP value (feature importance)")
ax.grid()
ax.set_xscale("log")
ax.legend()
fig.tight_layout()
fig.show()
fig.savefig("/eos/user/g/gkrzmanc/2024/eval_gnn_3004_l1_training/showers_df_evaluation/shap_plots/summary_comparison.pdf")

# TODO: as above, but per PID and/or energy range
