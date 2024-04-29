import gzip
import pickle
import matplotlib

matplotlib.rc("font", size=35)
import numpy as np
import pandas as pd
import os
import numpy as np

from utils.inference.event_metrics import plot_per_event_metrics
from utils.inference.per_particle_metrics import (
    plot_per_energy_resolution2,
    plot_efficiency_all,
)
import matplotlib.pyplot as plt
import mplhep as hep
import torch
import pickle

hep.style.use("CMS")
colors_list = ["#deebf7", "#9ecae1", "#3182bd"]  # color list Jan


input_filename = "/eos/user/g/gkrzmanc/eval_plots_EC/eval_EC_shap_testing/shap_vals.pkl"
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
fig.show()
