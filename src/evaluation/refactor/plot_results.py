
import matplotlib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.utils.inference.per_particle_metrics import plot_per_energy_resolution, reco_hist, \
    plot_mass_contribution_per_category, plot_mass_contribution_per_PID
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
# import mplhep as hep
from src.utils.inference.pandas_helpers import open_hgcal, open_mlpf_dataframe
from src.utils.inference.per_particle_metrics import (
    plot_per_energy_resolution2_multiple, plot_confusion_matrix, plot_confusion_matrix_pandora,
    plot_efficiency_all, calc_unit_circle_dist, plot_per_energy_resolution2, analyze_fakes, analyze_fakes_PID,
    plot_cm_per_energy, plot_fake_and_missed_energy_regions, quick_plot_mass,
    plot_cm_per_energy_on_overview
)
from src.utils.inference.track_cluster_eff_plots import plot_track_assignation_eval
from src.utils.inference.event_Ks import get_decay_type
import matplotlib.pyplot as plt
import torch
import pickle
from src.evaluation.refactor.preprocess import preprocess_dataframe, renumber_batch_idx
fs = 10
font = {'size': fs}
matplotlib.rc('font', **font)
# hep.style.use("CMS")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,
                    help="Path to the folder with the training in which checkpoints are saved",
                    default="/eos/home-g/gkrzmanc/results/2024/eval_clustering_plus_model_epoch4_Hss_300files")
parser.add_argument("--preprocess", type=str, help="Comma-separated list of scripts to apply",
                    default="")
parser.add_argument("--output_dir", type=str, default="",
                    help="Output directory (just the name of the folder, nested under the input path")
parser.add_argument("--mass-only", action="store_true", help="Only quickly plot mass in the energy resolution plots")
# parser.add_argument("--exclude-gt-clusters") # TODO: implement

args = parser.parse_args()
print("Preprocess:", args.preprocess)
PATH_store = os.path.join(args.path, args.output_dir)
if not os.path.exists(PATH_store):
    os.makedirs(PATH_store)
import sys
class Logger(object):
    def __init__(self, filename='log.txt'):
        self.terminal = sys.stdout
        self.log = open(os.path.join(PATH_store, filename), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
sys.stdout = Logger()
sys.stderr = Logger("err.txt")

PATH_store_individual_plots = os.path.join(PATH_store, "individual_plots")
PATH_store_summary_plots = os.path.join(PATH_store, "summary_plots")
if not os.path.exists(PATH_store_individual_plots):
    os.makedirs(PATH_store_individual_plots)
if not os.path.exists(PATH_store_summary_plots):
    os.makedirs(PATH_store_summary_plots)
path_ML = "showers_df_evaluation/0_0_None_hdbscan_gun_drlog_v9_dr01_4_57500_hdbscan_Hss_400_8_8_01.pt"
path_pandora = "showers_df_evaluation/0_0_None_pandora_gun_drlog_v9_dr01_4_57500_hdbscan_Hss_400_8_8_01.pt"
dir_top = args.path
print(PATH_store)
path_hgcal = os.path.join(dir_top, path_ML)
#path_hgcal_GTC = os.path.join(dir_top, path_GT_clusters)
sd_hgb, _ = open_mlpf_dataframe(path_hgcal, False)
sd_pandora, _ = open_mlpf_dataframe(os.path.join(dir_top, path_pandora), False)
sd_hgb, sd_pandora = preprocess_dataframe(sd_hgb, sd_pandora, args.preprocess.split(","))

#sd_hgb_gt = open_mlpf_dataframe(path_hgcal_GTC, False)
'''
ch = sd_hgb[sd_hgb.pred_pid_matched == 1]
ch_le = ch[ch.calibrated_E < 5.0]
cluster_E = ch_le.pred_showers_E
track_E = ch_le.calibrated_E
fakes_mask = pd.isna(ch_le.pid)
dist_trk = np.linalg.norm(np.stack(ch_le.pred_ref_pt_matched.values), axis=1)
ch_le_filter = ch_le[dist_trk<0.21

fig, ax = plt.subplots()
diff_E_truth = ch_le.true_showers_E - track_E
diff_E_truth_filter = ch_le_filter.true_showers_E - ch_le_filter.calibrated_E
bins = np.linspace(-7, 7, 100)
ax.hist(diff_E_truth, bins=bins, histtype="step", label="nofilter")
ax.hist(diff_E_truth_filter, bins=bins, histtype="step", label="filter")
ax.legend()
ax.set_yscale("log")
fig.show()



fig, ax = plt.subplots()
diff_E = cluster_E - track_E
bins = np.linspace(-7, 7, 100)
ax.hist(diff_E[fakes_mask], bins=bins, histtype="step", label="Fakes")
ax.hist(diff_E[~fakes_mask], bins=bins, histtype="step", label="Matched")
ax.legend()
ax.set_yscale("log")
fig.show()

fig, ax = plt.subplots()
bins = np.linspace(0, 2, 100)
ax.hist(dist_trk[ch_le.is_track_correct>=1.0], bins=bins, histtype="step", label="Is track correct")
ax.hist(dist_trk[ch_le.is_track_correct==0.0], bins=bins, histtype="step", label="Track not correct")
ax.legend()
ax.set_yscale("log")
fig.show()


diff_E_truth = ch_le.true_showers_E - track_E
diff_E_hits_truth = ch_le.true_showers_E - cluster_E

fig, ax = plt.subplots()
bins = np.linspace(-7, 7, 100)
ax.hist(diff_E_truth[ch_le.is_track_correct>=1.0], bins=bins, histtype="step", label="Is track correct >= 1")
ax.hist(diff_E_truth[ch_le.is_track_correct==0.0], bins=bins, histtype="step", label="Track not correct")
ax.legend()
ax.set_yscale("log")
ax.set_xlabel("$E_{true}-E_{pred}$")
fig.show()

bad_energy = diff_E_truth.abs() > 1.25
print("Frac bad energy", bad_energy.sum() / len(bad_energy))
fig, ax = plt.subplots()
bins=np.linspace(-5, 5, 100)
diff_E_truth_1 = ch_le.true_showers_E - track_E
diff_E_truth_1[bad_energy] = diff_E_hits_truth[bad_energy]
ax.hist(diff_E_truth_1, bins=bins, histtype="step", label="Correct")
ax.hist(diff_E_truth, bins=bins, histtype="step", label="No correct")
#ax.hist(diff_E_truth, bins=bins, histtype="step")
#ax.hist(diff_E_hits_truth, bins=bins, histtype="step")
ax.legend()
ax.set_yscale("log")
fig.show()

'''
#analyze_fakes(sd_pandora, sd_hgb, PATH_store_individual_plots)
analyze_fakes_PID(sd_pandora, sd_hgb, PATH_store_individual_plots)
plot_track_assignation_eval(sd_hgb, sd_pandora, PATH_store_summary_plots)

if args.mass_only:
    quick_plot_mass(sd_hgb, sd_pandora, PATH_store_summary_plots)
    sys.exit(0)

plot_mass_contribution_per_category(sd_hgb, sd_pandora, PATH_store_summary_plots)
plot_mass_contribution_per_category(sd_hgb, sd_pandora, PATH_store_summary_plots, energy_bins=[0, 1])
plot_mass_contribution_per_category(sd_hgb, sd_pandora, PATH_store_summary_plots, energy_bins=[1, 10])
plot_mass_contribution_per_category(sd_hgb, sd_pandora, PATH_store_summary_plots, energy_bins=[10, 100])
plot_mass_contribution_per_PID(sd_hgb, sd_pandora, PATH_store_summary_plots)
plot_fake_and_missed_energy_regions(sd_pandora, sd_hgb, PATH_store_summary_plots)
pandora_vertex = np.array(sd_pandora.vertex.values.tolist())

# Filter the df based on where decay type is 0
#plot_track_assignation_eval(sd_hgb, sd_pandora, PATH_store_individual_plots)
fig, ax = plt.subplots(4, 8, figsize=(28, 28 * 4 / 8))  # The overview figure of efficiencies #
fig_eff, ax_eff = plt.subplots(4, 4, figsize=(14, 14))
plot_cm_per_energy(sd_hgb, sd_pandora, PATH_store_summary_plots, PATH_store_individual_plots)
plot_efficiency_all(sd_pandora, [sd_hgb], PATH_store_individual_plots, ["ML"], ax=ax)
plot_efficiency_all(sd_pandora, [sd_hgb], PATH_store_individual_plots, ["ML"], ax=ax_eff)

plot_cm_per_energy_on_overview(sd_hgb, sd_pandora, PATH_store_individual_plots, ax=ax[:, 4:6])
reco_hist(sd_hgb, sd_pandora, PATH_store_individual_plots)
column_cm_full = 6
column_cm_full_p = 7
plot_confusion_matrix(sd_hgb, PATH_store_individual_plots, ax=ax[0, column_cm_full], ax1=ax[1, column_cm_full], ax2=ax[2, column_cm_full])
plot_confusion_matrix(sd_hgb, PATH_store_individual_plots, add_pie_charts=True, ax=ax[3, column_cm_full])
plot_confusion_matrix_pandora(sd_pandora, PATH_store_individual_plots, ax=ax[0, column_cm_full_p], ax1=ax[1, column_cm_full_p], ax2=ax[2, column_cm_full_p])
plot_confusion_matrix_pandora(sd_pandora, PATH_store_individual_plots, add_pie_charts=True, ax=ax[3, column_cm_full_p])
x_positions = [4 / 8, 6 / 8]
for x_position in x_positions:
    fig.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing if necessary
    fig.add_artist(plt.Line2D([x_position, x_position], [0, 1], color="black", linewidth=2, transform=fig.transFigure))
fig.tight_layout()
fig_eff.tight_layout()
fig_eff.savefig(os.path.join(PATH_store_summary_plots, "overview_Efficiency_FakeRate.pdf"))
fig.savefig(os.path.join(PATH_store_summary_plots, "overview_Efficiency_FakeRate_ConfusionMatrix.pdf"))
x = sd_hgb.pred_ref_pt_matched[sd_hgb.is_track_in_cluster==1].values
x = np.stack(x)
x = np.linalg.norm(x, axis=1)

e_ranges = [[0, 5], [5, 15], [15, 50]]

current_dir = PATH_store_individual_plots
current_dir_detailed = PATH_store_summary_plots
if not os.path.exists(current_dir):
    os.makedirs(current_dir)
if not os.path.exists(current_dir_detailed):
    os.makedirs(current_dir_detailed)

plot_per_energy_resolution2_multiple(
    sd_pandora,
    {"ML": sd_hgb},
    current_dir,
    tracks=True,
    perfect_pid=False,
    mass_zero=False,
    ML_pid=True,
    PATH_store_detailed_plots=current_dir_detailed
)

print("Done plotting")

def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di
