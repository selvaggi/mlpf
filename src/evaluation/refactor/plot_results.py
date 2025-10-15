
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
from src.utils.inference.pandas_helpers import open_mlpf_dataframe, concat_with_batch_fix
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


dir_top = args.path

sd_hgb1, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/0_0_Nonetest_perfectC_all.pt"), False, False)
sd_hgb2, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/0_0_Nonetest_100_200_gt.pt"), False, False)
sd_hgb3, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/0_0_Nonetest_200_300_gt.pt"), False, False)
sd_hgb = concat_with_batch_fix([sd_hgb1, sd_hgb2, sd_hgb3])

sd_pandora1, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/pandora_3M.pt"), False, False)
sd_pandora2, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/0_0_None_pandora_100_200.pt"), False, False)
sd_pandora3, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/0_0_None_pandora_200_300.pt"), False, False)
sd_pandora = concat_with_batch_fix([sd_pandora1, sd_pandora2, sd_pandora3])

sd_hgb_gt1, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/0_0_Nonetest_perfectC_all.pt"), False, False)
sd_hgb_gt2, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/0_0_Nonetest_100_200_gt.pt"), False, False)
sd_hgb_gt3, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/0_0_Nonetest_200_300_gt.pt"), False, False)
sd_hgb_gt = concat_with_batch_fix([sd_hgb_gt1, sd_hgb_gt2, sd_hgb_gt3])

sd_hgb, sd_pandora = preprocess_dataframe(sd_hgb, sd_pandora, args.preprocess.split(","))

sd_hgb = sd_hgb[(np.abs(sd_hgb.reco_showers_E.values)>0.5)+np.isnan(sd_hgb.reco_showers_E.values)]
sd_pandora = sd_pandora[(np.abs(sd_pandora.reco_showers_E.values)>0.5)+np.isnan(sd_pandora.reco_showers_E.values)]
sd_hgb_gt= sd_hgb_gt[(np.abs(sd_hgb_gt.reco_showers_E.values)>0.5)]

# #analyze_fakes(sd_pandora, sd_hgb, PATH_store_individual_plots)
# # analyze_fakes_PID(sd_pandora, sd_hgb, PATH_store_individual_plots)

plot_track_assignation_eval(sd_hgb, sd_pandora, PATH_store_summary_plots)
# charged only 
if args.mass_only:
    quick_plot_mass(sd_hgb, sd_pandora, PATH_store_summary_plots)
    sys.exit(0)


plot_mass_contribution_per_category(sd_hgb, sd_pandora, sd_hgb_gt, PATH_store_summary_plots)
# plot_mass_contribution_per_category(sd_hgb, sd_pandora, PATH_store_summary_plots, energy_bins=[0, 1])
# plot_mass_contribution_per_category(sd_hgb, sd_pandora, PATH_store_summary_plots, energy_bins=[1, 10])
# plot_mass_contribution_per_category(sd_hgb, sd_pandora, PATH_store_summary_plots, energy_bins=[10, 100])
# plot_mass_contribution_per_PID(sd_hgb, sd_pandora, PATH_store_summary_plots)
# plot_fake_and_missed_energy_regions(sd_pandora, sd_hgb, PATH_store_summary_plots)
# pandora_vertex = np.array(sd_pandora.vertex.values.tolist())

# Filter the df based on where decay type is 0
plot_track_assignation_eval(sd_hgb, sd_pandora, PATH_store_individual_plots)
fig, ax = plt.subplots(4, 8, figsize=(28, 28 * 4 / 8))  # The overview figure of efficiencies #
fig_eff, ax_eff = plt.subplots(4, 4, figsize=(14, 14))
# plot_cm_per_energy(sd_hgb, sd_pandora, PATH_store_summary_plots, PATH_store_individual_plots)
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
# x = sd_hgb.pred_ref_pt_matched[sd_hgb.is_track_in_cluster==1].values
# x = np.stack(x)
# x = np.linalg.norm(x, axis=1)

e_ranges = [[0, 5], [5, 15], [15, 50]]

current_dir = PATH_store_individual_plots
current_dir_detailed = PATH_store_summary_plots
if not os.path.exists(current_dir):
    os.makedirs(current_dir)
if not os.path.exists(current_dir_detailed):
    os.makedirs(current_dir_detailed)
print("plot_per_energy_resolution2_multiple")
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

# # def save_dict(di_, filename_):
# #     with open(filename_, "wb") as f:
# #         pickle.dump(di_, f)

# # def load_dict(filename_):
# #     with open(filename_, "rb") as f:
# #         ret_di = pickle.load(f)
# #     return ret_di
