
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
    plot_per_energy_resolution2_multiple, plot_confusion_matrix, plot_confusion_matrix_pandora
    , calc_unit_circle_dist, plot_per_energy_resolution2, analyze_fakes, analyze_fakes_PID,
    plot_cm_per_energy, plot_fake_and_missed_energy_regions, quick_plot_mass,
    plot_cm_per_energy_on_overview
)
from src.utils.inference.efficiency_calc_and_plots import plot_efficiency_all
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

sd_hgb1, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/test_save_dpc0_0_None.pt"), False, False)
sd_hgb = sd_hgb1 #concat_with_batch_fix([sd_hgb1, sd_hgb2, sd_hgb3])

sd_pandora1, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/test_save_dpc0_0_None_pandora.pt"), False, False)
sd_pandora = sd_pandora1 #concat_with_batch_fix([sd_pandora1, sd_pandora2, sd_pandora3])

sd_hgb_gt1, _ = open_mlpf_dataframe(os.path.join(dir_top, "showers_df_evaluation/test_gun_um1000_0_None.pt"), False, False)
sd_hgb_gt = sd_hgb_gt1 #concat_with_batch_fix([sd_hgb_gt1, sd_hgb_gt2, sd_hgb_gt3])

sd_hgb, sd_pandora = preprocess_dataframe(sd_hgb, sd_pandora, args.preprocess.split(","))

mask = (sd_hgb.pred_pid_matched==4)*(sd_hgb.calibrated_E<1.5)
sd_hgb.loc[mask, "pred_pid_matched"]=1

# mask = (sd_hgb.is_track_in_cluster==1)
# sd_hgb.loc[mask, "calibrated_E"]=np.log(sd_hgb.loc[mask, "calibrated_E"])
# sd_hgb.loc[mask, "calibration_factor"]=np.log(sd_hgb.loc[mask, "calibration_factor"])

## cheat photon calibration
# mask_p = (sd_pandora.pid==22)+(sd_pandora.pid==130)+(sd_pandora.pid==2112)
# mask = (sd_hgb.pid==22)+ (sd_hgb.pid==130)+ (sd_hgb.pid==2112)
# sd_hgb.loc[mask, "calibrated_E"] = sd_hgb.loc[mask, "reco_showers_E"]
# # sd_hgb.loc[mask, "calibrated_E"] = sd_pandora.loc[mask_p, "pandora_calibrated_pfo"]

# check charged only
# mask = (np.abs(sd_hgb.pid)==11)+ (np.abs(sd_hgb.pid)==211)+ (np.abs(sd_hgb.pid)==13)+ (np.abs(sd_hgb.pid)==321)+ (np.abs(sd_hgb.pid)==2212)
# #print(len( sd_hgb[mask*(sd_hgb.is_track_in_MC==1)]), len( sd_hgb[mask*(sd_hgb.is_track_in_MC==0)]), len( sd_hgb[mask]))
# sd_hgb = sd_hgb[mask*(sd_hgb.is_track_in_MC==1)*(sd_hgb.gen_status==1)]
# mask = (np.abs(sd_pandora.pid)==11)+ (np.abs(sd_pandora.pid)==211)+ (np.abs(sd_pandora.pid)==13)+ (np.abs(sd_pandora.pid)==321)+ (np.abs(sd_pandora.pid)==2212)
# sd_pandora = sd_pandora[mask*(sd_pandora.is_track_in_MC==1)*(sd_pandora.gen_status==1)]
# mask = (np.abs(sd_hgb.pid)==130) + (np.abs(sd_hgb.pid)==22) + (np.abs(sd_hgb.pid)==2112)+ (np.abs(sd_hgb.pid)==3212)
# sd_hgb = sd_hgb[mask]
# mask = (np.abs(sd_pandora.pid)==130) + (np.abs(sd_pandora.pid)==22) + (np.abs(sd_pandora.pid)==2112)+ (np.abs(sd_pandora.pid)==3212)
# sd_pandora = sd_pandora[mask]
# mask = (sd_hgb.reco_showers_E ==0)*(sd_hgb.labels==0)
# sd_hgb.loc[mask, "calibrated_E"] = sd_hgb.loc[mask, "true_showers_E"]
# sd_hgb.loc[mask, "pred_pid_matched"] = sd_hgb.loc[mask, "pid_4_class_true"]
# sd_hgb.loc[mask, "pred_pos_matched"] = sd_hgb.loc[mask, "true_pos"]

# mask = (sd_pandora.reco_showers_E ==0)*(sd_pandora.labels==0)
# sd_pandora.loc[mask, "pandora_calibrated_pfo"] = sd_pandora.loc[mask, "true_showers_E"]
# sd_pandora.loc[mask, "pandora_pid"] = sd_pandora.loc[mask, "pid_4_class_true"]
# sd_pandora.loc[mask, "pandora_calibrated_pos"] = sd_pandora.loc[mask, "true_pos"]
# mask = (np.abs(sd_hgb1.pid)==211)
# sd_hgb1.loc[mask, "calibrated_E"] = sd_hgb1.loc[mask, "true_showers_E"]
# store_logit_max = []
# for i in range(len(sd_hgb)):
#     store_logit_max.append(np.max(sd_hgb.matched_extra_features.iloc[i][1:]))
# sd_hgb["logit_max"] =store_logit_max
# mask = (sd_hgb.logit_max<1)*(sd_hgb.pred_showers_E<1.5)
# sd_hgb = sd_hgb[~mask]
# sd_hgb = sd_hgb[(np.abs(sd_hgb.reco_showers_E.values)>0.5)+np.isnan(sd_hgb.reco_showers_E.values)]
# sd_pandora = sd_pandora[(np.abs(sd_pandora.reco_showers_E.values)>0.5)+np.isnan(sd_pandora.reco_showers_E.values)]
# sd_hgb_gt= sd_hgb_gt[(np.abs(sd_hgb_gt.reco_showers_E.values)>0.5)]

# #analyze_fakes(sd_pandora, sd_hgb, PATH_store_individual_plots)
# # analyze_fakes_PID(sd_pandora, sd_hgb, PATH_store_individual_plots)

# charged only 
if args.mass_only:
    dic_true_mass = np.load("/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/041225_arc_05/dic_true_gen_tracks_1.npy")
    quick_plot_mass(sd_hgb, sd_pandora, PATH_store_summary_plots, dic_true_mass)
    sys.exit(0)

# # _____________________________plot mass contribution_____________________________
# plot_mass_contribution_per_category(sd_hgb, sd_pandora, sd_hgb_gt, PATH_store_summary_plots)

# _____________________________plot track-cluster link eff _____________________________
plot_track_assignation_eval(sd_hgb, sd_pandora, PATH_store_individual_plots)

## _____________________________plot efficiency plots _____________________________
fig, ax = plt.subplots(5, 9, figsize=(10*9, 10 * 5 ))  # The overview figure of efficiencies #
fig_eff, ax_eff = plt.subplots(5, 5, figsize=(17.5, 24))
plot_cm_per_energy(sd_hgb, sd_pandora, PATH_store_summary_plots, PATH_store_individual_plots)
plot_cm_per_energy(sd_hgb, sd_pandora, PATH_store_summary_plots, PATH_store_individual_plots, status1=True)
plot_efficiency_all(sd_pandora, [sd_hgb], PATH_store_individual_plots, ["ML"], ax=ax)
plot_efficiency_all(sd_pandora, [sd_hgb], PATH_store_individual_plots, ["ML"], ax=ax_eff)
column_cm_full = 7
column_cm_full_p = 8
plot_cm_per_energy_on_overview(sd_hgb, sd_pandora, PATH_store_individual_plots, ax=ax[:, 5:7])
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

# # # _____________________________plot cm per enegy and reco hist _____________________________

reco_hist(sd_hgb, sd_pandora, PATH_store_individual_plots)


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

