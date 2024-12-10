
import matplotlib
import sys
#sys.path.append("/afs/cern.ch/work/m/mgarciam/private/mlpf/")
from src.utils.inference.per_particle_metrics import plot_per_energy_resolution, reco_hist, \
    plot_mass_contribution_per_category, plot_mass_contribution_per_PID
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import mplhep as hep
import os
from src.utils.inference.pandas_helpers import open_hgcal, open_mlpf_dataframe
from src.utils.inference.comparison_gt_mlclustering import plot_mass
import matplotlib.pyplot as plt
import torch
import pickle
from src.utils.inference.per_particle_metrics import (
    plot_per_energy_resolution2_multiple, plot_confusion_matrix, plot_confusion_matrix_pandora,
    plot_efficiency_all, calc_unit_circle_dist, plot_per_energy_resolution2, analyze_fakes,
    plot_cm_per_energy, plot_fake_and_missed_energy_regions, quick_plot_mass,
    plot_cm_per_energy_on_overview
)

from src.evaluation.refactor.preprocess import preprocess_dataframe
fs = 10
font = {'size': fs}
matplotlib.rc('font', **font)
hep.style.use("CMS")

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
if not os.path.exists(PATH_store_individual_plots):
    os.makedirs(PATH_store_individual_plots)
PATH_store_summary_plots = os.path.join(PATH_store, "summary_plots")
if not os.path.exists(PATH_store_summary_plots):
    os.makedirs(PATH_store_summary_plots)

path_ML_gt = "showers_df_evaluation/0_0_None_hdbscan.pt"
path_pandora = "showers_df_evaluation/0_0_None_pandora.pt"


path_ML = "showers_df_evaluation/0_0_None_hdbscan_delta_MC02.pt"
path_pandora = "showers_df_evaluation/0_0_None_pandora.pt"


dir_top = args.path
print(PATH_store)
path_ml = os.path.join(dir_top, path_ML)
path_ml_gt = os.path.join(dir_top, path_ML_gt)
sd_hgb, _ = open_mlpf_dataframe(path_ml, False)
sd_hgb_gt, _ = open_mlpf_dataframe(path_ml_gt, False)
sd_pandora, _ = open_mlpf_dataframe(os.path.join(dir_top, path_pandora), False)
sd_hgb, sd_pandora = preprocess_dataframe(sd_hgb, sd_pandora, args.preprocess.split(","))
sd_hgb_gt, sd_pandora = preprocess_dataframe(sd_hgb_gt, sd_pandora, args.preprocess.split(","))

current_dir = PATH_store_individual_plots
current_dir_detailed = PATH_store_summary_plots
if not os.path.exists(current_dir):
    os.makedirs(current_dir)
if not os.path.exists(current_dir_detailed):
    os.makedirs(current_dir_detailed)

fig_eff, ax_eff = plt.subplots(4, 4, figsize=(14, 14))

plot_efficiency_all(sd_pandora, [sd_hgb , sd_hgb_gt], PATH_store_individual_plots, ["ML", "ML GTC"], ax=ax_eff)
fig_eff.tight_layout()
fig_eff.savefig(os.path.join(PATH_store_summary_plots, "overview_Efficiency_FakeRate.pdf"))
plot_cm_per_energy(sd_hgb, sd_pandora, PATH_store_summary_plots, PATH_store_individual_plots, sd_hgb_gt=sd_hgb_gt)
plot_per_energy_resolution2_multiple(
    sd_pandora,
    {"ML": sd_hgb, "ML GTC": sd_hgb_gt},
    current_dir,
    tracks=True,
    perfect_pid=False,
    mass_zero=False,
    ML_pid=True,
    PATH_store_detailed_plots=current_dir_detailed
)


e_ranges = [[0, 5], [5, 15], [15, 50]]
current_dir = PATH_store_individual_plots
current_dir_detailed = PATH_store_summary_plots

if not os.path.exists(current_dir):
    os.makedirs(current_dir)
if not os.path.exists(current_dir_detailed):
    os.makedirs(current_dir_detailed)
plot_mass(sd_hgb, sd_hgb_gt, sd_pandora, PATH_store)
print("Done plotting")

