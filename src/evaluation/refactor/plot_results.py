
import matplotlib
import sys
from src.utils.inference.per_particle_metrics import plot_per_energy_resolution, reco_hist
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import mplhep as hep
import os
from src.utils.inference.pandas_helpers import open_hgcal, open_mlpf_dataframe
from src.utils.inference.per_particle_metrics import (
    plot_per_energy_resolution2_multiple, plot_confusion_matrix, plot_confusion_matrix_pandora,
    plot_efficiency_all, calc_unit_circle_dist, plot_per_energy_resolution2, analyze_fakes, plot_cm_per_energy
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

args = parser.parse_args()
print("Preprocess:", args.preprocess)
PATH_store = os.path.join(args.path, args.output_dir)

import sys
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(PATH_store,"log.txt"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
sys.stdout = Logger()

if not os.path.exists(PATH_store):
    os.makedirs(PATH_store)
PATH_store_individual_plots = os.path.join(PATH_store, "individual_plots")
PATH_store_detailed_plots = os.path.join(PATH_store, "summary_plots")
if not os.path.exists(PATH_store_individual_plots):
    os.makedirs(PATH_store_individual_plots)
if not os.path.exists(PATH_store_detailed_plots):
    os.makedirs(PATH_store_detailed_plots)
path_ML = "showers_df_evaluation/0_0_None_hdbscan.pt"
path_pandora = "showers_df_evaluation/0_0_None_pandora.pt"
dir_top = args.path
print(PATH_store)
path_hgcal = os.path.join(dir_top, path_ML)
sd_hgb, matched_hgb = open_mlpf_dataframe(path_hgcal, False)
sd_pandora, matched_pandora = open_mlpf_dataframe(os.path.join(dir_top, path_pandora), False)


sd_hgb, sd_pandora = preprocess_dataframe(sd_hgb, sd_pandora, args.preprocess.split(","))


#plot_track_assignation_eval(sd_hgb, sd_pandora, PATH_store_detailed_plots)
analyze_fakes(sd_pandora, sd_hgb, PATH_store_detailed_plots)
pandora_vertex = np.array(sd_pandora.vertex.values.tolist())
# drop nan values
mask_nan = np.isnan(pandora_vertex).any(axis=1)
pandora_vertex = pandora_vertex[~mask_nan]
hgb_vertex = np.array(sd_hgb.vertex.values.tolist())
mask_nan_hgb = np.isnan(hgb_vertex).any(axis=1)
hgb_vertex = hgb_vertex[~mask_nan_hgb]
displacement_pandora = np.linalg.norm(pandora_vertex, axis=1)
displacement_hgb = np.linalg.norm(hgb_vertex, axis=1)
# filter the df based on where decay type is 0
ranges = [[0, 5000]]   # Ranges of the displacement to make the plots from, in cm
fig, ax = plt.subplots(4, 5, figsize=(22, 22*4/5)) # The overview figure of efficiencies
plot_cm_per_energy(sd_hgb, sd_pandora, PATH_store_detailed_plots, PATH_store_individual_plots)
plot_efficiency_all(sd_pandora, [sd_hgb], PATH_store_individual_plots, ["ML"], ax=ax)
reco_hist(sd_hgb, sd_pandora, PATH_store_individual_plots)
plot_confusion_matrix(sd_hgb, PATH_store_individual_plots, ax=ax[0, 3], ax1=ax[1, 3], ax2=ax[2, 3])
plot_confusion_matrix(sd_hgb, PATH_store_individual_plots, add_pie_charts=True, ax=ax[3, 3])
plot_confusion_matrix_pandora(sd_pandora, PATH_store_individual_plots, ax=ax[0, 4], ax1=ax[1, 4], ax2=ax[2, 4])
plot_confusion_matrix_pandora(sd_pandora, PATH_store_individual_plots, add_pie_charts=True, ax=ax[3, 4])
x_position = 3 / 5  # Normalize the position of the line between the 3rd and 4th columns
fig.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing if necessary
fig.add_artist(plt.Line2D([x_position, x_position], [0, 1], color="black", linewidth=2, transform=fig.transFigure))
fig.tight_layout()
fig.savefig(os.path.join(PATH_store_detailed_plots, "overview_Efficiency_FakeRate_ConfusionMatrix.pdf"))

for range in ranges:
    allowed_batch_idx = np.where((displacement_hgb < range[1]*10) & (displacement_hgb > range[0]*10))[0]
    sd_hgb_filtered = sd_hgb[sd_hgb.number_batch.isin(allowed_batch_idx)]
    allowed_batch_idx_pandora = np.where((displacement_pandora < range[1]*10) & (displacement_pandora > range[0]*10))[0]
    sd_pandora_filtered = sd_pandora[sd_pandora.number_batch.isin(allowed_batch_idx_pandora)]
    sd_pandora_filtered = renumber_batch_idx(sd_pandora_filtered)
    sd_hgb_filtered = renumber_batch_idx(sd_hgb_filtered)
    x = sd_hgb_filtered.pred_ref_pt_matched[sd_hgb_filtered.is_track_in_cluster==1].values
    x = np.stack(x)
    x = np.linalg.norm(x, axis=1)
    #fig, ax = plt.subplots()
    #bins = np.linspace(0, 0.25, 50)
    #ax.hist(x, bins=bins)
    #fig.savefig(os.path.join(PATH_store_individual_plots, "track_momentum_norm.pdf"))
    idx_pick_reco = np.where(x > 0.15)[0]  # If the track is super far away, pick the reco energy instead of the track energy (weird bad track)
    sd_hgb_filtered[sd_hgb_filtered.is_track_in_cluster==1].calibrated_E.iloc[idx_pick_reco] = sd_hgb_filtered[sd_hgb_filtered.is_track_in_cluster==1].pred_showers_E.iloc[idx_pick_reco]
    print("Range", range, ": Finished collection of data and started plotting")
    e_ranges = [[0, 5], [5, 15], [15, 50]]
    for i in e_ranges:
        print("Range: ", i,
            " | Pandora: ",
            len(
                sd_pandora[
                    (sd_pandora.pandora_calibrated_pfo > i[0]) & (sd_pandora.pandora_calibrated_pfo < i[1])
                ]
            ),
            "ML: ",
            len(sd_hgb[(sd_hgb.calibrated_E > i[0]) & (sd_hgb.calibrated_E < i[1])]),
        )
    if len(ranges) == 1:
        current_dir = PATH_store_individual_plots
        current_dir_detailed = PATH_store_detailed_plots
    else:
        current_dir =  os.path.join(PATH_store_individual_plots, "plots_range_" + str(range[0]) + "_" + str(range[1]))
        current_dir_detailed = os.path.join(PATH_store_detailed_plots, "plots_range_" + str(range[0]) + "_" + str(range[1]))
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    if not os.path.exists(current_dir_detailed):
        os.makedirs(current_dir_detailed)
    plot_per_energy_resolution2_multiple(
        sd_pandora_filtered,
        {"ML": sd_hgb_filtered},
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
