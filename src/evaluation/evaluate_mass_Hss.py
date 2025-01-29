# Similar to evaluate_mix, but plots a comparison between different ML methods on the same plot

import matplotlib
import sys
sys.path.append("/afs/cern.ch/work/m/mgarciam/private/mlpf/")
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
hep.style.use("CMS")
# set hep font size

fs = 10
font = {'size': fs}
matplotlib.rc('font', **font)

colors_list = ["#deebf7", "#9ecae1", "#d415bd"]  # color list Jan
all_E = True

neutrals_only = False
log_scale = False
tracks = True
perfect_pid = False   # Pretend we got ideal PID and rescale the momentum vectors accordingly
mass_zero = False    # Set the mass to zero for all particles
ML_pid = True       # Use the PID from the ML classification head (electron/CH/NH/gamma)

# Is there a problem with storing direction information with Pandora?
# /eos/user/g/gkrzmanc/2024/Sept24/Eval_Hss_test_Neutrals_Avg_direction_1file

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to the folder with the training in which checkpoints are saved",
                    default="/eos/home-g/gkrzmanc/results/2024/eval_clustering_plus_model_epoch4_Hss_300files")


args = parser.parse_args()
if all_E:
    PATH_store = (
        #"/eos/home-g/gkrzmanc/results/2024/eval_clustering_plus_model_epoch4_Hss/model_PID"
        os.path.join(args.path, "corrected_pid_classes_fix_no_beta_corr")
        #args.path
    )
    if not os.path.exists(PATH_store):
        os.makedirs(PATH_store)
    PATH_store_individual_plots = os.path.join(PATH_store, "individual_plots")
    PATH_store_detailed_plots = os.path.join(PATH_store, "summary_plots")
    if not os.path.exists(PATH_store_individual_plots):
        os.makedirs(PATH_store_individual_plots)
    if not os.path.exists(PATH_store_detailed_plots):
        os.makedirs(PATH_store_detailed_plots)

    path_list = [
        "showers_df_evaluation/0_0_None_hdbscan.pt"
    ]
    path_pandora = "showers_df_evaluation/0_0_None_pandora.pt"
    dir_top = args.path
    print(PATH_store)

labels = [
    "ML"
]

def renumber_batch_idx(df):
    # batch_idx has missing numbers
    # renumber it to be like 0,1,2...
    batch_idx = df.number_batch
    unique_batch_idx = np.unique(batch_idx)
    new_to_old_batch_idx = {}
    new_batch_idx = np.zeros(len(batch_idx))
    for idx, i in enumerate(unique_batch_idx):
        new_batch_idx[batch_idx == i] = idx
        new_to_old_batch_idx[idx] = i
    df.number_batch = new_batch_idx
    return df

def filter_df(df):
    # quick filter to exclude problematic particles
    df = df[(df.pid != 11) & (df.pid != 22) ]
    return df

pid_correction_track = {0: 0, 1: 1, 2: 1, 3: 0}
pid_correction_no_track = {0: 3, 1: 2, 2: 2, 3: 3}

def apply_class_correction(sd_hgb):
    # apply the correction to the class according to pid_correction_track and pid_correction_no_track
    # track
    #sd_hgb[sd_hgb.is_track_in_cluster==1].pred_pid_matched = sd_hgb[sd_hgb.is_track_in_cluster==1].pred_pid_matched.apply(lambda x: pid_correction_track.get(x, np.nan))
    sd_hgb.loc[sd_hgb.is_track_in_cluster==1, "pred_pid_matched"] = sd_hgb.loc[sd_hgb.is_track_in_cluster==1, "pred_pid_matched"].apply(lambda x: pid_correction_track.get(x, np.nan))
    # no track
    #sd_hgb[sd_hgb.is_track_in_cluster==0].pred_pid_matched = sd_hgb[sd_hgb.is_track_in_cluster==0].pred_pid_matched.apply(lambda x: pid_correction_no_track.get(x, np.nan))
    sd_hgb.loc[sd_hgb.is_track_in_cluster==0, "pred_pid_matched"] = sd_hgb.loc[sd_hgb.is_track_in_cluster==0, "pred_pid_matched"].apply(lambda x: pid_correction_no_track.get(x, np.nan))
    return sd_hgb

def apply_beta_correction(sd_hgb):
    highest_beta = np.stack(sd_hgb.matched_extra_features.values)[:, 1]
    filter = (np.isnan(highest_beta)) | (highest_beta >= 0.05) | (highest_beta <= 0.03)
    cali_E = sd_hgb[~filter].calibrated_E
    fakes = pd.isna(sd_hgb[~filter].pid)
    print("E stored in cut off fakes:", cali_E[fakes].sum())
    print("E stored in cut off matched particles:", cali_E[~fakes].sum()) # we can play a bit with this and see where the optimal cutoff is...
    sd_hgb = sd_hgb[filter]
    return sd_hgb

def main():
    df_list = []
    matched_all = {}
    for idx, i in enumerate(path_list):
        path_hgcal = os.path.join(dir_top, i)
        sd_hgb, matched_hgb = open_mlpf_dataframe(path_hgcal, neutrals_only)
        #sd_hgb.pred_showers_E = sd_hgb.reco_showers_E
        #print("!!!! Taking the sum of the hits for the energy !!!!")
        # sd_hgb = renumber_batch_idx(sd_hgb[(sd_hgb.pid==22) | (pd.isna(sd_hgb.pid))])
        #sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pid==22)] = sd_hgb.pred_showers_E[(~np.isnan(sd_hgb.calibrated_E)) & ((sd_hgb.pid==22))]
        sd_hgb = apply_class_correction(sd_hgb)
        #sd_hgb[sd_hgb.pred_pid_matched == 3].calibrated_E = sd_hgb[sd_hgb.pred_pid_matched == 3].pred_showers_E
        sd_hgb.loc[sd_hgb.pred_pid_matched == 3, "calibrated_E"] = sd_hgb.loc[sd_hgb.pred_pid_matched == 3, "pred_showers_E"]
        # set GT energy for 130, 2112, 22
        ##sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pid==130)] = sd_hgb.true_showers_E[(~np.isnan(sd_hgb.calibrated_E)) & ((sd_hgb.pid==130))]
        #sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pid==2112)] = sd_hgb.true_showers_E[(~np.isnan(sd_hgcalibrated_E)) & ((sd_hgb.pid==2112))]
        #sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pid==130)] = sd_hgb.pred_showers_E[(~np.isnan(sd_hgb.calibrated_E)) & ((sd_hgb.pid==130))]
        #sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pid==2112)] = sd_hgb.pred_showers_E[(~np.isnan(sd_hgb.calibrated_E)) & ((sd_hgb.pid==2112))]
        #sd_hgb.calibrated_E[~np.isnan(sd_hgb.calibrated_E) & (sd_hgb.pred_pid_matched==3)] = sd_hgb.reco_showers_E[~np.isnan(sd_hgb.calibrated_E) & (sd_hgb.pred_pid_matched==3)]
        #sd_hgb.calibrated_E[~np.isnan(sd_hgb.calibrated_E) & (sd_hgb.pred_pid_matched==2)] = sd_hgb.reco_showers_E[~np.isnan(sd_hgb.calibrated_E) & (sd_hgb.pred_pid_matched==3)]
        #sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pred_pid_matched==130)] = sd_hgb.reco_showers_E[(~np.isnan(sd_hgb.calibrated_E)) & ((sd_hgb.pid==130))]
        df_list.append(sd_hgb)
        matched_all[labels[idx]] = matched_hgb
    sd_pandora, matched_pandora = open_mlpf_dataframe(
        os.path.join(dir_top, path_pandora), neutrals_only
    )

    plot_track_assignation_eval(sd_hgb, sd_pandora, PATH_store_detailed_plots)
    analyze_fakes(sd_pandora, df_list[0], PATH_store_detailed_plots)
    sd_hgb = apply_beta_correction(sd_hgb)
    print("!!! Filtering !!!")
    # Plot distance from (0,0,0) for the matched particles
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
    # plot_cm_per_energy(df_list[0], sd_pandora, PATH_store_detailed_plots, PATH_store_individual_plots)
    plot_efficiency_all(sd_pandora, df_list, PATH_store_individual_plots, labels, ax=ax)
    reco_hist(sd_hgb, sd_pandora, PATH_store_individual_plots)
    plot_confusion_matrix(df_list[0], PATH_store_individual_plots, ax=ax[0, 3], ax1=ax[1, 3], ax2=ax[2, 3])
    plot_confusion_matrix(df_list[0], PATH_store_individual_plots, add_pie_charts=True, ax=ax[3, 3])
    plot_confusion_matrix_pandora(sd_pandora, PATH_store_individual_plots, ax=ax[0, 4], ax1=ax[1, 4], ax2=ax[2, 4])
    plot_confusion_matrix_pandora(sd_pandora, PATH_store_individual_plots, add_pie_charts=True, ax=ax[3, 4])
    x_position = 3 / 5  # Normalize the position of the line between the 3rd and 4th columns
    fig.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing if necessary
    fig.add_artist(plt.Line2D([x_position, x_position], [0, 1], color="black", linewidth=2, transform=fig.transFigure))
    fig.tight_layout()
    # Draw a vertical line between 3rd and 4th column in the big figure
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
        fig, ax = plt.subplots()
        bins = np.linspace(0, 0.25, 50)
        ax.hist(x, bins=bins)
        #ax.set_yscale("log")
        fig.savefig(os.path.join(PATH_store_individual_plots, "track_momentum_norm.pdf"))
        idx_pick_reco = np.where(x > 0.15)[0]  # If the track is super far away, pick the reco energy instead of the track energy (weird bad track)
        sd_hgb_filtered[sd_hgb_filtered.is_track_in_cluster==1].calibrated_E.iloc[idx_pick_reco] = sd_hgb_filtered[sd_hgb_filtered.is_track_in_cluster==1].pred_showers_E.iloc[idx_pick_reco]
        print("Range", range, ": Finished collection of data and started plotting")
        e_ranges = [[0, 5], [5, 15], [15, 50]]
        # Count number of photons in each energy range reconstructed with Pandora or ML and print this info in one line for each energy range
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
        #plot_per_energy_resolution(sd_pandora_filtered, sd_hgb_filtered, dir_reco)

        plot_per_energy_resolution2_multiple(
            sd_pandora_filtered,
            {"ML": sd_hgb_filtered},
            current_dir,
            tracks=tracks,
            perfect_pid=perfect_pid,
            mass_zero=mass_zero,
            ML_pid=ML_pid,
            PATH_store_detailed_plots=current_dir_detailed
        )
        print("Done")

if __name__ == "__main__":
    main()

def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di

