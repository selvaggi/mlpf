# Similar to evaluate_mix, but plots a comparison between different ML methods on the same plot

import matplotlib

from src.utils.inference.per_particle_metrics import plot_per_energy_resolution, reco_hist

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
#matplotlib.rc("font", size=35)
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
import os
from utils.inference.pandas_helpers import open_hgcal, open_mlpf_dataframe
from utils.inference.per_particle_metrics import (
    plot_per_energy_resolution2_multiple, plot_confusion_matrix,
    plot_efficiency_all, calc_unit_circle_dist, plot_per_energy_resolution2
)
from src.utils.inference.event_Ks import get_decay_type
import matplotlib.pyplot as plt
import mplhep as hep
import torch
import pickle
hep.style.use("CMS")
colors_list = ["#deebf7", "#9ecae1", "#d415bd"]  # color list Jan
all_E = True

neutrals_only = False
log_scale = False
tracks = True
perfect_pid = False   # Pretend we got ideal PID and rescale the momentum vectors accordingly
mass_zero = True    # Set the mass to zero for all particles
ML_pid = False       # Use the PID from the ML classification head (electron/CH/NH/gamma)

# Is there a problem with storing direction information with Pandora?
# /eos/user/g/gkrzmanc/2024/Sept24/Eval_Hss_test_Neutrals_Avg_direction_1file

if all_E:
    PATH_store = ( #/eos/user/g/gkrzmanc/2024/train/export_f_10_09_testset_300_files_avg_pos_reprod
        "/eos/user/g/gkrzmanc/2024/results/evalHss_reprod_clust_only_newmodel_Clustering061024"
    )
    if not os.path.exists(PATH_store):
        os.makedirs(PATH_store)
    plots_path = os.path.join(PATH_store, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    path_list = [
        #"Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters/showers_df_evaluation/0_0_None_hdbscan.pt"
        #"Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters_model_0610/showers_df_evaluation/0_0_None_hdbscan.pt"
        #"Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters_model_0610/showers_df_evaluation/0_0_None_hdbscan.pt"  # THIS ONE IS OK
        "evalHss_reprod_clust_only_newmodel_Clustering061024/showers_df_evaluation/0_0_None_hdbscan.pt"
    ]
    #path_pandora = "Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters/showers_df_evaluation/0_0_None_pandora.pt"
    path_pandora = "evalHss_reprod_clust_only_newmodel_Clustering061024/showers_df_evaluation/0_0_None_pandora.pt"
    dir_top = "/eos/user/g/gkrzmanc/2024/results/"
    #dir_top = "/eos/user/g/gkrzmanc/eval_plots_EC/"
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

def main():
    df_list = []
    matched_all = {}
    for idx, i in enumerate(path_list):
        path_hgcal = os.path.join(dir_top, i)
        sd_hgb, matched_hgb = open_mlpf_dataframe(path_hgcal, neutrals_only)
        #sd_hgb.pred_showers_E = sd_hgb.reco_showers_E
        #print("!!!! Taking the sum of the hits for the energy !!!!")
        # sd_hgb = renumber_batch_idx(sd_hgb[(sd_hgb.pid==22) | (pd.isna(sd_hgb.pid))])
        sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pid==22)] = sd_hgb.pred_showers_E[(~np.isnan(sd_hgb.calibrated_E)) & ((sd_hgb.pid==22))]
        # set GT energy for 130, 2112, 22
        #
        ##sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pid==130)] = sd_hgb.true_showers_E[(~np.isnan(sd_hgb.calibrated_E)) & ((sd_hgb.pid==130))]
        #sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pid==2112)] = sd_hgb.true_showers_E[(~np.isnan(sd_hgb.calibrated_E)) & ((sd_hgb.pid==2112))]
        #sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pid==130)] = sd_hgb.pred_showers_E[(~np.isnan(sd_hgb.calibrated_E)) & ((sd_hgb.pid==130))]
        #sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pid==2112)] = sd_hgb.pred_showers_E[(~np.isnan(sd_hgb.calibrated_E)) & ((sd_hgb.pid==2112))]
        #sd_hgb.calibrated_E[~np.isnan(sd_hgb.calibrated_E) & (sd_hgb.pred_pid_matched==3)] = sd_hgb.reco_showers_E[~np.isnan(sd_hgb.calibrated_E) & (sd_hgb.pred_pid_matched==3)]
        #sd_hgb.calibrated_E[~np.isnan(sd_hgb.calibrated_E) & (sd_hgb.pred_pid_matched==2)] = sd_hgb.reco_showers_E[~np.isnan(sd_hgb.calibrated_E) & (sd_hgb.pred_pid_matched==3)]
        #sd_hgb.calibrated_E[(~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pred_pid_matched==130)] = sd_hgb.reco_showers_E[(~np.isnan(sd_hgb.calibrated_E)) & ((sd_hgb.pid==130))]
        df_list.append(sd_hgb)
        matched_all[labels[idx]] = matched_hgb
    sd_pandora, matched_pandora = open_mlpf_dataframe(
        dir_top + path_pandora, neutrals_only
    )
    #sd_pandora = renumber_batch_idx(sd_pandora[(sd_pandora.pid == 22) | (pd.isna(sd_pandora.pid))])
    decay_type = get_decay_type(sd_hgb)
    decay_type_pandora = get_decay_type(sd_pandora)
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
    #sd_pandora = sd_pandora[decay_type_pandora == 1]
    #sd_hgb = sd_hgb[decay_type == 1]
    #allowed_batch_idx = np.where(decay_type_pandora == 1)[0]
    #allowed_batch_idx_pandora = np.where(decay_type == 1)[0]
    #sd_pandora = sd_pandora[sd_pandora.number_batch.isin(allowed_batch_idx)]
    #sd_hgb = sd_hgb[sd_hgb.number_batch.isin(allowed_batch_idx_pandora)]
    # filter the df based on where decay type is 0
    ranges = [[0, 5000]]    # Ranges of the displacement to make the plots from, in cm
    plot_efficiency_all(sd_pandora, df_list, PATH_store, labels)
    #fakes_ml = sd_hgb[pd.isna(sd_hgb.pid)].pred_showers_E.values
    #fakes_pandora = sd_pandora[pd.isna(sd_pandora.pid)].pred_showers_E.values
    # remove fakes
    #sd_hgb = sd_hgb[~pd.isna(sd_hgb.pid)]
    #sd_pandora = sd_pandora[~pd.isna(sd_pandora.pid)]
    reco_hist(sd_hgb, sd_pandora, PATH_store)
    plot_confusion_matrix(df_list[0], PATH_store)
    for range in ranges:
        #metrics = obtain_metrics(sd_pandora, df_list, labels)
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
        fig.show()
        idx_pick_reco = np.where(x > 0.25)[0]  # If the track is super far away, pick the reco energy instead of the track energy (weird bad track)
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
        current_dir =  os.path.join(PATH_store, "plots_range_" + str(range[0]) + "_" + str(range[1]))
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        dir_reco = os.path.join(current_dir, "reco")
        if not os.path.exists(dir_reco):
            os.makedirs(dir_reco)
        #plot_per_energy_resolution(sd_pandora_filtered, sd_hgb_filtered, dir_reco)
        plot_per_energy_resolution2_multiple(
            sd_pandora_filtered,
            {"ML": sd_hgb_filtered},
            current_dir,
            tracks=tracks,
            perfect_pid=perfect_pid,
            mass_zero=mass_zero,
            ML_pid=ML_pid,
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

