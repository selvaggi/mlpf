# Doesn't plot anything, just saves the mass data - for processing bigger files (>10k events)

from src.utils.pid_conversion import pid_conversion_dict
from torch_scatter import scatter_sum
import numpy as np
import os
import pandas as pd
def open_mlpf_dataframe(path_mlpf, neutrals_only=False):
    data = pd.read_pickle(path_mlpf)
    if neutrals_only:
        sd = pd.concat(
            [
                data[data["pid"] == 130],
                data[data["pid"] == 2112],
                data[data["pid"] == 211],
            ]
        )
    else:
        sd = data
    mask = (~np.isnan(sd["pred_showers_E"])) & (~np.isnan(sd["reco_showers_E"]))
    sd["pid_4_class_true"] = sd["pid"].map(pid_conversion_dict)
    if "pred_pid_matched" in sd.columns:
        sd.loc[sd["pred_pid_matched"] < -1, "pred_pid_matched"] = np.nan
    matched = sd[mask]
    return sd, matched

import pickle
colors_list = ["#deebf7", "#9ecae1", "#d415bd"]  # color list Jan
all_E = True
neutrals_only = False
log_scale = False
tracks = True
perfect_pid = False   # Pretend we got ideal PID and rescale the momentum vectors accordingly
mass_zero = False    # Set the mass to zero for all particles
ML_pid = True       # Use the PID from the ML classification head (electron/CH/NH/gamma)
import torch

particle_masses = {0: 0, 22: 0, 11: 0.00511, 211: 0.13957, 130: 0.493677, 2212: 0.938272, 2112: 0.939565}
particle_masses_4_class = {0: 0.00511, 1: 0.13957, 2: 0.939565, 3: 0.0} # electron, CH, NH, photon
def safeint(x, default_val=0):
    if np.isnan(x):
        return default_val
    return int(x)

def calculate_event_mass_resolution_light(df, pandora, perfect_pid=False, mass_zero=False, ML_pid=False, pandora_perf_pid=False):
    true_e = torch.Tensor(df.true_showers_E.values)
    mask_nan_true = np.isnan(df.true_showers_E.values)
    true_e[mask_nan_true] = 0
    batch_idx = df.number_batch
    if pandora:
        pred_E = df.pandora_calibrated_pfo.values
        nan_mask = np.isnan(df.pandora_calibrated_pfo.values)
        pred_E[nan_mask] = 0
        pred_e1 = torch.tensor(pred_E).unsqueeze(1).repeat(1, 3)
        pred_vect = torch.tensor(np.array(df.pandora_calibrated_pos.values.tolist()))
        nan_mask_p = torch.isnan(pred_vect).any(dim=1)
        pred_vect[nan_mask_p] = 0
        true_vect = torch.tensor(np.array(df.true_pos.values.tolist()))
        mask_nan_p = torch.isnan(true_vect).any(dim=1)
        true_vect[mask_nan_true] = 0
    else:
        pred_E = df.calibrated_E.values
        nan_mask = np.isnan(df.calibrated_E.values)
        print(np.sum(nan_mask))
        pred_E[nan_mask] = 0
        pred_e1 = torch.tensor(pred_E).unsqueeze(1).repeat(1, 3)
        pred_vect = torch.tensor(
            np.array(df.pred_pos_matched.values.tolist())
        )
        pred_vect[nan_mask] = 0
        true_vect = torch.tensor(
            np.array(df.true_pos.values.tolist())
        )
        true_vect[mask_nan_true] = 0
    if perfect_pid or mass_zero or ML_pid:
        pred_vect /= np.linalg.norm(pred_vect, axis=1).reshape(-1, 1)
        pred_vect[np.isnan(pred_vect)] = 0
        if ML_pid:
            #assert pandora is False
            if pandora:
                print("Using Pandora PID")
                #if pandora_perf_pid:
                #    m = np.array([particle_masses.get(abs(safeint(i)), 0) for i in df.pid])
                #else:
                m = np.array([particle_masses.get(abs(safeint(i)), 0) for i in df.pandora_pid])
            else:
                m = np.array([particle_masses_4_class.get(safeint(i), 0) for i in df.pred_pid_matched.values])
        else:
            m = np.array([particle_masses.get(abs(safeint(i)), 0) for i in df.pid])
        if mass_zero:
            m = np.array([0 for _ in m])
        p_squared = (pred_E ** 2 - m ** 2)
        p_squared[p_squared < 0] = 0 # they are always like of order -1e-8
        pred_vect = np.sqrt(p_squared).reshape(-1, 1) * np.array(pred_vect)
    batch_idx = torch.tensor(batch_idx.values).long()
    pred_E = torch.tensor(pred_E)
    true_jet_vect = scatter_sum(true_vect, batch_idx, dim=0)
    pred_jet_vect = scatter_sum(torch.tensor(pred_vect), batch_idx, dim=0)
    true_E_jet = scatter_sum(torch.tensor(true_e), batch_idx)
    pred_E_jet = scatter_sum(torch.tensor(pred_E), batch_idx)
    true_jet_p = torch.norm(true_jet_vect, dim=1)  # This is actually momentum resolution
    pred_jet_p = torch.norm(pred_jet_vect, dim=1)
    mass_true = torch.sqrt(torch.abs(true_E_jet ** 2) - true_jet_p ** 2)
    mass_pred_p = torch.sqrt(
        torch.abs(pred_E_jet ** 2) - pred_jet_p ** 2)  ## TODO: fix the nan values in pred_jet_p!!!!!
    return mass_pred_p, mass_true

# Is there a problem with storing direction information with Pandora?
# /eos/user/g/gkrzmanc/2024/Sept24/Eval_Hss_test_Neutrals_Avg_direction_1file
#/eos/user/g/gkrzmanc/2024/Sept24/1file_Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters0610
if all_E:
    PATH_store = (
        #"/eos/user/g/gkrzmanc/2024/Sept24/MoreFiles_Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters/no_gamma_corr"
        #"/eos/user/g/gkrzmanc/2024/Sept24/MoreFiles_Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters_PandoraPID"
       # "/eos/user/g/gkrzmanc/2024/Sept24/MoreFiles_Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters_PandoraPID"
        "/eos/user/g/gkrzmanc/2024/Sept24/1file_Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters0610_hdbeps02"
    )
    if not os.path.exists(PATH_store):
        os.makedirs(PATH_store)
    plots_path = os.path.join(PATH_store, "plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    path_list = [
        #"Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters/showers_df_evaluation/0_0_None_hdbscan.pt"
        #"MoreFiles_Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters_PandoraPID/showers_df_evaluation/0_0_None_hdbscan.pt"
        #"Eval_Hss_test_GT_clustering_101505_model/showers_df_evaluation/0_0_None_hdbscan.pt"
        #"1file_Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters0610/showers_df_evaluation/0_0_None_hdbscan.pt"
        "1file_Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters0610_hdbeps02/showers_df_evaluation/0_0_None_hdbscan.pt"
    ]
    #path_pandora = "MoreFiles_Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters_PandoraPID/showers_df_evaluation/0_0_None_pandora.pt"
    path_pandora = "1file_Eval_Hss_test_Neutrals_Avg_FT_E_p_PID_Use_model_Clusters0610_hdbeps02/showers_df_evaluation/0_0_None_pandora.pt"
    dir_top = "/eos/user/g/gkrzmanc/2024/Sept24/"
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

do_pandora = True
do_ml = True
pd.options.mode.copy_on_write = True
def main():
    if do_ml:
        for idx, i in enumerate(path_list):
            path_hgcal = os.path.join(dir_top, i)
            print("Opening model df")
            sd_hgb, matched_hgb = open_mlpf_dataframe(path_hgcal, neutrals_only)
            print("Opened model df")
            mask = (~np.isnan(sd_hgb.calibrated_E)) & (sd_hgb.pid==22)
            print("Computed mask")
            mask_idx = np.where(mask)[0]
            #sd_hgb.calibrated_E[mask] = sd_hgb.pred_showers_E[mask]
            # split mask_idx into 10 parts and do each one separately:
            print("Filtering...")
            for i in range(10):
                print(f"Filtering part {i}")
                mask_idx_part = mask_idx[i*len(mask_idx)//10:(i+1)*len(mask_idx)//10]
                mask_part = np.zeros(len(sd_hgb), dtype=bool)
                mask_part[mask_idx_part] = True
                sd_hgb.loc[mask_part, "calibrated_E"] = sd_hgb.loc[mask_part, "pred_showers_E"]
            sd_hgb.iloc[mask_idx, sd_hgb.columns.get_loc("calibrated_E")] = sd_hgb.iloc[mask_idx, sd_hgb.columns.get_loc("pred_showers_E")]
            print("Filtered!")
            print("Set calibrated_E to sum_hits for photons...")
        print("Compute mass resolution")
        m_model, m_true_model = calculate_event_mass_resolution_light(sd_hgb, pandora=False, ML_pid=True)
        print("Save mass model")
        save_dict(m_model, os.path.join(PATH_store, "mass_model.pkl"))
        print("Save mass true")
        save_dict(m_true_model, os.path.join(PATH_store, "mass_true_model.pkl"))
        print("Done")
    if do_pandora:
        print("Opening pandora df")
        sd_pandora, matched_pandora = open_mlpf_dataframe(
            dir_top + path_pandora, neutrals_only
        )
        print("Opened pandora df")
        sd_pandora_copy = sd_pandora.copy()

        m_pandora, m_true_pandora = calculate_event_mass_resolution_light(sd_pandora, pandora=True, ML_pid=True)
        m_pandora_perfpid, m_true_pandora_perfpid = calculate_event_mass_resolution_light(sd_pandora_copy, pandora=True, ML_pid=True, pandora_perf_pid=True)
        # "save these files"
        save_dict(m_pandora, os.path.join(PATH_store, "mass_pandora.pkl"))
        save_dict(m_true_pandora, os.path.join(PATH_store, "mass_true_pandora.pkl"))
        save_dict(m_pandora_perfpid, os.path.join(PATH_store, "mass_pandora_perfpid.pkl"))
        save_dict(m_true_pandora_perfpid, os.path.join(PATH_store, "mass_true_pandora_perfpid.pkl"))


def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


if __name__ == "__main__":
    main()
