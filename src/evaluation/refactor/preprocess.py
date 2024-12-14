import numpy as np
import pandas as pd
from src.utils.pid_conversion import pid_conversion_dict
from src.utils.inference.per_particle_metrics import compute_score_certainty
import torch

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

pid_correction_track = {0: 0, 1: 1, 2: 1, 3: 0, 4: 4}
pid_correction_no_track = {0: 3, 1: 2, 2: 2, 3: 3, 4: 3}

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
    filter = (np.isnan(highest_beta)) | (highest_beta >= 0.3) | (highest_beta <= 0.03)
    cali_E = sd_hgb[~filter].calibrated_E
    fakes = pd.isna(sd_hgb[~filter].pid)
    print("E stored in cut off fakes:", cali_E[fakes].sum())
    print("E stored in cut off matched particles:", cali_E[~fakes].sum()) # we can play a bit with this and see where the optimal cutoff is...
    sd_hgb = sd_hgb[filter]
    return sd_hgb

def remove_fakes(df):
    return df[~pd.isna(df.pid)]


def ablation_study(df, column):
    # Replace all except the column in df with ground truth - to study only the effect of a single column on the results
    print("Setting everything except column", column, "to GT")
    is_pandora = "pandora_calibrated_pfo" in df.columns
    assert column in ["E", "p", "pid"]
    fakes = pd.isna(df.pid)
    missed = pd.isna(df.pred_showers_E)
    index = (~fakes) & (~missed)
    if column == "E":
        if is_pandora:
            df.loc[index, "pandora_calibrated_pos"] = df.loc[index, "true_pos"]  # set pxyz to true
            df.loc[index, "pandora_pid"] = df.loc[index, "pid"]#.map(pid_conversion_dict)  # set pxyz to true
        else:
            df.loc[index, "pred_pos_matched"] = df.loc[index, "true_pos"]  # set pxyz to true
            df.loc[index, "pred_pid_matched"] = df.loc[index, "pid"].map(pid_conversion_dict)  # set pxyz to true]
    elif column == "p":
        if is_pandora:
            df.loc[index, "pandora_calibrated_pfo"] = df.loc[index, "true_showers_E"]
            df.loc[index, "pandora_pid"] = df.loc[index, "pid"]#.map(pid_conversion_dict)  # set pxyz to true
        else:
            df.loc[index, "calibrated_E"] = df.loc[index, "true_showers_E"] # set E to true
            df.loc[index, "pred_pid_matched"] = df.loc[index, "pid"].map(pid_conversion_dict)  # set pxyz to true]
    elif column == "pid":
        if is_pandora:
            df.loc[index, "pandora_calibrated_pfo"] = df.loc[index, "true_showers_E"]
            df.loc[index, "pandora_calibrated_pos"] = df.loc[index, "true_pos"]  # set pxyz to true
        else:
            df.loc[index, "calibrated_E"] = df.loc[index, "true_showers_E"] # set E to true
            df.loc[index, "pred_pos_matched"] = df.loc[index, "true_pos"]  # set pxyz to true
    print("Done")
    return df
def preprocess_dataframe(sd_hgb, sd_pandora, names=""):
    # names: list of scripts to do on data
    # sd_hgb: pandas dataframe with the HGB data
    # sd_pandora: pandas dataframe with the Pandora data
    #sd_hgb = sd_hgb[sd_hgb.reco_showers_E != 0.0]
    #sd_pandora = sd_pandora[sd_pandora.reco_showers_E != 0.0]
    if "class_correction" in names:
        sd_hgb = apply_class_correction(sd_hgb)
    if "no_correct_reco_photons" in names:
        print("Leaving photons predicted energies as they are")
    elif "reco_correct_gt_photons" in names:
        sd_hgb.loc[sd_hgb.pid == 22, "calibrated_E"] = sd_hgb.loc[
            sd_hgb.pid == 22, "reco_showers_E"]
    else:
        sd_hgb.loc[sd_hgb.pred_pid_matched == 3, "calibrated_E"] = sd_hgb.loc[
            sd_hgb.pred_pid_matched == 3, "pred_showers_E"] # Correct photons
    if "beta_correction" in names:
        sd_hgb = apply_beta_correction(sd_hgb)
    if "remove_fakes" in names:
        sd_hgb = remove_fakes(sd_hgb)
        sd_pandora = remove_fakes(sd_pandora)
    if "ablation_study_E" in names:
        sd_hgb = ablation_study(sd_hgb, "E")
        sd_pandora = ablation_study(sd_pandora, "E")
    if "ablation_study_p" in names:
        sd_hgb = ablation_study(sd_hgb, "p")
        sd_pandora = ablation_study(sd_pandora, "p")
    if "ablation_study_pid" in names:
        sd_hgb = ablation_study(sd_hgb, "pid")
        sd_pandora = ablation_study(sd_pandora, "pid")
    if "take_out_gt_photons" in names:
        print("Take out GT photons")
        sd_hgb = sd_hgb[sd_hgb.pid != 22]
        sd_pandora = sd_pandora[sd_pandora.pid != 22]
    if "take_out_pred_photons" in names:
        print("Take out predicted photons")
        sd_hgb = sd_hgb[sd_hgb.pred_pid_matched != 3]
        sd_pandora = sd_pandora[sd_pandora.pandora_pid != 22]
    if "take_out_pred_photons_0_1" in names:
        print("Take out predicted photons [0,1 GeV]")
        mask_energy = sd_hgb.calibrated_E < 1.0
        if "take_out_fakes_only" in names:
            print("Fakes only!")
            mask_energy = mask_energy & (pd.isna(sd_hgb.pid))
        mask = mask_energy & (sd_hgb.pred_pid_matched == 3)
        sd_hgb = sd_hgb[~mask]
        mask_energy_p = sd_pandora.reco_showers_E < 1.0
        if "take_out_fakes_only" in names:
            mask_energy_p = mask_energy_p & (pd.isna(sd_pandora.pid))
        mask_p = mask_energy_p & (sd_pandora.pandora_pid == 22)
        sd_pandora = sd_pandora[~mask_p]
    if "take_out_pred_photons_1_10" in names:
        print("Take out predicted photons [1,10 GeV]")
        mask_energy = (sd_hgb.calibrated_E < 10) & (sd_hgb.calibrated_E > 1.0)
        if "take_out_fakes_only" in names:
            print("Fakes only!")
            mask_energy = mask_energy & (pd.isna(sd_hgb.pid))
        mask = mask_energy & (sd_hgb.pred_pid_matched == 3)
        sd_hgb = sd_hgb[~mask]
        mask_energy_p = (sd_pandora.pandora_calibrated_pfo < 10) & (sd_pandora.pandora_calibrated_pfo > 1.0)
        mask_p = mask_energy_p & (sd_pandora.pandora_pid == 22)
        if "take_out_fakes_only" in names:
            mask_p = mask_p & (pd.isna(sd_pandora.pid))
        sd_pandora = sd_pandora[~mask_p]
    if "take_out_pred_photons_10_100" in names:
        print("Take out predicted photons [10,100 GeV]")
        mask_energy = (sd_hgb.calibrated_E < 100) & (sd_hgb.calibrated_E > 10)
        if "take_out_fakes_only" in names:
            print("Fakes only!")
            mask_energy = mask_energy & (pd.isna(sd_hgb.pid))
        mask = mask_energy & (sd_hgb.pred_pid_matched == 3)
        sd_hgb = sd_hgb[~mask]
        mask_energy_p = (sd_pandora.pandora_calibrated_pfo < 100) & (sd_pandora.pandora_calibrated_pfo > 10)
        mask_p = mask_energy_p & (sd_pandora.pandora_pid == 22)
        if "take_out_fakes_only" in names:
            mask_p = mask_p & (pd.isna(sd_pandora.pid))
        sd_pandora = sd_pandora[~mask_p]
    #if "remove_weird_tracks" in names:
    #    x = sd_hgb.pred_ref_pt_matched.values
    #    x = np.stack(x)
    #    x = np.linalg.norm(x, axis=1)
    #    idx_pick_reco = (x > 0.15) & ((sd_hgb.is_track_in_cluster == 1).values)
    #    # If the track is super far away, pick the reco energy instead of the track energy (weird bad track)
    #    sd_hgb.loc[idx_pick_reco, "calibrated_E"] = sd_hgb.loc[idx_pick_reco, "pred_showers_E"]
    if "filt_LE_CH" in names:
        print("Filtering low-energy CH")
        dist_trk = np.linalg.norm(np.stack(sd_hgb.pred_ref_pt_matched.values), axis=1)
        ch_le_filter = (dist_trk >= 0.21) & (sd_hgb.pred_pid_matched == 1) & (sd_hgb.calibrated_E < 5.0)
        # remove ch_le_filter
        # this doesn't work! try another way?
        sd_hgb.loc[ch_le_filter, "calibrated_E"] = np.nan
        sd_hgb.loc[ch_le_filter, "pred_showers_E"] = np.nan
        sd_hgb.loc[ch_le_filter, "pred_pos"] = np.nan
        sd_hgb.loc[ch_le_filter, "pred_pid_matched"] = np.nan
        sd_hgb.loc[ch_le_filter, "pred_ref_pt_matched"] = np.nan
    if "replace_LE_CH" in names:
        print("Filtering low-energy CH - replacing the track with sum of the hits")
        dist_trk = np.linalg.norm(np.stack(sd_hgb.pred_ref_pt_matched.values), axis=1)
        ch_le_filter = (dist_trk >= 0.21) & (sd_hgb.pred_pid_matched == 1) & (sd_hgb.calibrated_E < 5.0)
        # remove ch_le_filter
        #sd_hgb[ch_le_filter].calibrated_E = sd_hgb[ch_le_filter].pred_showers_E
        sd_hgb.loc[ch_le_filter, "calibrated_E"] = sd_hgb.loc[ch_le_filter, "pred_showers_E"]
    if "remove_uncertain_particles" in names:
        logits = torch.tensor(np.stack(sd_hgb.matched_extra_features.values))[:, 2:]
        logits = torch.softmax(logits, dim=1)
        score = compute_score_certainty(logits)
        mask = (pd.isna(score)) | (score > 0.85) # makes it even slightly worse
        mask = pd.Series(mask)
        sd_hgb = sd_hgb[mask]
        #sd_hgb = sd_hgb.iloc[torch.where(~mask)[0]]
    return renumber_batch_idx(sd_hgb), renumber_batch_idx(sd_pandora)
