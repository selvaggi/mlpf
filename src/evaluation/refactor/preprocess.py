import numpy as np
import pandas as pd
from src.utils.pid_conversion import pid_conversion_dict

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

def remove_fakes(df):
    return df[~pd.isna(df.pid)]


def ablation_study(df, column):
    # Replace all except the column in df with ground truth - to study only the effect of a single column on the results
    is_pandora = "pandora_calibrated_pfo" in df.columns
    assert column in ["E", "p", "pid"]
    fakes = pd.isna(df.pid)
    missed = pd.isna(df.pred_showers_E)
    if column == "E":
        if is_pandora:
            #df.loc[(~fakes) & (~missed), "pandora_calibrated_pfo"] = df.loc[~fakes, "true_showers_E"]
            df.loc[(~fakes) & (~missed), "pandora_calibrated_pos"] = df.loc[~fakes, "true_pos"]  # set pxyz to true
            df.loc[(~fakes) & (~missed), "pandora_pid"] = df.loc[~fakes, "pid"].map(pid_conversion_dict)  # set pxyz to true
        else:
            #df.loc[(~fakes) & (~missed), "calibrated_E"] = df.loc[~fakes, "true_showers_E"] # set E to true
            df.loc[(~fakes) & (~missed), "pred_pos_matched"] = df.loc[~fakes, "true_pos"]  # set pxyz to true
            df.loc[(~fakes) & (~missed), "pred_pid_matched"] = df.loc[~fakes, "pid"].map(pid_conversion_dict)  # set pxyz to true]
    elif column == "p":
        if is_pandora:
            df.loc[(~fakes) & (~missed), "pandora_calibrated_pfo"] = df.loc[~fakes, "true_showers_E"]
            #df.loc[(~fakes) & (~missed), "pandora_calibrated_pos"] = df.loc[~fakes, "true_pos"]  # set pxyz to true
            df.loc[(~fakes) & (~missed), "pandora_pid"] = df.loc[~fakes, "pid"].map(pid_conversion_dict)  # set pxyz to true
        else:
            df.loc[(~fakes) & (~missed), "calibrated_E"] = df.loc[~fakes, "true_showers_E"] # set E to true
            #df.loc[(~fakes) & (~missed), "pred_pos_matched"] = df.loc[~fakes, "true_pos"]  # set pxyz to true
            df.loc[(~fakes) & (~missed), "pred_pid_matched"] = df.loc[~fakes, "pid"].map(pid_conversion_dict)  # set pxyz to true]
    elif column == "pid":
        if is_pandora:
            df.loc[(~fakes) & (~missed), "pandora_calibrated_pfo"] = df.loc[~fakes, "true_showers_E"]
            df.loc[(~fakes) & (~missed), "pandora_calibrated_pos"] = df.loc[~fakes, "true_pos"]  # set pxyz to true
            #df.loc[(~fakes) & (~missed), "pandora_pid"] = df.loc[~fakes, "pid"].map(pid_conversion_dict)  # set pxyz to true
        else:
            df.loc[(~fakes) & (~missed), "calibrated_E"] = df.loc[~fakes, "true_showers_E"] # set E to true
            df.loc[(~fakes) & (~missed), "pred_pos_matched"] = df.loc[~fakes, "true_pos"]  # set pxyz to true
            #df.loc[(~fakes) & (~missed), "pred_pid_matched"] = df.loc[~fakes, "pid"].map(pid_conversion_dict)  # set pxyz to true]
    return df
def preprocess_dataframe(sd_hgb, sd_pandora, names=""):
    # names: list of scripts to do on data
    # sd_hgb: pandas dataframe with the HGB data
    # sd_pandora: pandas dataframe with the Pandora data
    sd_hgb = sd_hgb[sd_hgb.reco_showers_E != 0.0]
    sd_pandora = sd_pandora[sd_pandora.reco_showers_E != 0.0]
    if "class_correction" in names:
        sd_hgb = apply_class_correction(sd_hgb)
    sd_hgb.loc[sd_hgb.pred_pid_matched == 3, "calibrated_E"] = sd_hgb.loc[
        sd_hgb.pred_pid_matched == 3, "pred_showers_E"] # correct photons
    if "beta_correction" in names:
        sd_hgb = apply_beta_correction(sd_hgb)
    if "remove_fakes" in names:
        sd_hgb = remove_fakes(sd_hgb)
        sd_pandora = remove_fakes(sd_pandora)
    return renumber_batch_idx(sd_hgb), renumber_batch_idx(sd_pandora)

