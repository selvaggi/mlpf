import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.rc("font", size=35)
import matplotlib.pyplot as plt
import torch
from src.utils.inference.inference_metrics import get_sigma_gaussian
from torch_scatter import scatter_sum, scatter_mean


def calculate_event_energy_resolution(df, pandora=False, full_vector=False):

    true_e = torch.Tensor(df.true_showers_E.values)
    mask_nan_true = np.isnan(df.true_showers_E.values)
    true_e[mask_nan_true] = 0
    batch_idx = df.number_batch
    if pandora:
        pred_E = df.pandora_calibrated_E.values
        nan_mask = np.isnan(df.pandora_calibrated_E.values)
        pred_E[nan_mask] = 0
        pred_e1 = torch.tensor(pred_E).unsqueeze(1).repeat(1, 3)
        pred_vect = torch.tensor(np.array(df.pandora_calibrated_pos.values.tolist()))
        pred_vect[nan_mask] = 0
        true_vect = torch.tensor(np.array(df.true_pos.values.tolist()))
        true_vect[mask_nan_true] = 0
    else:
        pred_E = df.calibrated_E.values
        nan_mask = np.isnan(df.calibrated_E.values)
        print(np.sum(nan_mask))
        pred_E[nan_mask] = 0
        pred_e1 = torch.tensor(pred_E).unsqueeze(1).repeat(1, 3)
        pred_vect = torch.tensor(
            np.array(df.pred_pos_matched.values.tolist()) * pred_e1.numpy()
        )
        pred_vect[nan_mask] = 0
        true_vect = torch.tensor(np.array(df.true_pos.values.tolist()))
        true_vect[mask_nan_true] = 0

    batch_idx = torch.tensor(batch_idx.values).long()
    pred_E = torch.tensor(pred_E)

    true_jet_vect = scatter_sum(true_vect, batch_idx, dim=0)
    pred_jet_vect = scatter_sum(pred_vect, batch_idx, dim=0)
    true_E_jet = scatter_sum(torch.tensor(true_e), batch_idx)
    pred_E_jet = scatter_sum(torch.tensor(pred_E), batch_idx)
    true_jet_p = torch.norm(
        true_jet_vect, dim=1
    )  # This is actually momentum resolution
    pred_jet_p = torch.norm(pred_jet_vect, dim=1)

    mass_true = torch.sqrt(torch.abs(true_E_jet**2 - true_jet_p**2))
    mass_pred = torch.sqrt(torch.abs(pred_E_jet**2 - pred_jet_p**2))

    mass_over_true = mass_pred / mass_true

    return mass_over_true


def get_response_for_event_energy(matched_pandora, matched_):
    mass_over_true_pandora = calculate_event_energy_resolution(
        matched_pandora, True, True
    )

    mass_over_true_model = calculate_event_energy_resolution(matched_, False, True)
    dic = {}
    dic["mass_over_true_model"] = mass_over_true_model
    dic["mass_over_true_pandora"] = mass_over_true_pandora
    return dic
