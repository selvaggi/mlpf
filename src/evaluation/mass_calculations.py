 
from torch_scatter import scatter_sum
import pandas as pd
import numpy as np 
import torch 

particle_masses = {0: 0, 22: 0, 11: 0.00511, 211: 0.13957, 130: 0.493677, 2212: 0.938272, 2112: 0.939565}
particle_masses_4_class = {0: 0.00511, 1: 0.13957, 2: 0.939565, 3: 0.0, 4: 0.10566} # electron, CH, NH, photon, muon
def safeint(x, default_val=0):
    if np.isnan(x):
        return default_val
    return int(x)

def get_mass(new_dataset, new_dataset2):
    df = new_dataset
    batch_idx = df.number_batch
    batch_idx = torch.tensor(batch_idx.values).long()
    true_vect = torch.tensor(
                np.array(df.true_pos.values.tolist())
            )
    mask_nan_true = np.isnan(df.true_showers_E.values)
    true_vect[mask_nan_true] = 0

    true_e = torch.Tensor(df.true_showers_E.values)
    true_e[mask_nan_true] = 0
    true_E_jet = scatter_sum(true_e, batch_idx)
    true_jet_vect = scatter_sum(true_vect, batch_idx, dim=0)
    true_jet_p = torch.norm(true_jet_vect, dim=1) 

    mass_true = torch.sqrt((true_E_jet ** 2).abs() - true_jet_p ** 2)
    print(true_E_jet, true_jet_p)
    pred_vect = torch.tensor(
        np.array(df.pred_pos_matched.values.tolist())
    )
    pred_E = df.calibrated_E.values
    nan_mask = np.isnan(df.calibrated_E.values)
    pred_E[nan_mask] = 0
    pred_vect[nan_mask] = 0

    if len(pred_vect) > 0:
        pred_vect /= np.linalg.norm(pred_vect, axis=1).reshape(-1, 1)
        pred_vect[torch.isnan(pred_vect)] = 0

    m = np.array([particle_masses.get(abs(safeint(i)), 0) for i in df.pid])
 
    p_squared = (pred_E ** 2 - m ** 2)
    p_squared[p_squared < 0] = 0 # they are always like of order -1e-8
    pred_vect = np.sqrt(p_squared).reshape(-1, 1) * np.array(pred_vect)

    
    
    pred_E = torch.tensor(pred_E)
    pred_E_jet = scatter_sum(pred_E, batch_idx)
    pred_jet_vect = scatter_sum(torch.tensor(pred_vect), batch_idx, dim=0)
    pred_jet_p = torch.norm(pred_jet_vect, dim=1)
    mass_pred_p_1 = torch.sqrt(torch.abs(pred_E_jet ** 2) - pred_jet_p ** 2)
    print(pred_E_jet, pred_jet_p)


    df = new_dataset2
    batch_idx = df.number_batch
    batch_idx = torch.tensor(batch_idx.values).long()
    pred_vect = torch.tensor(
        np.array(df.pandora_calibrated_pos.values.tolist())
    )

    if len(pred_vect) > 0:
        pred_vect /= np.linalg.norm(pred_vect, axis=1).reshape(-1, 1)
        pred_vect[torch.isnan(pred_vect)] = 0

    m = np.array([particle_masses.get(abs(safeint(i)), 0) for i in df.pid])


    pred_E = df.pandora_calibrated_pfo.values
    nan_mask = np.isnan(df.pandora_calibrated_pfo.values)
    pred_E[nan_mask] = 0
    pred_vect[nan_mask] = 0
    pred_E = torch.tensor(pred_E)
    p_squared = (pred_E ** 2 - m ** 2)
    p_squared[p_squared < 0] = 0 # they are always like of order -1e-8
    pred_vect = np.sqrt(p_squared).reshape(-1, 1) * np.array(pred_vect)
    pred_E_jet = scatter_sum(pred_E, batch_idx)
    pred_jet_vect = scatter_sum(pred_vect, batch_idx, dim=0)
    pred_jet_p = torch.norm(pred_jet_vect, dim=1)
    mass_pred_p_pandora = torch.sqrt(torch.abs(pred_E_jet ** 2) - pred_jet_p ** 2)
    print(pred_E_jet, pred_jet_p)
    return mass_pred_p_1,mass_pred_p_pandora,  mass_true
