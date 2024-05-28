import mplhep as hep

hep.style.use("CMS")
import matplotlib

matplotlib.rc("font", size=15)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.inference.pandas_helpers import open_mlpf_dataframe
import seaborn as sns


def main():
    image_path = "/eos/user/m/mgarciam/datasets_mlpf/models_trained_Ks/evaluation_mass/"
    path_hgcal = "/eos/user/m/mgarciam/datasets_mlpf/models_trained_Ks/test_mass/showers_df_evaluation/0_0_None_hdbscan.pt"
    path_pandora = "/eos/user/m/mgarciam/datasets_mlpf/models_trained_Ks/test_mass/showers_df_evaluation/0_0_None_pandora.pt"
    
    sd_hgb, matched_hgb = open_mlpf_dataframe(path_hgcal, False)
    sd_hgb1, matched_hgbq1 = open_mlpf_dataframe(path_pandora, False)

    mass_ks_true, mass_ks_p, mass_rel_pandora = calculate_mass(sd_hgb1)
    plot_ks_rel_mass(mass_rel_pandora)


def plot_ks_rel_mass(mass_rel_pandora, image_path):

    fig = plt.figure()
    sns.histplot(mass_rel_pandora, binwidth=0.1, stat="density")
    plt.xlim([0, 5])
    fig.savefig(image_path, bbox_inches="tight")

def calculate_mass(sd_hgb1):
    p_v_true = sd_hgb1.true_pos.values
    E_true = sd_hgb1.true_showers_E.values
    p_v_pandora = sd_hgb1.pandora_calibrated_pos.values
    E_pandora = sd_hgb1.pandora_calibrated_pfo.values
    batch_number = sd_hgb1.number_batch.values
    mass_ks_true = []
    mass_ks_p = []
    mass_rel_pandora = []
    for batch_id in range(0, int(np.max(batch_number))):
        mask = batch_number == batch_id
        if np.sum(mask) > 0:
            energy_ks = np.nansum(E_true[mask])
            p_v_true_ = [p_v_true[mask][i] for i in range(0, len(p_v_true[mask]))]
            p_v_true_ = np.array(p_v_true_)
            pxj = np.nansum(p_v_true_[:, 0])
            pyj = np.nansum(p_v_true_[:, 1])
            pzj = np.nansum(p_v_true_[:, 2])
            mj = (np.abs(energy_ks**2 - (pxj**2 + pyj**2 + pzj**2))) ** (1 / 2)

            energy_ks_p = np.nansum(E_pandora[mask])
            p_v_p = [p_v_pandora[mask][i] for i in range(0, len(p_v_pandora[mask]))]
            p_v_p = np.array(p_v_p)
            pxj_p = np.nansum(p_v_p[:, 0])
            pyj_p = np.nansum(p_v_p[:, 1])
            pzj_p = np.nansum(p_v_p[:, 2])
            mj_p = (
                np.abs(energy_ks_p**2 - (pxj_p**2 + pyj_p**2 + pzj_p**2))
            ) ** (1 / 2)
            if mj > 0:
                mass_ks_true.append(mj)
                if mj_p > 0:
                    mass_ks_p.append(mj_p)
                    mass_rel_pandora.append(mj_p / mj)
                else:
                    mass_ks_p.append(-1)
    return mass_ks_true, mass_ks_p, mass_rel_pandora


if __name__ == "__main__":
    main()
