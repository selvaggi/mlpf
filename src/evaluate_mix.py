import gzip
import pickle
import matplotlib

matplotlib.rc("font", size=25)
import numpy as np
import pandas as pd

# from mycolorpy import colorlist as mcp
import numpy as np
from utils.inference.inference_metrics_hgcal import obtain_metrics_hgcal
from utils.inference.inference_metrics import obtain_metrics
from utils.inference.pandas_helpers import open_hgcal, open_mlpf_dataframe
from utils.inference.plots import plot_metrics, plot_histograms_energy
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")
# colors_list = mcp.gen_color(cmap="cividis", n=3)
colors_list = ["#fff7bc", "#fec44f", "#d95f0e"]

all_E = True
neutrals_only = False
if all_E:
    path_hgcal = None  # "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/hgcal/logs_1015_1911/pandora/analysis/out.bin.gz"
    # path_mlpf = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/mlpf/mlpf_all_energies_hgcal_loss/showers_df_evaluation/0_0_None.pt"
    # path_pandora = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/mlpf/mlpf_all_energies_hgcal_loss/showers_df_evaluation/0_0_None_pandora.pt"
    path_mlpf = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/mlpf/mlpf_all_energies_hgcal_loss/showers_df_evaluation_all_E_notrackeri/0_0_None.pt"
    path_pandora = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/mlpf/mlpf_all_energies_hgcal_loss/showers_df_evaluation_all_E_notrackeri/0_0_None_pandora.pt"
else:
    path_hgcal = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/all_energies_10_15/hgcal/logs_1015_1911/pandora/analysis/out.bin.gz"
    path_mlpf = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/2309/mlpf/mlpf_v3/showers_df_evaluation/0_0_None.pt"
    path_pandora = "/eos/user/m/mgarciam/datasets_mlpf/models_trained/2309/mlpf/mlpf_v3/showers_df_evaluation/0_0_None_pandora.pt"


def main():
    dic1 = False
    dic2 = False
    if path_hgcal is not None:
        dic1 = True
        sd, ms = open_hgcal(path_hgcal, neutrals_only)
        dict_1 = obtain_metrics_hgcal(sd, matched, ms)
    else:
        dict_1 = None
    if path_mlpf is not None:
        dic2 = True
        sd, matched = open_mlpf_dataframe(path_mlpf, neutrals_only)
        dict_2 = obtain_metrics(sd, matched)

        sd, matched = open_mlpf_dataframe(path_pandora, neutrals_only)
        dict_3 = obtain_metrics(sd, matched, pandora=True)

    plot_metrics(neutrals_only, dic1, dic2, dict_1, dict_2, dict_3, colors_list)
    plot_histograms_energy(dic1, dic2, dict_1, dict_2, dict_3)


if __name__ == "__main__":
    main()
