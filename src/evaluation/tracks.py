import matplotlib

matplotlib.rc("font", size=35)
import pandas as pd
import os
import matplotlib.pyplot as plt
import mplhep as hep
import pickle
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.utils.inference.pandas_helpers import open_mlpf_dataframe
from src.evaluation.tracks_efficiency import (
    plot_efficiency_all,
)
from src.evaluation.tracks_containment import plot_metrics, obtain_metrics


hep.style.use("CMS")
colors_list = ["#deebf7", "#9ecae1", "#3182bd"]  # color list Jan


log_scale = True

PATH_store = [
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/180324_Zcard/showers_df_evaluation/",
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/180324_Zcard_fullp/showers_df_evaluation/",
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/180324_Zcard_v/showers_df_evaluation/",
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/180324_Zcard_v_full/showers_df_evaluation/",
]
label_list = [
    "180324_Zcard",
    "180324_Zcard_fullp",
    "180324_Zcard_v",
    "180324_Zcard_v_full",
]
PATH_comparison = (
    "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/plots_comparison/"
)
path_hgcal = "0_0_0.pt"


def main():
    list_dataframes = []
    for path in PATH_store:
        sd_hgb, matched_hgb = open_mlpf_dataframe(path + path_hgcal, False)
        list_dataframes.append(sd_hgb)
    # dict_1 = obtain_metrics(matched_hgb)

    plot_efficiency_all(
        list_dataframes, PATH_store, PATH_comparison, label_list, log=log_scale
    )

    # plot_metrics(
    #     dict_1,
    #     PATH_store=PATH_store,
    # )


if __name__ == "__main__":
    main()
