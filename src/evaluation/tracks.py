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


log_scale = False

PATH_store = "/eos/user/m/mgarciam/EVAL_REPOS/Tracking_wcoc/models/200124_global_1/showers_df_evaluation/"
path_hgcal = PATH_store + "0_0_0.pt"


def main():

    sd_hgb, matched_hgb = open_mlpf_dataframe(path_hgcal, False)
    dict_1 = obtain_metrics(matched_hgb)

    plot_efficiency_all(sd_hgb, PATH_store, log=True)

    plot_metrics(
        dict_1,
        PATH_store=PATH_store,
    )


if __name__ == "__main__":
    main()
