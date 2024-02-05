import gzip
import pickle
import matplotlib

matplotlib.rc("font", size=35)
import numpy as np
import pandas as pd
import os
import numpy as np
from utils.inference.inference_metrics_hgcal import obtain_metrics_hgcal
from utils.inference.inference_metrics import obtain_metrics
from utils.inference.pandas_helpers import open_hgcal, open_mlpf_dataframe
from utils.inference.plots import (
    plot_metrics,
    plot_histograms_energy,
    plot_correction,
    plot_for_jan,
)
from utils.inference.event_metrics import plot_per_event_metrics
from utils.inference.per_particle_metrics import (
    plot_per_energy_resolution2,
    plot_efficiency_all,
)
import matplotlib.pyplot as plt
import mplhep as hep
import torch
import pickle
from os import listdir
from os.path import isfile, join

hep.style.use("CMS")
# colors_list = mcp.gen_color(cmap="cividis", n=3)
# colors_list = ["#fff7bc", "#fec44f", "#d95f0e"]
# colors_list = ["#fff7bc", "#b82d7dff", "#b143b9ff"]  # color list poster neurips
colors_list = ["#deebf7", "#9ecae1", "#3182bd"]  # color list Jan


def main():
    # list_all_ = ! ls "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/211223_v8/1627/"
    mypath = "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/211223_v8/1627/"
    list_all_ = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    list_all_df_hdb = []
    list_all_df_pandora = []
    for i in list_all_:
        print(i)
        if len(i.split("_")) > 3:
            type = i.split("_")[3]
        else:
            type = ""
        if type == "hdbscan.pt":
            pd_add = pd.read_pickle(mypath + i)
            list_all_df_hdb.append(pd_add)
        elif type == "pandora.pt":
            pd_add = pd.read_pickle(mypath + i)
            list_all_df_pandora.append(pd_add)
    sd_hgb = pd.concat(list_all_df_hdb)
    sd_pandora = pd.concat(list_all_df_pandora)
    PATH_store = "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/211223_v8/1627/"
    plot_per_event_metrics(sd_hgb, sd_pandora, PATH_store=PATH_store)


if __name__ == "__main__":
    main()
