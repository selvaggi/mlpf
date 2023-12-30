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
from utils.inference.plots import plot_metrics, plot_histograms_energy, plot_correction
import matplotlib.pyplot as plt
import mplhep as hep
import torch


def main():

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())


if __name__ == "__main__":
    main()
