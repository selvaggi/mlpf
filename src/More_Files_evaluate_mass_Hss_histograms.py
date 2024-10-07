import pickle
from src.More_Files_evaluate_mass_Hss import PATH_store
import os
import matplotlib.pyplot as plt
import numpy as np


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


def plot_model(model, label, ax, ax_abs):
    mass_pred = os.path.join(PATH_store, f"mass_{model}.pkl")
    mass_true = os.path.join(PATH_store, f"mass_true_{model}.pkl")
    m_model = load_dict(mass_pred)
    m_true_model = load_dict(mass_true)
    bins = np.linspace(0,2,500)
    #bins_abs = [50, 51, 52...]
    bins_abs = np.arange(50, 250)# 500)
    ax.hist(m_model/m_true_model, bins=bins, histtype="step",  label=label)
    ax.hist(m_model / m_true_model, bins=bins, histtype="step", label=label)
    ax.legend()
    ax_abs.hist(m_model, bins=bins_abs, histtype="step", label=label)
    ax_abs.legend()
    #fig.show()

fig, ax = plt.subplots()
fig_abs, ax_abs = plt.subplots()

plot_model("model", "Model", ax, ax_abs)
plot_model("pandora", "Pandora", ax, ax_abs)
ax.set_xlabel("Mpred/Mtrue")
ax.set_ylabel("Frequency")
ax_abs.set_xlabel("Mpred")
ax_abs.set_ylabel("Frequency")

fig.show()
fig_abs.show()



print("Done")

