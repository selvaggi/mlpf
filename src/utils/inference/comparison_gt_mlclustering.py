from src.utils.inference.event_metrics import get_response_for_event_energy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def plot_mass(sd_hgb, sd_hgb_gt, sd_pandora, PATH_store):


    perfect_pid=False
    mass_zero=False
    ML_pid=True
    matched_all = {"ML": sd_hgb, "ML_GTC": sd_hgb_gt}
    matched_pandora = sd_pandora
    event_res_dic = {} 
    for key in matched_all:
            matched_ = matched_all[key]
            event_res_dic[key] = get_response_for_event_energy(
                    matched_pandora, matched_, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid
                )
    old_font_size = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1, 2,figsize=(16, 8))
    # set fontsize to 20
    ax[0].set_xlabel(r"$m_{pred}/m_{true}$")
    bins = np.linspace(0, 2, 200)
    ax[0].hist(
        event_res_dic["ML"]["mass_over_true_model"],
        bins=bins,
        histtype="step",
        label="ML $\mu$={}".format(
            round((event_res_dic["ML"]["mean_mass_model"]), 4)
        )+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic["ML"]["var_mass_model"]), 4),
        ),
        color="red",
        density=True,
    )
    ax[0].hist(
        event_res_dic["ML_GTC"]["mass_over_true_model"],
        bins=bins,
        histtype="step",
        label="ML GTC $\mu$={}".format(
            round((event_res_dic["ML_GTC"]["mean_mass_model"]), 4)
        )+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic["ML_GTC"]["var_mass_model"]), 4),
        ),
        color="green",
        density=True,
    )


    ax[0].hist(
        event_res_dic["ML"]["mass_over_true_pandora"],
        bins=bins,
        histtype="step",
        label="Pandora $\mu$={}".format(
            round((event_res_dic["ML"]["mean_mass_pandora"]), 4)
        )+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic["ML_GTC"]["var_mass_pandora"]), 4),
        ),
        color="blue",
        density=True,
    )
    ax[0].grid()
    ax[0].legend(loc='upper left')
    #ax.set_xlim([0, 10])
    mean_e_over_true_pandora, sigma_e_over_true_pandora = round(event_res_dic["ML"]["mean_energy_over_true_pandora"], 4), round(
        event_res_dic["ML"]["var_energy_over_true_pandora"], 4)
    mean_e_over_true, sigma_e_over_true = round(event_res_dic["ML"]["mean_energy_over_true"], 4), round(
        event_res_dic["ML"]["var_energy_over_true"], 4)
    mean_e_over_true_gtc, sigma_e_over_true_gtc = round(event_res_dic["ML_GTC"]["mean_energy_over_true"], 4), round(
        event_res_dic["ML_GTC"]["var_energy_over_true"], 4)
    ax[1].hist(event_res_dic["ML"]["energy_over_true"], bins=bins, histtype="step",
                # label=r"ML $\mu$={} $\sigma / \mu$={}".format(mean_e_over_true, sigma_e_over_true),
                color="red",
                label="ML $\mu$={}".format(
                    mean_e_over_true
                )+"\n"+"$\sigma/\mu$={}".format(sigma_e_over_true
                ),
                density=True)
    ax[1].hist(event_res_dic["ML_GTC"]["energy_over_true"], bins=bins, histtype="step",
                # label=r"ML $\mu$={} $\sigma / \mu$={}".format(mean_e_over_true, sigma_e_over_true),
                color="green",
                label="ML GTC $\mu$={}".format(
                    mean_e_over_true_gtc
                )+"\n"+"$\sigma/\mu$={}".format(sigma_e_over_true_gtc
                ),
                density=True)
    ax[1].hist(event_res_dic["ML"]["energy_over_true_pandora"], bins=bins, histtype="step",
                        # label=r"Pandora $\mu$={} $\sigma / \mu$={}".format(mean_e_over_true_pandora,
                        #                                                     sigma_e_over_true_pandora),
                        label="Pandora  $\mu$={}".format(
                            mean_e_over_true_pandora
                        )+"\n"+"$\sigma/\mu$={}".format(sigma_e_over_true_pandora
                        ),
                        color="blue",
                        density=True)

    ax[1].grid(1)
    ax[1].set_xlabel(r"$E_{vis,pred} / E_{vis,true}$")
    ax[1].legend(loc='upper left')

    fig.tight_layout()

    import os
    fig.savefig(os.path.join(PATH_store, "mass_resolution_comp.pdf"), bbox_inches="tight")
    matplotlib.rcParams.update({'font.size': old_font_size})
