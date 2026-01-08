from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
from src.utils.pid_conversion import our_to_pandora_mapping, pandora_to_our_mapping, pid_conversion_dict
import numpy as np 
import matplotlib.pyplot as plt
import os 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
def create_eff_dic_pandora(matched_pandora, id):
    our_id = pandora_to_our_mapping[id]
    id_group = our_to_pandora_mapping[our_id]
    mask_id_true = matched_pandora.pid.isin(id_group)
    matched_pandora_id = matched_pandora[mask_id_true]
    # calculate eff without pid (pure clustering)
    eff_p, energy_eff_p, errors_p = calculate_eff(matched_pandora_id, False, pandora=True )
    
    #calculate efficiency taking into account pid 
    pandora_pid_mask = ~(matched_pandora_id.pandora_pid.isin(id_group))
    matched_pandora_id.loc[pandora_pid_mask,"pandora_calibrated_E"]=np.nan
    eff_p_pid, energy_eff_p, errors_p_pid = calculate_eff(matched_pandora_id, False, pandora=True )
    eff_p_pid_status1, energy_eff_status1_p, errors_p_pid_status1 = calculate_eff(matched_pandora_id[matched_pandora_id.gen_status==1], False, pandora=True )
    eff_p_pid_track, energy_eff_track_p, errors_p_pid_track = calculate_eff(matched_pandora_id[matched_pandora_id.is_track_in_MC==1], False, pandora=True )
    
    fakes_p, energy_fakes_p, fake_percent_energy, fake_percent_reco, fake_errors = calculate_fakes(matched_pandora, None, False, pandora=True, id=id)
    photons_dic = {}
    photons_dic["eff_p"] = eff_p
    photons_dic["eff_p_pid"] = eff_p_pid
    photons_dic["eff_p_status1"] = eff_p_pid_status1
    photons_dic["eff_p_track"] = eff_p_pid_track
    photons_dic["errors_p_pid"]= errors_p_pid
    photons_dic["errors_p_status1"]= errors_p_pid_status1
    photons_dic["errors_p_tracj"]= errors_p_pid_track
    photons_dic["errors_p"]= errors_p
    photons_dic["eff_p"] = eff_p
    photons_dic["energy_eff_p"] = energy_eff_p
    photons_dic["energy_eff_p_status1"] = energy_eff_status1_p
    photons_dic["energy_eff_p_track"] = energy_eff_track_p
    photons_dic["fakes_p"] = fakes_p
    photons_dic["fakes_errors_p"]  = fake_errors
    photons_dic["energy_fakes_p"] = energy_fakes_p
    photons_dic["fake_percent_energy_p"] = fake_percent_energy
    photons_dic["fake_percent_energy_reco_p"] = fake_percent_reco
    return photons_dic



def create_eff_dic(photons_dic, matched_, id, var_i, calc_fakes=True):
    pids = np.abs(matched_["pid"].values)
    our_id = pandora_to_our_mapping[id]
    id_group = our_to_pandora_mapping[our_id]
    mask_id_gt = matched_.pid.isin(id_group)
    matched_id = matched_[mask_id_gt]
    # calculate eff without pid (pure clustering)
    eff, energy_eff, errors = calculate_eff(matched_id, False)
  
    #calculate efficiency taking into account pid 
    matched_id.loc[matched_id.pred_pid_matched!=our_id,"pred_showers_E"]=np.nan
    eff_pid, energy_eff, errors_pid = calculate_eff(matched_id, False)
    eff_pid_status1, energy_eff_status1, errors_pid_status1 = calculate_eff(matched_id[matched_id.gen_status==1], False)
    eff_pid_track, energy_eff_track, errors_pid_track = calculate_eff(matched_id[matched_id.is_track_in_MC==1], False)
    photons_dic["eff_pid_" + str(var_i)] = eff_pid
    photons_dic["eff_status1_" + str(var_i)] = eff_pid_status1
    photons_dic["eff_track_" + str(var_i)] = eff_pid_track
    photons_dic["eff_" + str(var_i)] = eff
    photons_dic["errors_pid_" + str(var_i)] = errors_pid
    photons_dic["errors_status1_" + str(var_i)] = errors_pid_status1
    photons_dic["errors_track_" + str(var_i)] = errors_pid_track
    photons_dic["errors_" + str(var_i)] = errors
    photons_dic["energy_eff_" + str(var_i)] = energy_eff
    photons_dic["energy_eff_status1_" + str(var_i)] = energy_eff_status1
    photons_dic["energy_eff_track_" + str(var_i)] = energy_eff_track

    if calc_fakes:
        fakes, energy_fakes, fake_percent_energy, fake_percent_reco, fake_errors = calculate_fakes(matched_, None, False, pandora=False, id=id)
        photons_dic["fakes_" + str(var_i)] = fakes
        photons_dic["fakes_errors" + str(var_i)] = fake_errors
        photons_dic["energy_fakes_" + str(var_i)] = energy_fakes
        photons_dic["fake_percent_energy_" + str(var_i)] = fake_percent_energy
        photons_dic["fake_percent_energy_reco_" + str(var_i)] = fake_percent_reco
    return photons_dic


def create_fakes_dic(photons_dic, matched_, id, var_i):
    pids = np.abs(matched_["pid"].values)
    mask_id = pids == id
    our_to_pandora_mapping
    df_id = matched_[mask_id]
    eff, energy_eff, errors = calculate_eff(df_id, False)
    fakes, energy_fakes, fake_percent_energy, fake_percent_reco = calculate_fakes(df_id, None, False, pandora=False, id=id)
    photons_dic["eff_" + str(var_i)] = eff
    photons_dic["errors_fakes_" + str(var_i)] = errors
    photons_dic["energy_eff_" + str(var_i)] = energy_eff
    photons_dic["fakes_" + str(var_i)] = fakes
    photons_dic["energy_fakes_" + str(var_i)] = energy_fakes
    return photons_dic

def limit_error_bars(y, yerr, upper_limit=1):
    yerr_upper = np.minimum(y + yerr, upper_limit) - y
    yerr_lower = yerr  # Lower error bars remain unchanged
    return yerr_lower, yerr_upper

def plot_eff(title, photons_dic, label1, PATH_store, labels, ax=None, pandora=False, pid=False):
    colors_list = ["red", "green", "blue", "black"]
    savefig = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    j = 0
    ax.set_xlabel("Energy [GeV]")
    if pid:
        ax.set_ylabel("Efficiency w.PID")
    else:
        ax.set_ylabel("Efficiency")
    ax.set_title(label1)
    ax.grid()
    if pid:
        add = "_pid"
    else:
        add = ""
    for i in range(0, len(labels)):
        if pid:
            print(label1, photons_dic["eff"+ add + "_" + str(i)])
        
        ax.plot(photons_dic["energy_eff_" + str(i)],
            photons_dic["eff"+ add + "_" + str(i)], "--", color=colors_list[i])
        if pid:
            ax.plot(photons_dic["energy_eff_status1_" + str(i)],
            photons_dic["eff_status1" + "_" + str(i)], "--", color=colors_list[i])
            # ax.plot(photons_dic["energy_eff_track_" + str(i)],
            # photons_dic["eff_track" + "_" + str(i)], "-.", color=colors_list[i])
        ax.scatter(
            photons_dic["energy_eff_" + str(i)],
            photons_dic["eff"+ add + "_" + str(i)],
            label=labels[i], # temporarily, for the ML-Pandora comparison plots, change if plotting more labels!
            color=colors_list[i],
            s=50,
            marker="x"
        )
    energy = photons_dic["energy_eff_" + str(i)]
    eff = photons_dic["eff"+ add + "_" + str(i)]
    error_y = photons_dic["errors"+ add + "_" + str(i)]
    yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
    ax.errorbar(energy, eff ,yerr= [yerr_lower, yerr_upper], ecolor=colors_list[i], linestyle='none', capsize=4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if pandora:
        if pid:
            ax.plot(photons_dic["energy_eff_p_status1"],
            photons_dic["eff_p_status1"], "--", color=colors_list[2])
            # ax.plot(photons_dic["energy_eff_p_track"],photons_dic["eff_p_track"], "-.", color=colors_list[3])
        ax.plot(photons_dic["energy_eff_p"],
            photons_dic["eff_p"+add], "--", color=colors_list[2])
        ax.scatter(
            photons_dic["energy_eff_p"],
            photons_dic["eff_p"+add],
            #facecolors=colors_list[2],
            #edgecolors=colors_list[2],
            color=colors_list[2],
            label="Pandora",
            # Add -- line
            s=50,
            marker="x"
        )
        energy = photons_dic["energy_eff_p"]
        eff = photons_dic["eff_p"+add]
        error_y = photons_dic["errors_p"+add]
        yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
        ax.errorbar(energy, eff ,yerr= [yerr_lower, yerr_upper], ecolor=colors_list[2], linestyle='none', capsize=4)

    ax.legend(loc="lower right")
    # ax.set_xscale("log")
    if savefig:
        fig.savefig(
            os.path.join(PATH_store, "Efficiency_" + label1 + ".pdf"),
            bbox_inches="tight",
        )
    else:
        plot_eff(title, photons_dic, label1, PATH_store, labels, ax=None)

def plot_eff_and_fakes(title, photons_dic, label1, PATH_store, labels):
    colors_list = ["#FF0000",  "#00FF00", "#0000FF"]
    fig, ax = plt.subplots()
    j = 0
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Efficiency")
    # ax[row_i, j].set_xscale("log")
    ax.set_title(title)
    ax.grid(1)
    for i in range(0, len(labels)):
        ax.plot(photons_dic["energy_eff_" + str(i)],
            photons_dic["eff_" + str(i)], "--", color=colors_list[i])
        ax.scatter(
            photons_dic["energy_eff_" + str(i)],
            photons_dic["eff_" + str(i)],
            label="ML " + label1, # Temporarily, for the ML-Pandora comparison plots, change if plotting more labels!
            color=colors[labels[i]],
            s=50,
        )
    ax.plot(photons_dic["energy_eff_p"],
        photons_dic["eff_p"], "--", color=colors_list[2])
    ax.scatter(
        photons_dic["energy_eff_p"],
        photons_dic["eff_p"],
        facecolors=colors_list[2],
        edgecolors=colors_list[2],
        label="Pandora " + label1,
        s=50,
    )
    ax.legend(loc="upper right")
    if title == "Electromagnetic":
        ax.set_ylim([0.5, 1.1])
    else:
        ax.set_ylim([0.5, 1.1])
    ax.set_xscale("log")
    ax.set_xlabel("Efficiency")
    ax_fakes = inset_axes(ax,
                            width="50%",  # width = 30% of parent_bbox
                            height="40%",  # height : 1 inch
                            loc="lower right")
    ax_fakes.set_ylabel("Fake rate")
    for i in range(0, len(labels)):
        ax_fakes.plot(photons_dic["energy_fakes_" + str(i)],
            photons_dic["fakes_" + str(i)], "--", color=colors_list[0])
        ax_fakes.scatter(
            photons_dic["energy_fakes_" + str(i)],
            photons_dic["fakes_" + str(i)],
            label="ML", # Temporarily, for the ML-Pandora comparison plots, change if plotting more labels!
            color=colors_list[0],
            s=50,
        )
    ax_fakes.grid()
    ax_fakes.set_xlabel("Energy [GeV]")
    ax_fakes.plot(photons_dic["energy_fakes_p"],
        photons_dic["fakes_p"], "--", color=colors_list[2])
    ax_fakes.scatter(
        photons_dic["energy_fakes_p"],
        photons_dic["fakes_p"],
        facecolors=colors_list[2],
        edgecolors=colors_list[2],
        label="Pandora",
        # add -- line
        s=50,
    )
    fig.savefig(
        os.path.join(PATH_store, "DoublePlot_" + title + label1 + ".pdf"),
        bbox_inches="tight",
    )


def plot_fakes_E(title, photons_dic, label1, PATH_store, labels, ax=None, reco=""): # Set reco to 'reco_'
    colors_list = ["red",  "green", "blue"]
    savefig = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    j = 0
    ax.set_xlabel("Energy [GeV]")
    if "reco" in reco:
        ax.set_ylabel("Fake reco energy rate")
    else:
        ax.set_ylabel("Fake energy rate")
    # ax[row_i, j].set_xscale("log")
    ax.set_title(label1)
    ax.grid()
    for i in range(0, len(labels)):
        ax.plot(photons_dic["energy_fakes_" + str(i)],
            photons_dic["fake_percent_energy_" + reco + str(i)], "--", color=colors_list[i])
        ax.scatter(
            photons_dic["energy_fakes_" + str(i)],
            photons_dic["fake_percent_energy_" + reco + str(i)],
            label=labels[i], # Temporarily, for the ML-Pandora comparison plots, change if plotting more labels!
            color=colors_list[i],
            s=50,
            marker="x"
        )
    ax.plot(photons_dic["energy_fakes_p"],
        photons_dic["fake_percent_energy_" + reco + "p"], "--", color=colors_list[2])
    ax.scatter(
        photons_dic["energy_fakes_p"],
        photons_dic["fake_percent_energy_" + reco + "p"],
        facecolors=colors_list[2],
        edgecolors=colors_list[2],
        label="Pandora",
        s=50,
        marker="x"
    )
    ax.legend(loc="upper right")
    #if title == "Electromagnetic":
    #    plt.ylim([0.0, 0.5])
    #else:
    #    plt.ylim([0.0, 0.5])
    ax.set_xscale("log")
    if savefig:
        fig.savefig(
            os.path.join(PATH_store, "Fake_Energy_Rate_"  + reco + label1 + ".pdf"),
            bbox_inches="tight",
        )
    #else:
    #    plot_fakes_E(title, photons_dic, label1, PATH_store, labels, ax=None)

def plot_fakes(title, photons_dic, label1, PATH_store, labels, ax=None):
    colors_list = ["red",  "green", "blue"]
    savefig = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    j = 0
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Fake rate")
    ax.set_title(label1)
    ax.grid()
    for i in range(0, len(labels)):
        ax.plot(photons_dic["energy_fakes_" + str(i)],
            photons_dic["fakes_" + str(i)], "--", color=colors_list[i])
        ax.scatter(
            photons_dic["energy_fakes_" + str(i)],
            photons_dic["fakes_" + str(i)],
            label=labels[i], # Temporarily, for the ML-Pandora comparison plots, change if plotting more labels!
            color=colors_list[i],
            s=50,
            marker="x"
        )
    ax.plot(photons_dic["energy_fakes_p"],
        photons_dic["fakes_p"], "--", color=colors_list[2])
    ax.scatter(
        photons_dic["energy_fakes_p"],
        photons_dic["fakes_p"],
        facecolors=colors_list[2],
        edgecolors=colors_list[2],
        label="Pandora",
        marker="x",
        # add -- line
        s=50,

    )
    energy = photons_dic["energy_fakes_" + str(i)]
    eff = photons_dic["fakes_" + str(i)]
    error_y = photons_dic["fakes_errors"+ str(i)]
    # yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
    error_y = np.array(error_y)/2
    ax.errorbar(energy, eff ,yerr= [error_y, error_y], ecolor=colors_list[i], linestyle='none', capsize=4)

    energy = photons_dic["energy_fakes_p"]
    eff = photons_dic["fakes_p"]
    error_y = photons_dic["fakes_errors_p"]
    # yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
    error_y = np.array(error_y)/2
    ax.errorbar(energy, eff ,yerr= [error_y, error_y], ecolor=colors_list[2], linestyle='none', capsize=4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.legend()
    #if title == "Electromagnetic":
    #    plt.ylim([0.0, 0.07])
    #else:
    #    plt.ylim([0.0, 0.07])
    # ax.set_xscale("log")
    if savefig:
        fig.savefig(
            os.path.join(PATH_store, "Fake_Rate_" + label1 + ".pdf"),
            bbox_inches="tight",
        )
    else:
        plot_fakes(title, photons_dic, label1, PATH_store, labels, ax=None)




def plot_efficiency_all(sd_pandora, df_list, PATH_store, labels, ax=None):
    import matplotlib
    matplotlib.rcParams["font.size"] = 15
    # Particle configuration
    particles = {
        "photons":   {"pid": 22,  "name": "Photons",           "group": "Electromagnetic", "row": 0},
        "electrons": {"pid": 11,  "name": "Electrons",         "group": "Electromagnetic", "row": 3},
        "pions":     {"pid": 211, "name": "Charged hadrons",   "group": "Hadronic",        "row": 1},
        "kaons":     {"pid": 130, "name": "Neutral hadrons",   "group": "Hadronic",        "row": 2},
        "muons":     {"pid": 13,  "name": "Muons",             "group": "Hadronic",        "row": 4},
    }
    # Initialise dictionaries
    calc_fakes = sd_pandora is not None
    pandora = calc_fakes
    eff_dic = {
        k: create_eff_dic_pandora(sd_pandora, v["pid"]) if pandora else {}
        for k, v in particles.items()
    }
    # Fill dictionaries from HGB dataframes
    for var_i, sd_hgb in enumerate(df_list):
        for k, v in particles.items():
            eff_dic[k] = create_eff_dic(
                eff_dic[k], sd_hgb, v["pid"],
                var_i=var_i,
                calc_fakes=calc_fakes
            )

    # Helper for repeated plotting
    def plot_block(group, dic, name, row):
        plot_eff(group, dic, name, PATH_store, labels, ax=ax[row, 0], pandora=pandora)
        plot_eff(group, dic, name, PATH_store, labels, ax=ax[row, 1], pandora=pandora, pid=True)

        if calc_fakes:
            plot_fakes(group, dic, name, PATH_store, labels, ax=ax[row, 2])
            plot_fakes_E(group, dic, name, PATH_store, labels, ax=ax[row, 3])
            plot_fakes_E(group, dic, name, PATH_store, labels, ax=ax[row, 4], reco="reco_")

    # Plot everything
    for k, cfg in particles.items():
        dic = eff_dic[k]
        if dic.get("eff_0") and len(dic["eff_0"]) > 0:
            plot_block(cfg["group"], dic, cfg["name"], cfg["row"])




def calculate_eff(sd, log_scale=False, pandora=False, pid=False):
    if log_scale:
        bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    else:
        bins = [0, 5, 10, 35, 50]
    eff = []
    energy_eff = []
    errors = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = sd.reco_showers_E.values <= bin_i1
        mask_below = sd.reco_showers_E.values > bin_i
        mask = mask_below * mask_above
        
        number_of_non_reconstructed_showers = np.sum(
            np.isnan(sd.pred_showers_E.values)[mask]
        )
        total_showers = len(sd.true_showers_E.values[mask])
        if pandora:
            number_of_non_reconstructed_showers = np.sum(
                np.isnan(sd.pandora_calibrated_E.values)[mask]
            )
            total_showers = len(sd.pandora_calibrated_E.values[mask])
        if total_showers > 0:
            eff.append(
                (total_showers - number_of_non_reconstructed_showers) / total_showers
            )
            energy_eff.append((bin_i1 + bin_i) / 2)
            n_total = total_showers
            n_r = total_showers-number_of_non_reconstructed_showers
            error = (n_r/(n_total**2)*np.sqrt(n_total))**2+(1/n_total*np.sqrt(n_r))**2
            error = np.sqrt(error)
            errors.append(error)
    return eff, energy_eff, errors


def calculate_fakes(sd, matched, log_scale=False, pandora=False, id=None):
    if log_scale:
        bins_fakes = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    else:
        bins_fakes = [0, 5, 15, 35, 50]
    fake_rate = []
    energy_fakes = []
    fake_percent_energy = []
    fake_percent_reco_energy = []
    id_our = pandora_to_our_mapping[id]
    fake_errors = []
    for i in range(len(bins_fakes) - 1):
        bin_i = bins_fakes[i]
        bin_i1 = bins_fakes[i + 1]
        if pandora:
            mask_above = sd.pred_showers_E.values <= bin_i1
            mask_below = sd.pred_showers_E.values > bin_i
            mask_pid = sd.pandora_pid.isin(our_to_pandora_mapping[id_our])
            mask_pid_truth = mask_above * mask_below * sd.pid.isin(our_to_pandora_mapping[id_our]) # The matched ones
            mask = mask_below * mask_above * mask_pid
            fakes = np.sum(np.isnan(sd.pid)[mask])
            non_fakes_mask = ~np.isnan(sd.pid)[mask]
            fakes_mask = np.isnan(sd.pid)[mask]
            energy_in_fakes = np.sum(sd.pandora_calibrated_pfo[mask].values[fakes_mask])
            reco_in_fakes = np.sum(sd.pred_showers_E[mask].values[fakes_mask])
            total_E_meas = np.sum(sd.pandora_calibrated_pfo.values[mask])
            total_E_reco = np.sum(sd.pred_showers_E.values[mask])
            total_showers = len(sd.pred_showers_E.values[mask]) # The true showers
        else:
            mask_above = sd.pred_showers_E.values <= bin_i1
            mask_below = sd.pred_showers_E.values > bin_i
            mask_pid = sd.pred_pid_matched == id_our
            mask_pid_truth = sd.pid.isin(our_to_pandora_mapping[id_our]) # The matched ones!
            mask = mask_below * mask_above * mask_pid
            fakes = np.sum(np.isnan(sd.pid)[mask])
            total_showers_true = len(sd.pred_showers_E.values[mask_pid_truth])
            total_showers = sum(~np.isnan(sd.pred_showers_E.values[mask]))
            fakes_mask = np.isnan(sd.pid)[mask]
            energy_in_fakes = np.sum(sd.calibrated_E[mask].values[fakes_mask])
            reco_in_fakes = np.sum(sd.pred_showers_E[mask].values[fakes_mask])
            #non_fakes_mask = ~np.isnan(sd.pid)[mask]
            total_E_meas = np.sum(sd.calibrated_E.values[mask])
            total_E_reco = np.sum(sd.pred_showers_E.values[mask])
        if total_showers > 0:
            fake_rate.append(fakes / total_showers)
            n_r = fakes
            n_total = total_showers
            error = (n_r/(n_total**2)*np.sqrt(n_total))**2+(1/n_total*np.sqrt(n_r))**2
            error = np.sqrt(error)
            fake_errors.append(error)
            energy_fakes.append((bin_i1 + bin_i) / 2)
            fake_percent_energy.append(energy_in_fakes / total_E_meas)
            fake_percent_reco_energy.append(reco_in_fakes / total_E_reco)
    return fake_rate, energy_fakes, fake_percent_energy, fake_percent_reco_energy, fake_errors 

