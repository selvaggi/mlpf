import wandb
import numpy as np
import mplhep as hep

hep.style.use("CMS")
import matplotlib

matplotlib.rc('font', size=13)

# %%
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import xgboost

os.environ.get("LD_LIBRARY_PATH")
# %%
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np
import torch

print("CUDA available:", torch.cuda.is_available())  # in case needed

DEVICE = torch.device("cuda:0")


def get_dataset(cap=None):
    path = "/afs/cern.ch/work/g/gkrzmanc/mlpf_results/clustering_gt_with_pid_and_mean_features/cluster_features"
    r = {}
    n = 0
    # nmax = 257
    for file in os.listdir(path):
        n += 1
        if cap is not None and n > cap:
            break
        f = pickle.load(open(os.path.join(path, file), "rb"))
        for key in f:
            if key not in r:
                r[key] = f[key]
            else:
                r[key] = torch.concatenate((r[key], f[key]), axis=0)
    x_names = ["ecal_E", "hcal_E", "num_hits", "track_p", "ecal_dispersion", "hcal_dispersion", "sum_e", "num_tracks"]
    h_names = ["hit_x_avg", "hit_y_avg", "hit_z_avg"]
    h1_names = ["hit_eta_avg", "hit_phi_avg"]
    print("x shape:", r["x"].shape)
    xyz = r["node_features_avg"][:, [0, 1, 2]].cpu()
    eta_phi = torch.stack([calculate_eta(xyz[:, 0], xyz[:, 1], xyz[:, 2]), calculate_phi(xyz[:, 0], xyz[:, 1])], dim=1)
    return torch.concatenate([r["x"], xyz, eta_phi], dim=1), x_names + h_names + h1_names, r["true_e_corr"], r[
        "e_true"], r["e_reco"], r["y_particles"][:, 6]


def get_split(ds, overfit=True, same_train_test=True):
    from sklearn.model_selection import train_test_split
    x, _, y, etrue, _, pids = ds
    xtrain, xtest, ytrain, ytest, energiestrain, energiestest, pid_train, pid_test = train_test_split(
        x, y, etrue, pids, test_size=0.2, random_state=42
    )
    #if overfit:
    #    k = 2000
    #    return xtrain[:k], xtrain[:k], ytrain[:k], ytrain[:k], energiestrain[:k], energiestrain[:k], pid_train[:k], pid_train[:k]
    if same_train_test:
        return xtrain, xtrain, ytrain, ytrain, energiestrain, energiestrain, pid_train, pid_train
    return xtrain, xtest, ytrain, ytest, energiestrain, energiestest, pid_train, pid_test

def get_gb():
    # from sklearn.ensemble import GradientBoostingRegressor
    # model = GradientBoostingRegressor(verbose=1, max_depth=7, n_estimators=1000)
    # model = catboost.CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=True, task_type="GPU", devices=DEVICE)
    # xgboost model
    model = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, learning_rate=0.05, verbosity=1,
                                 tree_method="gpu_hist", gpu_id=DEVICE)
    return model


def calculate_phi(x, y):
    return np.arctan2(y, x)


def calculate_eta(x, y, z):
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    return -np.log(np.tan(theta / 2))


def get_gb():
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(verbose=1, max_depth=7, n_estimators=1000)
    return model

def get_nn(patience):
    # pytorch impl. of a neural network
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.model = nn.ModuleList([
               nn.BatchNorm1d(13),
                nn.Linear(13, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)]
            )
            '''self.model = nn.ModuleList([
                nn.Linear(13, 1, bias=False)
            ])'''

        def forward(self, x):
            for layer in self.model:
                x = layer(x)
            return x

        def freeze_batchnorm(self):
            for layer in self.model:
                if isinstance(layer, nn.BatchNorm1d):
                    layer.eval()
                    print("Frozen batchnorm in 1st layer only - ", layer)
                    break

    class NetWrapper():
        def __init__(self):
            pass

        def predict(self, x):
            x = torch.tensor(x).to(DEVICE)
            self.model.eval()
            with torch.no_grad():
                return self.model(x).cpu().numpy().flatten()

        def fit(self, x, y):
            print("---> Fit - x.shape", x.shape, " y.shape", y.shape)
            x = torch.tensor(x).to(DEVICE)
            y = torch.tensor(y).to(DEVICE)
            self.model = Net()
            self.model.to(DEVICE)
            self.model.train()
            batch_size = 32
            #optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            # add weight decay
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)
            criterion = nn.MSELoss()
            tolerance = 1e-3
            epochs = 1000
            # calc loss of last 100 batches
            best_loss = 1e9
            patience_counter = 0
            freeze_bn_steps = 1000
            losses_all = []
            epoch_losses = []
            for epoch in range(epochs):
                losses_this_epoch = []
                for i in (pbar := tqdm.tqdm(range(0, len(x)))):
                    if i == freeze_bn_steps:
                        self.model.freeze_batchnorm()
                    xbatch = x[i:i + batch_size].to(DEVICE)
                    ybatch = y[i:i + batch_size].to(DEVICE)
                    optimizer.zero_grad()
                    ypred = self.model(xbatch)
                    loss = criterion(ypred.flatten(), ybatch)
                    loss.backward()
                    losses_all.append(loss.item())
                    losses_this_epoch.append(loss.item())
                    loss_running_mean = np.mean(losses_all[-4000:])
                    #pbar.set_description(
                    #    "Loss: " + str(round(loss.item(), 3)) + " / running loss: " + str(round(loss_running_mean, 3)))
                    optimizer.step()
                    if loss_running_mean < best_loss - tolerance:
                        best_loss = loss_running_mean
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter > patience:
                            print("Early stopping at running mean loss:", loss_running_mean)
                            break
                if patience_counter > patience:
                    break
                epoch_losses.append(np.mean(losses_this_epoch))
                if epoch % 10 == 0:
                    print("Epoch", epoch, "loss:", np.mean(losses_this_epoch))
            return losses_all
    return NetWrapper()


def main(ds, train_only_on_tracks=False, train_only_on_neutral=False, train_energy_regression=False,
         train_only_on_PIDs=[], remove_sum_e=False, use_model="gradboost", patience=1000, return_data=False):
    split = list(get_split(ds))
    model = get_nn(patience=patience)
    #model = get_gb()

    # elif use_model == "gradboost1":
    #    # gradboost with more depth, longer training
    #    from sklearn.ensemble import GradientBoostingRegressor

    #    model = GradientBoostingRegressor(verbose=1, max_depth=7, n_estimators=1000)

    if train_only_on_tracks:
        mask = (split[0][:, 3] > 0) & (split[0][:, 7] == 1)
        split[0] = split[0][mask]
        split[2] = split[2][mask]
        split[4] = split[4][mask]
    elif train_only_on_neutral:
        mask = (split[0][:, 3] == 0) & (split[0][:, 7] == 0)
        split[0] = split[0][mask]
        split[2] = split[2][mask]
        split[4] = split[4][mask]
    elif train_only_on_PIDs:
        print("getting mask")
        mask = [i in train_only_on_PIDs for i in split[6].tolist()]
        masktest = [i in train_only_on_PIDs for i in split[7].tolist()]
        print("got mask")
        mask = torch.tensor(mask)
        print("Mask covers this fraction:", mask.float().mean().item())
        split[0] = split[0][mask]
        split[1] = split[1][masktest]
        split[2] = split[2][mask]
        split[3] = split[3][masktest]
        split[4] = split[4][mask]
        split[5] = split[5][masktest]
        split[6] = split[6][mask]
        split[7] = split[7][masktest]
        CROP_DS = True
        if CROP_DS:
            k = 1000
            split[0] = split[0][:k]
            split[1] = split[1][:k]
            split[2] = split[2][:k]
            split[3] = split[3][:k]
            split[4] = split[4][:k]
            split[5] = split[5][:k]
            split[6] = split[6][:k]
            split[7] = split[7][:k]


    if remove_sum_e:
        split[0][:, 6] = 0.0  # remove the sum of the hits
    if not train_energy_regression:
        print("Fitting")
        result = model.fit(split[0].numpy(), split[2].numpy())
        # print("Fitted model:", result)
        # validation
        ysum = split[1][:, 6]
        ypred = model.predict(split[1].numpy())
        epred = ysum * (1 + ypred)
        ytrue = split[3]
        energies = split[5]
        return ytrue, epred, energies, split[1], model, split, result
    else:
        print("Fitting")
        result = model.fit(split[0].numpy(), split[4].numpy())
        # print("Fitted model:", result)
        # validation
        epred = model.predict(split[1].numpy())
        ytrue = split[5]
        ysum = split[1][:, 6]
        ypred = epred / ysum - 1
        energies = split[5]
        return ytrue, epred, ytrue, split[1], model, split, result

        # log scatterplots of validation results per energy

# %%
ds = get_dataset(cap=200)
print("Loaded dataset")
# %%
yt, yp, en, _, model, split, result = main(ds=ds,
                                           train_energy_regression=True,
                                           train_only_on_PIDs=[211],
                                           remove_sum_e=False,
                                           patience=15000)
#len(ds)

#%%
fig, ax = plt.subplots()
ax.scatter(yt, yp)
p_tracks = split[1][:, 3].clone().detach().cpu().numpy()
ax.scatter(yt, p_tracks, color="red")
# add a gray diag. line
ax.plot([0,max(yt)], [0, max(yt)], "--", color="gray")
# plot tracs
ax.set_ylabel("Predicted energy")
ax.set_xlabel("True energy")
plt.show()


#%%
# momentum and weight decay



# %%
import numpy as np


def get_std68(theHist, bin_edges, percentage=0.683, epsilon=0.01):
    # theHist, bin_edges = np.histogram(data_for_hist, bins=bins, density=True)
    wmin = 0.2
    wmax = 1.0

    weight = 0.0
    points = []
    sums = []

    # fill list of bin centers and the integral up to those point
    for i in range(len(bin_edges) - 1):
        weight += theHist[i] * (bin_edges[i + 1] - bin_edges[i])
        points.append([(bin_edges[i + 1] + bin_edges[i]) / 2, weight])
        sums.append(weight)
    low = wmin
    high = wmax
    width = 100
    for i in range(len(points)):
        for j in range(i, len(points)):
            wy = points[j][1] - points[i][1]
            if abs(wy - percentage) < epsilon:
                wx = points[j][0] - points[i][0]
                if wx < width:
                    low = points[i][0]
                    high = points[j][0]
                    width = wx

    return 0.5 * (high - low), low, high


def mean_without_outliers(data):
    remove_count = int(len(data) * 0.01)
    # Sort the array
    sorted_arr = np.sort(data)
    # Remove the lowest and highest 1% of the elements
    trimmed_arr = sorted_arr[remove_count:-remove_count]
    # Calculate the mean of the trimmed array
    mean = np.mean(trimmed_arr)
    return mean


def obtain_MPV_and_68_raw(data_for_hist, bins_per_binned_E=np.arange(-1, 5, 0.01), epsilon=0.01):
    hist, bin_edges = np.histogram(data_for_hist, bins=bins_per_binned_E, density=True)
    ind_max_hist = np.argmax(hist)
    # MPV = (bin_edges[ind_max_hist] + bin_edges[ind_max_hist + 1]) / 2
    std68, low, high = get_std68(hist, bin_edges, epsilon=epsilon)
    MPV = mean_without_outliers(data_for_hist)
    return MPV, std68, low, high


def obtain_MPV_and_68(data_for_hist, *args, **kwargs):
    # trim the data for hist by removing the top and bottom 1%
    if len(data_for_hist) != 0:
        data_for_hist = data_for_hist[
            (data_for_hist > np.percentile(data_for_hist, 1)) & (data_for_hist < np.percentile(data_for_hist, 99))]
    # bins_per_binned_E = np.linspace(data_for_hist.min(), data_for_hist.max(), 1000)
    bins_per_binned_E = np.arange(0, 2, 1e-3)
    return obtain_MPV_and_68_raw(data_for_hist, bins_per_binned_E)


# %%
def get_charged_response_resol_plot_for_PID(pid, yt, yp, en, model, split, neutral=False):
    e_thresholds = [0, 6, 12, 18, 24, 30, 36, 42, 48]  # True E thresholds!
    mpvs_model, s68s_model = [], []
    mpvs_pandora, s68s_pandora = [], []
    mpvs_sum_hits, s68s_sum_hits = [], []
    e_true = (1 + yt) * split[1][:, 6].numpy()
    e_pred = yp
    frac_pred = e_pred / e_true
    frac_e_sum = split[1][:, 6].clone().detach().cpu().numpy() / e_true
    e_track = split[1][:, 3].clone().detach().cpu().numpy()
    frac_track = e_track / e_true
    track_filter = ((split[1][:, 3] > 0) & (split[1][:, 7] == 1))
    if neutral:
        track_filter = ~track_filter
        track_filter = track_filter & (split[1][:, 7] == 0)
    track_filter = track_filter & (split[-1] == pid).cpu()
    binsize = 0.01
    bins_x = []

    for i, e_threshold in enumerate(e_thresholds):
        if i == 0:
            continue
        bins_x.append(0.5 * (e_thresholds[i] + e_thresholds[i - 1]))
        filt_energy = (e_true < e_thresholds[i]) & (e_true >= e_thresholds[i - 1])
        mpv, s68, lo, hi = obtain_MPV_and_68(frac_pred[filt_energy & track_filter],
                                             bins_per_binned_E=np.arange(0, 5, binsize))
        mpvs_model.append(mpv)
        s68s_model.append(s68)
        mpv, s68, lo, hi = obtain_MPV_and_68(frac_track[filt_energy & track_filter].clip(max=5),
                                             bins_per_binned_E=np.arange(0, 5, binsize))
        mpvs_pandora.append(mpv)
        s68s_pandora.append(s68)
        mpv, s68, _, _ = obtain_MPV_and_68(frac_e_sum[filt_energy & track_filter],
                                           bins_per_binned_E=np.arange(0, 5, binsize))
        mpvs_sum_hits.append(mpv)
        s68s_sum_hits.append(s68)

    fig, ax = plt.subplots(2, 1, figsize=(7, 4), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})  # Height of 2 subplots.
    ax[0].plot(bins_x, np.array(s68s_model) / np.array(mpvs_model), ".--", label="model")
    ax[0].plot(bins_x, np.array(s68s_pandora) / np.array(mpvs_pandora), ".--", label="track p")
    ax[0].plot(bins_x, np.array(s68s_sum_hits) / np.array(mpvs_sum_hits), ".--", label="sum hits")
    ax[0].legend()
    ax[1].set_xlabel("Energy [GeV]")
    ax[0].set_ylabel("σ / E")
    # ax[0].set_ylim([0, 0.4])
    # ax[0].set_ylim([0, 0.4])
    # ax[0].set_ylim([0, 0.4])
    ax[1].plot(bins_x, mpvs_model, ".--", label="GradBoost")
    ax[1].plot(bins_x, mpvs_pandora, ".--", label="track p")
    ax[1].plot(bins_x, mpvs_sum_hits, ".--", label="sum hits")
    ax[1].set_ylim([0.95, 1.05])
    ax[1].set_ylabel("response")
    ax[0].set_title("PID: " + str(pid))
    upper_plot = {"ML": np.array(s68s_model) / np.array(mpvs_model),
                  "p": np.array(s68s_pandora) / np.array(mpvs_pandora),
                  "sum": np.array(s68s_sum_hits) / np.array(mpvs_sum_hits)}
    lower_plot = {"ML": mpvs_model, "p": mpvs_pandora, "sum": mpvs_sum_hits}
    return fig, upper_plot, lower_plot, bins_x



def is_pid_neutral(pid):
    return pid in [22, 130, 2112]


def get_plots(PIDs, energy_regression=False, remove_sum_e=False, use_model="gradboost", patience=1000):
    yt, yp, en, _, model, split, lossfn = main(ds=ds, train_energy_regression=energy_regression, train_only_on_PIDs=PIDs,
                                       remove_sum_e=remove_sum_e, use_model=use_model, patience=patience)
    import shap
    import numpy as np
    # te = shap.TreeExplainer(model)
    # shap_vals_r = te.shap_values(np.array(split[1]))
    x_names = ["ecal_E", "hcal_E", "num_hits", "track_p", "ecal_dispersion", "hcal_dispersion", "sum_e", "num_tracks"]
    h_names = ["hit_x_avg", "hit_y_avg", "hit_z_avg", "eta", "phi"]
    # shap.summary_plot(shap_vals_r, split[1], feature_names=x_names + h_names, use_log_scale=True, show=False)
    # plt.show()
    results = {}
    for pid in PIDs:
        fig, upper, lower, x = get_charged_response_resol_plot_for_PID(pid, yt, yp, en, model, split,
                                                                       neutral=is_pid_neutral(pid))
        results[pid] = [fig, upper, lower, x, model]

    return results, lossfnc

#all_pids = [22, 130, 2112, 211, -211, 2212, -2212]
all_pids = [211]
prefix = "/eos/user/g/gkrzmanc/2024/26_3_small_network_1/"

# get weights of the only layer in the model
weight_matrix = model.model.model[0].weight.detach().cpu().numpy()


# plot loss curve with running average
fig, ax = plt.subplots()
ax.plot(lossfn)
# now add running average
k = 5000
running_mean = np.convolve(lossfn, np.ones(k) / k, mode='valid')

ax.plot(np.arange(k, len(lossfn)), running_mean[:-1], label="Running average")
ax.set_ylabel("Loss")
ax.set_xlabel("Batch")
ax.set_yscale("log")
plt.show()


