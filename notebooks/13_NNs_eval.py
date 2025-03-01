#%%
# add "/afs/cern.ch/work/g/gkrzmanc/mlpf_2024" to PATH - a quick fix
import sys
sys.path.extend("/afs/cern.ch/work/g/gkrzmanc/mlpf_2024") # quick fix
# TODOs: try with old data (reproduce the neutral plots), try loading the "finetuned" model to see if it works better then with new data
import src
import wandb
import numpy as np
import mplhep as hep
import wandb
hep.style.use("CMS")
import matplotlib
matplotlib.rc('font', size=13)

from src.utils.inference.inference_metrics import get_sigma_gaussian
from src.models.GATr.Gatr_pf_e import ExampleWrapper as GravnetModel

# %%
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from PIL import Image

os.environ.get("LD_LIBRARY_PATH")

# %%
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import torch
import argparse

# args: prefix, wandb name, PIDs to train on, loss to use
parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, required=True)
parser.add_argument("--wandb_name", type=str, required=True)
parser.add_argument("--PIDs", type=str, required=True) # comma-separated list of PIDs to train and evaluate on
parser.add_argument("--loss", type=str, default="default") # loss to use
parser.add_argument("--patience", type=int, default=50000) # patience for early stopping
parser.add_argument("--pid-loss", action="store_true")
parser.add_argument("--ecal-hcal-loss", action="store_true")
parser.add_argument("--dataset-path", type=str, default="/afs/cern.ch/work/g/gkrzmanc/mlpf_results/clustering_gt_with_pid_and_mean_features/cluster_features")
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()
prefix = args.prefix
wandb_name = args.wandb_name

all_pids = args.PIDs.split(",")

assert len(all_pids) > 0
all_pids = [int(pid) for pid in all_pids]
print("Training on PIDs:", all_pids)
print("CUDA available:", torch.cuda.is_available())  # in case needed

DEVICE = torch.device("cuda:0")
#prefix = "/eos/user/g/gkrzmanc/2024/1_4_/BS64_train_neutral_l1_loss_only/"

wandb.init(project="mlpf_debug_energy_corr", entity="fcc_ml", name=wandb_name)
# wandb log code
wandb.run.log_code(".")

# make dir
os.makedirs(prefix, exist_ok=True)

def get_eval_fig(ytrue, ypred, step, criterion, p=None):
    # calc losses, and plot loss histogram for  energy (ytrue) ranges [0, 6], [6, 12] etc. You need to filter by ytrue!
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].scatter(ytrue, ypred, alpha=0.2)
    if p is not None:
        ax[0].scatter(ytrue, p, color="red", alpha=0.2)
    ax[0].set_ylabel("Predicted energy")
    ax[0].set_xlabel("True energy")
    acceptable_loss = 1e-2
    rages = [[0, 6], [6, 12], [12, 18], [18, 24], [24, 30], [30, 36], [36, 42], [42, 48], [48, 54], [54, 60]]
    for i, r in enumerate(rages):
        mask = (ytrue >= r[0]) & (ytrue < r[1])
        # % BELOW ACCEPTABLE LOSS
        #frac = torch.mean((((ypred[mask] - ytrue[mask]).abs() < acceptable_loss))) * 100
        #print(frac)
        #print(torch.mean(((ypred[mask] - ytrue[mask]).abs() < acceptable_loss).float()).item())
        frac = torch.mean(((ypred[mask] - ytrue[mask]).abs() < acceptable_loss).float())
        # if is nan, change to 0
        if torch.isnan(frac):
            frac = 0
        if mask.sum() > 0:
            ypred_mask = ypred[mask]
            ytrue_mask = ytrue[mask]
            losses = [criterion(ypred_mask[i], ytrue_mask[i], step).detach().cpu() for i in range(mask.sum())]
            losses = torch.tensor(losses)
            ax[1].hist(torch.clamp(torch.log10(losses), -5, 5), bins=100, alpha=0.5, label=f"{r[0]}-{r[1]} ({str(int(frac*100))}%)")
    if len(ytrue) > 0:
        ax[0].plot([0, max(ytrue)], [0, max(ytrue)], "--", color="gray")
    ax[1].set_xlim([-5, 5])
    ax[1].set_xlabel("log10(loss)")
    # log scale
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Count")
    ax[1].legend()
    return fig

import io

def get_dataset(save_ckpt=None):
    if save_ckpt is not None:
        # check if file exists
        if os.path.exists(save_ckpt):
            print("Loading dataset from", save_ckpt)
            return pickle.load(open(save_ckpt, "rb"))

    path = args.dataset_path
    r = {}
    n = 0
    nmax = 10000000000
    print("Loading dataset (a bit slow, TODO fix)")
    for file in tqdm.tqdm(os.listdir(path)):
        n += 1
        if n > nmax:
            break
        #f = pickle.load(open(os.path.join(path, file), "rb"))
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)
        f = CPU_Unpickler(open(os.path.join(path, file), "rb")).load()
        if (f["e_reco"].flatten() == 1.).all():
            continue  # some old files, ignore them for now
        old_dataset = False
        for key in f:
            if key == "pid_y":
                if key not in r:
                    r[key] = torch.tensor(f[key])
                else:
                    r[key] = torch.concatenate([r[key], torch.tensor(f[key])])
            elif key != "y_particles" or old_dataset:
                if key not in r:
                    r[key] = f[key]
                else:
                    r[key] = torch.concatenate((r[key], f[key]), axis=0)
            else:
                if "pid_y" not in f.keys():
                    if key not in r:
                        r[key] = f[key].pid.flatten()
                    else:
                        r[key] = torch.concatenate((r[key], f[key].pid.flatten()), axis=0)
    x_names = ["ecal_E", "hcal_E", "num_hits", "track_p", "ecal_dispersion", "hcal_dispersion", "sum_e", "num_tracks"]
    h_names = ["hit_x_avg", "hit_y_avg", "hit_z_avg"]
    h1_names = ["hit_eta_avg", "hit_phi_avg"]
    print("x shape:", r["x"].shape)

    if old_dataset:
        r["y_particles"] = r["y_particles"][:, 6]
        xyz = r["node_features_avg"][:, [0, 1, 2]].cpu()
        eta_phi = torch.stack([calculate_eta(xyz[:, 0], xyz[:, 1], xyz[:, 2]), calculate_phi(xyz[:, 0], xyz[:, 1])],
                              dim=1)
        r["x"] = torch.cat([r["x"], xyz, eta_phi], dim=1)
    if "pid_y" in r:
        r["y_particles"] = r["pid_y"]
    if save_ckpt is not None:
        ds = r["x"], x_names + h_names + h1_names, r["true_e_corr"], r[
        "e_true"], r["e_reco"], r["y_particles"]
        pickle.dump(ds, open(save_ckpt, "wb"))
        print("Dumped dataset to file", save_ckpt)
    return r["x"], x_names + h_names + h1_names, r["true_e_corr"], r[
        "e_true"], r["e_reco"], r["y_particles"]

def get_split(ds, overfit=False):
    from sklearn.model_selection import train_test_split
    x, _, y, etrue, _, pids = ds
    xtrain, xtest, ytrain, ytest, energiestrain, energiestest, pid_train, pid_test = train_test_split(
        x, y, etrue, pids, test_size=0.2, random_state=42
    )
    if overfit:
        return xtrain[:100], xtest[:100], ytrain[:100], ytest[:100], energiestrain[:100], energiestest[:100], pid_train[:100], pid_test[:100]
    return xtrain, xtest, ytrain, ytest, energiestrain, energiestest, pid_train, pid_test


def get_gb():
    # from sklearn.ensemble import GradientBoostingRegressor
    # model = GradientBoostingRegressor(verbose=1, max_depth=7, n_estimators=1000)
    # model = catboost.CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=True, task_type="GPU", devices=DEVICE)
    # xgboost model
    model = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, learning_rate=0.05, verbosity=1,
                                 tree_method="gpu_hist", gpu_id=DEVICE)
    return model


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
    if remove_count > 0:
        # Sort the array
        sorted_arr = np.sort(data)
        # Remove the lowest and highest 1% of the elements
        trimmed_arr = sorted_arr[remove_count:-remove_count]
        # Calculate the mean of the trimmed array
        mean = np.mean(trimmed_arr)
        return mean
    else:
        return data.mean()


def obtain_MPV_and_68_raw(data_for_hist, bins_per_binned_E=np.arange(-1, 5, 0.01), epsilon=0.01):
    hist, bin_edges = np.histogram(data_for_hist, bins=bins_per_binned_E, density=True)
    ind_max_hist = np.argmax(hist)
    # MPV = (bin_edges[ind_max_hist] + bin_edges[ind_max_hist + 1]) / 2
    std68, low, high = get_std68(hist, bin_edges, epsilon=epsilon)
    #MPV = data_for_hist.mean()  # REMOVE THE MEAN WITHOUT OUTLIERS
    MPV = mean_without_outliers(data_for_hist)
    return MPV, std68, low, high

def obtain_MPV_and_68(data_for_hist, *args, **kwargs):
    # trim the data for hist by removing the top and bottom 1%
    #if len(data_for_hist) != 0:
    #    data_for_hist = data_for_hist[
    #        (data_for_hist > np.percentile(data_for_hist, 1)) & (data_for_hist < np.percentile(data_for_hist, 99))]
    # bins_per_binned_E = np.linspace(data_for_hist.min(), data_for_hist.max(), 1000)
    bins_per_binned_E = np.arange(0, 2, 1e-3)
    if len(data_for_hist) == 0:
        return 0, 0, 0, 0
    response, resolution = get_sigma_gaussian(np.nan_to_num(data_for_hist), bins_per_binned_E)
    #return obtain_MPV_and_68_raw(data_for_hist, bins_per_binned_E)
    return response, resolution, 0, 0

# %%
def get_charged_response_resol_plot_for_PID(pid, e_true, e_pred, e_sum_hits, pids, e_track, n_track, neutral=False):
    e_thresholds = [0, 6, 12, 18, 24, 30, 36, 42, 48]  # True E thresholds!
    mpvs_model, s68s_model = [], []
    mpvs_pandora, s68s_pandora = [], []
    mpvs_sum_hits, s68s_sum_hits = [], []
    #e_true = (1 + yt) * split[1][:, 6].numpy()
    #e_pred = yp
    frac_pred = e_pred / e_true
    frac_e_sum = e_sum_hits.clone().detach().cpu().numpy() / e_true
    #e_track = split[1][:, 3].clone().detach().cpu().numpy()
    frac_track = e_track / e_true
    track_filter = ((e_track > 0) & (n_track > 0))
    if neutral:
        track_filter = ~track_filter
        track_filter = track_filter & (n_track == 0)
    track_filter = track_filter & (torch.tensor(pids) == pid).cpu()
    track_filter = track_filter.numpy()
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
    ax[0].set_ylabel("resolution")
    # ax[0].set_ylim([0, 0.4])
    # ax[0].set_ylim([0, 0.4])
    # ax[0].set_ylim([0, 0.4])
    ax[1].plot(bins_x, mpvs_model, ".--", label="GradBoost")
    ax[1].plot(bins_x, mpvs_pandora, ".--", label="track p")
    ax[1].plot(bins_x, mpvs_sum_hits, ".--", label="sum hits")
    #ax[1].set_ylim([0.95, 1.05])
    ax[1].set_ylabel("response")
    ax[0].set_title("PID: " + str(pid))
    upper_plot = {"ML": np.array(s68s_model) / np.array(mpvs_model),
                  "p": np.array(s68s_pandora) / np.array(mpvs_pandora),
                  "sum": np.array(s68s_sum_hits) / np.array(mpvs_sum_hits)}
    lower_plot = {"ML": mpvs_model, "p": mpvs_pandora, "sum": mpvs_sum_hits}
    return fig, upper_plot, lower_plot, bins_x

def calculate_phi(x, y):
    return np.arctan2(y, x)

def calculate_eta(x, y, z):
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    return -np.log(np.tan(theta / 2))

def get_nn(patience, save_to_folder=None, wandb_log_name=None, pid_predict_channels=0):
    # pytorch impl. of a neural network
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    class Net(nn.Module):
        def __init__(self, out_features=1):
            super(Net, self).__init__()
            self.out_features = out_features
            self.model = nn.ModuleList([
                #nn.BatchNorm1d(13),
                nn.Linear(13, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                #nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, out_features)]
            )
            self.wandb_log_name = wandb_log_name
            '''self.model = nn.ModuleList([
                nn.Linear(13, 1, bias=False)
            ])'''

        def forward(self, x):
            for layer in self.model:
                x = layer(x)
            if self.out_features > 1:
                return x[:, 0], x[:, 1:]
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
                pred = self.model(x)
                if isinstance(pred, tuple):
                    return pred[0].cpu().numpy().flatten()
                return self.model(x).cpu().numpy().flatten()

        def fit(self, x, y, pid=None, eval_callback=None):
            # PID: one-hot encoded values of the PID (or some other identification) to additionally use in the loss.
            # It can be real PID or some other identification, i.e. has track / ECAL / ECAL+HCAL etc.
            print("---> Fit - x.shape", x.shape, " y.shape", y.shape)
            if pid is not None:
                print("   --> Also using PID")
            x = torch.tensor(x).to(DEVICE)
            y = torch.tensor(y).to(DEVICE)
            if pid is not None:
                pid = torch.tensor(pid).to(DEVICE)
            total_step = 0
            self.model = Net(out_features = 1+pid_predict_channels)
            self.model.to(DEVICE)
            self.model.train()
            batch_size = args.batch_size
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            '''def criterion(ypred, ytrue, step):
                if step < 5000:
                    return F.l1_loss(ypred, ytrue)
                else:
                    # cut the top 5 % of losses by setting them to 0
                    #losses = F.l1_loss(ypred, ytrue, reduction='none') #+ F.l1_loss(ypred, ytrue, reduction = 'none') / ytrue.abs()
                    losses = F.l1_loss(ypred, ytrue, reduction = 'none') / ytrue.abs()
                    if len(losses.shape) > 0:
                        if int(losses.size(0) * 0.05) > 1:
                            top_percentile = torch.kthvalue(losses, int(losses.size(0) * 0.95)).values
                            mask = (losses > top_percentile)
                            losses[mask] = 0.0
                    return losses.mean()
            def criterion_L1(ypred, ytrue, step):
                return F.l1_loss(ypred, ytrue)'''
            '''if args.loss == "L1":
                print("Using L1 loss")
                criterion = criterion_L1'''

    return NetWrapper()


def main(ds, train_only_on_tracks=False, train_only_on_neutral=False, train_energy_regression=False,
         train_only_on_PIDs=[], remove_sum_e=False, use_model="gradboost", patience=1000, save_to_folder=None, wandb_log_name=None, load_model_ckpt=None):
    split = list(get_split(ds))
    if args.pid_loss:
        pid_channels = len(all_pids)
    else:
        pid_channels = 0
    model = get_nn(patience=patience, save_to_folder=save_to_folder, wandb_log_name=wandb_log_name, pid_predict_channels=pid_channels)
    print("train only on PIDs:", train_only_on_PIDs)
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
    if remove_sum_e:
        split[0][:, 6] = 0.0  # remove the sum of the hits
    if not train_energy_regression:
        print("Fitting")
        if pid_channels > 0:
            pids = split[6].detach().cpu().numpy()  # todo fix this?
            pids_onehot = np.zeros((len(pids), pid_channels))
            for i, pid in enumerate(pids):
                pids_onehot[i, all_pids.index(pid)] = 1.
            result = model.fit(split[0].numpy(), split[2].numpy(), pids_onehot)
        else:
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
        print("LOADING THE MODEL")
        if pid_channels > 0:
            pids = split[6].detach().cpu().tolist()
            pids = [int(x) for x in pids]
            pids_onehot = np.zeros((len(pids), pid_channels))
            for i, pid in enumerate(pids):
                pids_onehot[i, all_pids.index(pid)] = 1.
            result = model.fit(split[0].numpy(), split[4].numpy(), pids_onehot)

            # Try loading the "fine-tuned" model ...
            args.correction = True
            #model1 = GravnetModel.load_from_checkpoint(
            #    "/afs/cern.ch/work/g/gkrzmanc/mlpf_results/FT_EC_perffix_GT_Clusters/_epoch=0_step=9000.ckpt", args=args, dev=0
            #)
            model1 = GravnetModel.load_from_checkpoint(
                "/afs/cern.ch/work/g/gkrzmanc/mlpf_results/FT_EC_perffix/_epoch=0_step=9000.ckpt", args=args, dev=0)
            model.model.model = model1.ec_model_wrapper_neutral.model.model
            model.model.to(DEVICE)
            #print("Set the model to the head inside GATr")
            #model.model.model = pickle.load(open("/eos/user/g/gkrzmanc/2024/models/neutral22000.pkl", "rb"))
        else:
            result = None
            model.model.model = pickle.load(open("/eos/user/g/gkrzmanc/2024/models/neutral22000.pkl", "rb"))
        epred = model.predict(split[1].numpy())
        epred[epred < 0] = 0.
        ytrue = split[3]
        ysum = split[1][:, 6]
        ypred = epred / ysum - 1
        energies = split[5]
        #if save_to_folder is not None:
        #    fig = get_eval_fig(ytrue, ypred, 0, lambda x, y, z: 0, p=split[1][:, 3])
        #    fig.savefig(os.path.join(save_to_folder, "eval.pdf"))
        #    fig.clf()
        e_pred = model.predict(np.nan_to_num(split[1].numpy()))
        pids = split[7].detach().cpu()
        for _pid in all_pids:
            # pid, e_true, e_pred, e_sum_hits, pids, e_track, n_track,
            n_track = split[1][:, 7]
            e_track = split[1][:, 3]
            e_sum_hits = split[1][:, 6]
            e_true = split[5]
            data = get_charged_response_resol_plot_for_PID(_pid, e_true, e_pred, e_sum_hits, pids, e_track, n_track,
                                                           neutral=is_pid_neutral(_pid))
            fig = data[0]
            plot = [data[1], data[2]]
            if save_to_folder is not None:
                try:
                    pickle.dump(plot, open(os.path.join(save_to_folder, f"plots_pid_{_pid}.pkl"), "wb"))
                    fig.savefig(os.path.join(save_to_folder, f"fig_pid_{_pid}.pdf"))
                except:
                    print("Could not save intermediate eval. plots and model")
        return ytrue, epred, energies, split[1], model, split, result
        # log scatterplots of validation results per energy

# %%
ds = get_dataset(save_ckpt="/eos/user/g/gkrzmanc/2024/datasets/dataset_new_model_clusters.pkl")
ds = get_dataset(save_ckpt="/eos/user/g/gkrzmanc/2024/datasets/dataset_new_model_clusters_with_eval_DS.pkl")
print("Loaded dataset")

# print value counts of ds[-1]
import pandas as pd
print("Value counts:", pd.Series(ds[-1]).value_counts( normalize=True))
# %%
# yt, yp, en, _, model, split = main(ds=ds, train_energy_regression=False, train_only_on_PIDs=[211], remove_sum_e=False)

def is_pid_neutral(pid):
    return pid in [22, 130, 2112]


def get_plots(PIDs, energy_regression=False, remove_sum_e=False, use_model="gradboost", patience=1000, save_to_folder=None, wandb_log_name=None):
    if not os.path.exists(save_to_folder):
        os.makedirs(save_to_folder)
    yt, yp, en, _, model, split, lossfn = main(ds=ds, train_energy_regression=energy_regression, train_only_on_PIDs=PIDs,
                                               remove_sum_e=remove_sum_e, use_model=use_model, patience=patience, save_to_folder=save_to_folder,
                                               wandb_log_name = wandb_log_name)
    # import shap
    # import numpy as np
    # te = shap.TreeExplainer(model)
    # shap_vals_r = te.shap_values(np.array(split[1]))
    x_names = ["ecal_E", "hcal_E", "num_hits", "track_p", "ecal_dispersion", "hcal_dispersion", "sum_e", "num_tracks"]
    h_names = ["hit_x_avg", "hit_y_avg", "hit_z_avg", "eta", "phi"]
    # shap.summary_plot(shap_vals_r, split[1], feature_names=x_names + h_names, use_log_scale=True, show=False)
    # plt.show()
    #results = {}
    #for pid in PIDs:
        #fig, upper, lower, x = get_charged_response_resol_plot_for_PID(pid, yt, yp, en, model, split,
        #                                                               neutral=is_pid_neutral(pid))
        #results[pid] = [fig, upper, lower, x, model]
        #if  wandb_log_name is not None:
        #    try:
        #        wandb.log({"fig_" + wandb_log_name: fig})
        ##    except:
        #        print("Could not log fig")
    #return results, lossfn
    return model

#all_pids = [22,130,2112]
#all_pids = [211, -211, 2212, -2212]

model = get_plots(all_pids, energy_regression=True, patience=args.patience,
                                  save_to_folder=os.path.join(prefix, "intermediate_plots"),
                                  wandb_log_name="loss_train_all")
'''fig, ax = plt.subplots()
ax.plot(list(range(len(result_all[1]))), result_all[1])
ax.set_yscale("log")
ax.set_xlabel("Batch")
ax.set_ylabel("Loss")
fig.savefig(prefix + "train_all_loss.pdf")
fig.clf()

for pid in all_pids:
    model = plots_all[pid][4].model.model
    pickle.dump(model.model.model, open(prefix + "NN_model_all_{}.pkl".format(pid), "wb"))
    fig = plots_all[pid][0]
    fig.savefig(prefix + "NN_train_all_{}.pdf".format(pid))
    plots = [plots_all[pid][1], plots_all[pid][2]]
    pickle.dump(plots, open(prefix + "plots_train_all_{}.pkl".format(pid), "wb"))
    wandb.log("final_plot_" + str(pid), fig)

'''