# Compact training script to train a simple NN
import mplhep as hep
import wandb
hep.style.use("CMS")
import matplotlib
matplotlib.rc('font', size=13)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np
import torch
import argparse
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp


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
parser.add_argument("--corrected-energy", action="store_true", default=False) # whether to use the daughters-corr. energy
parser.add_argument("--gnn-features-placeholders", type=int, default=0)
parser.add_argument("--regress-pos", default=False, action="store_true")
parser.add_argument("--load-model-weights", type=str, default=None)
# will add some N(0, 1) iid features to the NN to be used as placeholders for GNN features that will come later
# - first just overfit the simple NN to perform energy regression

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
#wandb.run.log_code(".")

# make dir
os.makedirs(prefix, exist_ok=True)

def get_eval_fig(ytrue, ypred, step, criterion, p=None):
    # calc losses, and plot loss histogram for  energy (ytrue) ranges [0, 6], [6, 12] etc. You need to filter by ytrue!
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].scatter(ytrue, ypred, alpha=0.3)
    if p is not None:
        ax[0].scatter(ytrue, p, color="red", alpha=0.3)
    ax[0].set_ylabel("Predicted energy")
    ax[0].set_xlabel("True energy")
    acceptable_loss = 1e-2
    rages = [[0, 5], [5, 15], [15, 35], [35, 50]]
    for i, r in enumerate(rages):
        mask = (ytrue >= r[0]) & (ytrue < r[1])
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
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)
    if save_ckpt is not None:
        # check if exists
        if os.path.exists(save_ckpt):
            print("Loading dataset from", save_ckpt)
            #r = pickle.ile exiload(open(save_ckpt, "rb"))
            r = CPU_Unpickler(open(save_ckpt, "rb")).load()
            print("len x", len(r["x"]))
        else:
            r = None
    else:
        r = None
    old_dataset = False
    if r is None:
        path = args.dataset_path
        r = {}
        n = 0
        nmax = 1000000
        print("Loading dataset (a bit slow way of doing it, TODO fix)")
        print("Dataset path:", path)
        for file in tqdm.tqdm(os.listdir(path)):
            n += 1
            if n > nmax: #or os.path.isdir(os.path.join(path, file)):
                break
            if os.path.isdir(os.path.join(path, file)):
                continue
            #f = pickle.load(open(os.path.join(path, file), "rb"))
            f = CPU_Unpickler(open(os.path.join(path, file), "rb")).load()
            if (len(file) != len("8510eujir6.pkl")): # in case some temporary files are still stored there
                continue
            #print(f.keys())
            if (f["e_reco"].flatten() == 1.).all():
                continue  # some old files, ignore them for now
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
                        r[key] = torch.concatenate([torch.tensor(r[key]), torch.tensor(f[key])], axis=0)
                else:
                    if "pid_y" not in f.keys():
                        print(key)
                        if key not in r:
                            r[key] = f[key].pid.flatten()
                            r["part_coord"] = f[key].coord
                        else:
                            r[key] = torch.concatenate((r[key], f[key].pid.flatten()), axis=0)
                            r["part_coord"] = torch.concatenate((r["part_coord"], f[key].coord))
                            assert len(r["part_coord"]) == len(r[key])
    x_names = ["ecal_E", "hcal_E", "num_hits", "track_p", "ecal_dispersion", "hcal_dispersion", "sum_e", "num_tracks", "track_p_chis"]
    h_names = ["hit_x_avg", "hit_y_avg", "hit_z_avg"]
    h1_names = ["hit_eta_avg", "hit_phi_avg"]
    print(r.keys())
    print("x shape:", r["x"].shape)
    if old_dataset:
        r["y_particles"] = r["y_particles"][:, 6]
        xyz = r["node_features_avg"][:, [0, 1, 2]].cpu()
        eta_phi = torch.stack([calculate_eta(xyz[:, 0], xyz[:, 1], xyz[:, 2]), calculate_phi(xyz[:, 0], xyz[:, 1])], dim=1)
        r["x"] = torch.cat([r["x"], xyz, eta_phi], dim=1)
    key = "e_true"
    true_e_corr_f = r["true_e_corr"]
    if args.corrected_energy:
        key = "e_true_corrected_daughters"
        true_e_corr_f = r["e_true_corrected_daughters"] / r["e_reco"] - 1
    if "pid_y" in r:
        r["y_particles"] = r["pid_y"]
        #if args.regress_pos:
        #    r["eta"] = calculate_eta(r["coords_y"][:, 0],r["coords_y"][:, 1], r["coords_y"][:, 2])
        #    r["phi"] = calculate_phi(r["coords_y"][:, 0], r["coords_y"][:, 1], r["coords_y"][:, 2])
        #else:
        #    r["eta"] = calculate_eta(r["x"][:, 0], r["x"][:, 1], r["x"][:, 2])
        #    r["phi"] = calculate_phi(r["x"][:, 0], r["x"][:, 1], r["x"][:, 2]) # not regressing positions
    abs_energy_diff = np.abs(r["e_true_corrected_daughters"] - r["e_true"])
    electron_brems_mask = (r["pid_y"] == 11) & (abs_energy_diff > 0)
    if save_ckpt is not None and not os.path.exists(save_ckpt):
        #ds = r["x"], x_names + h_names + h1_names, r["true_e_corr"], r[
        #key], r["e_reco"], r["y_particles"],  torch.concatenate([r["eta"].reshape(1, -1), r["phi"].reshape(1, -1)], axis=0).T
        #pickle.dump(ds, open(save_ckpt, "wb"))
        #print("Dumped dataset to file", save_ckpt)
        pickle.dump(r, open(save_ckpt, "wb"))
    return r["x"], x_names + h_names + h1_names, r["e_true"], r[key], r["e_reco"], r["y_particles"], r["coords_y"] #torch.concatenate([r["eta"].reshape(1, -1), r["phi"].reshape(1, -1)], axis=0).T

def get_split(ds, overfit=False):
    from sklearn.model_selection import train_test_split
    x, _, y, etrue, _, pids, positions = ds
    xtrain, xtest, ytrain, ytest, energiestrain, energiestest, pid_train, pid_test, pos_train, pos_test = train_test_split(
        x, y, etrue, pids, positions, test_size=0.2, random_state=42
    )
    if overfit:
        return xtrain[:100], xtest[:100], ytrain[:100], ytest[:100], energiestrain[:100], energiestest[:100], pid_train[:100], pid_test[:100]
    return xtrain, xtest, ytrain, ytest, energiestrain, energiestest, pid_train, pid_test, pos_train, pos_test # 8,9 are pos train and pos test



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

def obtain_MPV_and_68_raw(data_for_hist, bins_per_binned_E=np.arange(0, 3, 1e-3), epsilon=0.01):
    hist, bin_edges = np.histogram(data_for_hist, bins=bins_per_binned_E, density=True)
    ind_max_hist = np.argmax(hist)
    # MPV = (bin_edges[ind_max_hist] + bin_edges[ind_max_hist + 1]) / 2
    std68, low, high = get_std68(hist, bin_edges, epsilon=epsilon)
    MPV = mean_without_outliers(data_for_hist)
    return MPV, std68, low, high

def get_sigma_gaussian(e_over_reco, bins_per_binned_E):
    hist, bin_edges = np.histogram(e_over_reco, bins=bins_per_binned_E, density=True)
    # Calculating the Gaussian PDF values given Gaussian parameters and random variable X
    def gaus(X, C, X_mean, sigma):
        return C * exp(-((X - X_mean) ** 2) / (2 * sigma**2))

    n = len(hist)
    x_hist = np.zeros((n), dtype=float)
    for ii in range(n):
        x_hist[ii] = (bin_edges[ii + 1] + bin_edges[ii]) / 2

    y_hist = hist
    if (torch.tensor(hist) == 0).all():
        return 0,0
    mean = sum(x_hist * y_hist) / sum(y_hist)
    sigma = sum(y_hist * (x_hist - mean) ** 2) / sum(y_hist)
    try:
        param_optimised, param_covariance_matrix = curve_fit(
            gaus, x_hist, y_hist, p0=[max(y_hist), mean, sigma], maxfev=10000
        )
    except:
        return mean, sigma/mean
    if param_optimised[2] < 0:
        param_optimised[2] = sigma
    if param_optimised[1] < 0:
       param_optimised[1] = mean  # due to some weird fitting errors
    assert param_optimised[1] >= 0
    assert param_optimised[2] >= 0
    return param_optimised[1], param_optimised[2] / param_optimised[1]


def obtain_MPV_and_68(data_for_hist, *args, **kwargs):
    # trim the data for hist by removing the top and bottom 1%
    #if len(data_for_hist) != 0:
    #    data_for_hist = data_for_hist[
    #        (data_for_hist > np.percentile(data_for_hist, 1)) & (data_for_hist < np.percentile(data_for_hist, 99))]
    # bins_per_binned_E = np.linspace(data_for_hist.min(), data_for_hist.max(), 1000)
    bins_per_binned_E = np.arange(0, 2, 1e-3)
    if len(data_for_hist) == 0:
        return 0, 0, 0, 0
    #response, resolution = get_sigma_gaussian(np.nan_to_num(data_for_hist), bins_per_binned_E)
    return obtain_MPV_and_68_raw(data_for_hist, bins_per_binned_E)
    #return response, resolution, 0, 0

# %%
def get_charged_response_resol_plot_for_PID(pid, e_true, e_pred, e_sum_hits, pids, e_track, n_track, neutral=False):
    e_thresholds = [0, 5, 15, 35, 50]  # True E thresholds!
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
        mpv, s68 = get_sigma_gaussian(frac_pred[filt_energy & track_filter],
                                             bins_per_binned_E=np.arange(0, 5, binsize))
        mpvs_model.append(mpv)
        s68s_model.append(s68)
        mpv, s68 = get_sigma_gaussian(frac_track[filt_energy & track_filter].clip(max=5),
                                             bins_per_binned_E=np.arange(0, 5, binsize))
        mpvs_pandora.append(mpv)
        s68s_pandora.append(s68)
        mpv, s68 = get_sigma_gaussian(frac_e_sum[filt_energy & track_filter],
                                           bins_per_binned_E=np.arange(0, 5, binsize))
        mpvs_sum_hits.append(mpv)
        s68s_sum_hits.append(s68)
        print("MPV sum hits", mpvs_sum_hits)
        print("s68 sum hits", s68s_sum_hits)


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
    ax[1].set_ylim([0.95, 1.05])
    ax[1].set_ylabel("response")
    ax[0].set_title("PID: " + str(pid))
    upper_plot = {"ML": np.array(s68s_model) / np.array(mpvs_model),
                  "p": np.array(s68s_pandora) / np.array(mpvs_pandora),
                  "sum": np.array(s68s_sum_hits) / np.array(mpvs_sum_hits)}
    lower_plot = {"ML": mpvs_model, "p": mpvs_pandora, "sum": mpvs_sum_hits}
    return fig, upper_plot, lower_plot, bins_x

def calculate_phi(x, y, z=None):
    return np.arctan2(y, x)

def calculate_eta(x, y, z):
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    return -np.log(np.tan(theta / 2))

def get_nn(patience, save_to_folder=None, wandb_log_name=None, pid_predict_channels=0, muons=0):
    # pytorch impl. of a neural network
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    class Net(nn.Module):
        def __init__(self, out_features=1):
            super(Net, self).__init__()
            self.out_features = out_features
            self.n_gnn_feat = args.gnn_features_placeholders
            self.model = nn.ModuleList([
                #nn.BatchNorm1d(13),
                nn.Linear(14 + args.gnn_features_placeholders + muons, 64),
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
            # pad x with self.n_gnn_feat randomly distributed features
            if self.n_gnn_feat > 0:
                x = torch.cat([x, torch.randn(x.size(0), self.n_gnn_feat).to(DEVICE)], dim=1)
            for layer in self.model:
                x = layer(x)
            #if self.out_features > 1:
            #    return x[:, 0], x[:, 1:]
            return x

        def freeze_batchnorm(self):
            for layer in self.model:
                if isinstance(layer, nn.BatchNorm1d):
                    layer.eval()
                    print("Frozen batchnorm in 1st layer only - ", layer)
                    break

    class NetWrapper():
        def __init__(self):
            self.bce_pid_loss = not args.regress_pos
            pass
        def predict(self, x):
            x = torch.tensor(x).to(DEVICE)
            self.model.eval()
            with torch.no_grad():
                # pred = self.model(x)
                #if isinstance(pred, tuple):
                #    return pred.cpu().numpy()  #, pred[1:].cpu().numpy().flatten()
                return self.model(x).cpu().numpy()

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
            if args.load_model_weights is not None:
                print("Loading model weights from", args.load_model_weights)
                self.model.model = pickle.load(open(args.load_model_weights, "rb"))
            self.model.to(DEVICE)
            self.model.train()
            batch_size = args.batch_size
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            def criterion(ypred, ytrue, step):
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
                return F.l1_loss(ypred, ytrue)
            if args.loss == "L1":
                print("Using L1 loss")
                criterion = criterion_L1
            tolerance = 1e-3
            epochs = 10000
            # calc loss of last 100 batches
            best_loss = 1e9
            patience_counter = 0
            freeze_bn_steps = 1000
            losses_all = []
            epoch_losses = []
            for epoch in range(epochs):
                losses_this_epoch = []
                ps_epoch, ytrue_epoch, ypred_epoch = [], [], []
                num_batches = len(x) // batch_size
                for i in (pbar := tqdm.tqdm(range(0, num_batches))):
                    total_step += 1
                    if i == freeze_bn_steps:
                        self.model.freeze_batchnorm()
                    xbatch = x[i*batch_size:i*batch_size + batch_size].to(DEVICE)
                    ybatch = y[i*batch_size:i*batch_size + batch_size].to(DEVICE)
                    if pid is not None:
                        pidbatch = pid[i*batch_size:i*batch_size + batch_size].to(DEVICE)
                    # if only one sample, skip
                    if xbatch.shape[0] == 1:
                        print("Skipping batch of size 1")
                        continue
                    optimizer.zero_grad()
                    if pid is None:
                        ypred = self.model(xbatch)
                    else:
                        # concat xbatch and pidbatch
                        ypred = self.model(xbatch)
                        if pid is not None:
                            ypred, pidpred = ypred[:, 0], ypred[:, 1:]
                    loss_y = criterion(ypred.flatten(), ybatch.flatten(), total_step)
                    pid_loss = 0.
                    if pid is not None:
                        if self.bce_pid_loss:
                            pid_loss = torch.nn.BCEWithLogitsLoss()(pidpred.float(), pidbatch.float())
                        else:
                            pid_loss = torch.nn.L1Loss()(pidpred.float(), pidbatch.float())
                    loss = loss_y + pid_loss
                    loss.backward()
                    losses_all.append(loss.item())
                    losses_this_epoch.append(loss.item())
                    if i % 100 == 0 and self.model.wandb_log_name is not None:
                        wandb.log({self.model.wandb_log_name: loss.item()})
                        if pid is not None:
                            wandb.log({"pid_loss" + self.model.wandb_log_name: pid_loss.item()})
                            # log also confusion matrix
                            pidpred_idx = pidpred.argmax(dim=1)
                            pidtrue_idx = pidbatch.argmax(dim=1)
                            class_names = all_pids
                            wandb.log({"confusion_matrix" + self.model.wandb_log_name: wandb.sklearn.plot_confusion_matrix(pidtrue_idx.cpu().numpy(), pidpred_idx.cpu().numpy())})
                    if i < 100 and epoch % 30 == 0:
                        ytrue_epoch += ybatch.detach().cpu().numpy().tolist()
                        ypred_epoch += ypred.detach().cpu().numpy().flatten().tolist()
                        ps_epoch += xbatch[:, 3].detach().cpu().numpy().flatten().tolist()
                    #pbar.set_description(
                    #    "Loss: " + str(round(loss.item(), 3)) + " / running loss: " + str(round(loss_running_mean, 3)))
                    optimizer.step()
                    rolling_loss = losses_all[-3000:]
                    if np.mean(rolling_loss) < best_loss - tolerance:
                        best_loss = np.mean(rolling_loss)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter > patience:
                            print("Early stopping at running mean loss:", np.mean(rolling_loss))
                            break
                    if total_step % 1000 == 0:
                        # make eval plots data
                        print("Evaluating!")
                        if eval_callback is not None:
                            eval_callback(self, total_step)
                if patience_counter > patience:
                    break
                epoch_losses.append(np.mean(losses_this_epoch))
                fname = prefix + "losses.pdf"
                fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                ax[0].plot(losses_all)
                ax[0].set_ylabel("Loss")
                ax[0].set_xlabel("Epoch")
                ax[1].plot(epoch_losses)
                ax[0].set_yscale("log")
                ax[1].set_ylabel("Loss")
                ax[1].set_xlabel("Epoch")
                ax[1].set_yscale("log")
                fig.savefig(fname)
                plt.clf()
                if epoch % 30 == 0:
                    print("Epoch", epoch, "loss:", np.mean(losses_this_epoch))
                    if save_to_folder is not None:
                        fig = get_eval_fig(torch.tensor(ytrue_epoch), torch.tensor(ypred_epoch), total_step,
                                           criterion, p=ps_epoch)
                        fig.savefig(os.path.join(save_to_folder, f"epoch_{epoch}.pdf"))
                        fig.clf()
                        # also save the losses
                        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                        ax[0].plot(losses_all)
                        ax[0].set_ylabel("Loss")
                        ax[0].set_xlabel("Epoch")
                        ax[1].plot(epoch_losses)
                        ax[1].set_ylabel("Loss")
                        ax[1].set_xlabel("Epoch")
                        ax[1].set_yscale("log")
                        fig.savefig(os.path.join(save_to_folder, f"losses.pdf"))
                        pickle.dump(losses_all, open(os.path.join(save_to_folder, "losses_all_.pkl"), "wb"))
                        pickle.dump(epoch_losses, open(os.path.join(save_to_folder, "epoch_losses_.pkl"), "wb"))
            return losses_all, epoch_losses
    return NetWrapper()

def plot_deltaR_distr(dict):
    fig, ax = plt.subplots()
    for key in dict:
        ax.hist(dict[key], bins=np.linspace(0, 1, 1000), label=key, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig

def main(ds, train_only_on_tracks=False, train_only_on_neutral=False, train_energy_regression=False,
         train_only_on_PIDs=[], remove_sum_e=False, use_model="gradboost", patience=1000, save_to_folder=None, wandb_log_name=None, load_model_ckpt=None):
    split = list(get_split(ds))
    if args.pid_loss:
        pid_channels = len(all_pids)
    else:
        pid_channels = 0
    if args.regress_pos:
        pid_channels = 3
    muons = 2*int(split[0].shape[1] == 16)
    model = get_nn(patience=patience, save_to_folder=save_to_folder, wandb_log_name=wandb_log_name, pid_predict_channels=pid_channels, muons=muons)
    print("train only on PIDs:", train_only_on_PIDs)
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
        split[8] = split[8][mask]
        split[9] = split[9][masktest]
    if remove_sum_e:
        split[0][:, 6] = 0.0  # Remove the sum of the hits
    if not train_energy_regression:
        print("Fitting")
        if pid_channels > 0:
            if args.regress_pos:
                print("p vectors to regress", split[8][:10])
                print("------------------")
                model.fit(split[0].numpy(), split[2].numpy(), split[8].numpy())
            else:
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
        print("Fitting")
        def eval_callback(self, epoch):
            e_pred = self.predict(split[1].numpy())
            if args.regress_pos:
                e_pred, pos_pred = e_pred[:, 0], e_pred[:, 1:]
                pos_true = split[9]
                print("p_true:", pos_true[:10])
                print("p_pred:", pos_pred[:10])
                #pos_avg_hits = split[1][:, -2:]
                #deltar_avg_hits = torch.sum((pos_avg_hits - pos_true) ** 2, dim=1).sqrt()
                deltar_pred = torch.sum((torch.tensor(pos_pred) - pos_true) ** 2, dim=1).sqrt()
            pids = split[7].detach().cpu()
            print("etrue", split[5].flatten())
            print("e_pred", e_pred.flatten())
            wandb.log({"validation_energy_loss": torch.nn.L1Loss()(split[5], torch.tensor(e_pred).flatten())})

            for _pid in all_pids:
                if args.regress_pos:
                    fig = plot_deltaR_distr({"ML": deltar_pred[pids == _pid]})#, "hit average": deltar_avg_hits[pids == _pid]})
                    fig.suptitle("PID: " + str(_pid) + " / epoch " + str(epoch))
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    wandb.log({"eval_position_regression" + str(_pid): wandb.Image(Image.open(buf))})
                # pid, e_true, e_pred, e_sum_hits, pids, e_track, n_track,
                n_track = split[1][:, 7]
                e_track = split[1][:, 3]
                e_sum_hits = split[1][:, 6]
                e_true = split[5]
                data = get_charged_response_resol_plot_for_PID(_pid, e_true.flatten(), e_pred.flatten(), e_sum_hits, pids, e_track, n_track, neutral=is_pid_neutral(_pid))
                fig = data[0]
                plot = [data[1], data[2]]
                if save_to_folder is not None:
                    try:
                        pickle.dump(plot, open(os.path.join(save_to_folder, f"plots_step_{epoch}_pid_{_pid}.pkl"), "wb"))
                        pickle.dump(self.model.model, open(os.path.join(save_to_folder, f"model_step_{epoch}_pid_{_pid}.pkl"), "wb"))  # save model
                    except:
                        print("Could not save intermediate eval. plots and model")
                fig.suptitle("step" + str(epoch))
                #wandb.log({"eval_fig_eval_data_" + str(pid): fig})
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                wandb.log({"eval_fig_eval_data_" + str(_pid): wandb.Image(Image.open(buf))})
                buf.close()
            ''' # ALSO WITH TRAIN DATA
            e_pred = self.predict(split[0].numpy())
            for pid in all_pids:
                data = get_charged_response_resol_plot_for_PID(pid, e_pred, split[2], split[4], self, split, neutral=False)
                fig = data[0]
                fig.suptitle("PID: " + str(pid) + " / epoch " + str(epoch) + " (Training data!!)")
                wandb.log({"eval_fig_train_data_" + str(pid): fig})'''
        if pid_channels > 0:
            if args.regress_pos:
                print("p vectors to regress", split[8][:10])
                print("------------------")
                model.fit(split[0].numpy(), split[4].numpy(), split[8].numpy(), eval_callback=eval_callback)
            else:
                pids = split[6].detach().cpu().numpy()  # todo fix this?
                pids_onehot = np.zeros((len(pids), pid_channels))
                for i, pid in enumerate(pids):
                    pids_onehot[i, all_pids.index(pid)] = 1.
                result = model.fit(split[0].numpy(), split[4].numpy(), pids_onehot)
        else:
            result = model.fit(split[0].numpy(), split[4].numpy(), eval_callback=eval_callback)
        epred = model.predict(split[1].numpy())
        ytrue = split[3]
        ysum = split[1][:, 6]
        ypred = epred / ysum - 1
        energies = split[5]
        return ytrue, epred, energies, split[1], model, split, result
        # log scatterplots of validation results per energy

# %%
ds = get_dataset(save_ckpt=os.path.join(args.dataset_path + "ds_corr_d.pkl"))
print("Loaded dataset")
# %%
# yt, yp, en, _, model, split = main(ds=ds, train_energy_regression=False, train_only_on_PIDs=[211], remove_sum_e=False)
len(ds)

def is_pid_neutral(pid):
    return pid in [22, 130, 2112]

def get_plots(PIDs, energy_regression=False, remove_sum_e=False, use_model="gradboost", patience=1000, save_to_folder=None, wandb_log_name=None):
    if not os.path.exists(save_to_folder):
        os.makedirs(save_to_folder)
    yt, yp, en, _, model, split, lossfn = main(ds=ds, train_energy_regression=energy_regression, train_only_on_PIDs=PIDs,
                                               remove_sum_e=remove_sum_e, use_model=use_model, patience=patience, save_to_folder=save_to_folder,
                                               wandb_log_name = wandb_log_name)

    x_names = ["ecal_E", "hcal_E", "num_hits", "track_p", "ecal_dispersion", "hcal_dispersion", "sum_e", "num_tracks"]
    h_names = ["hit_x_avg", "hit_y_avg", "hit_z_avg", "eta", "phi"]
    return model


model = get_plots(all_pids, energy_regression=True, patience=args.patience,
                                  save_to_folder=os.path.join(prefix, "training_energy_correction_head"),
                                  wandb_log_name="loss_train_all")
