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

def gen_synthetic_dataset(x):
    # generate synthetic dataset with random features distribution
    print("Generate synthetic dataset")
    #x_syn = torch.randn_like(x)
    # instead of random normal, generate a uniform distr between 0 and 50
    # x_syn = torch.rand_like(x) * 50
    # log_uniform x_syn
    #x_syn = torch.exp(torch.rand_like(x) * np.log(50))
    #y_syn = x_syn[:, 5]  # pick some random feature to use as target
    #return x_syn, y_syn
    return x, x[:, 3]

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

DEVICE = torch.device("cuda:1")
SYNTHETIC_DATASET = False

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
    xtrue = torch.concatenate([r["x"], xyz, eta_phi], dim=1)
    if SYNTHETIC_DATASET:
        xtrue, y = gen_synthetic_dataset(xtrue)
    else:
        y = r["e_true"]
    return xtrue, x_names + h_names + h1_names, r["true_e_corr"], y, r["e_reco"], r["y_particles"][:, 6]
    #  ds == x_ture, names, e_corr_factor, true_e, true_e_reco, pids


def get_split(ds, overfit=True, same_train_test=False):
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


def get_eval_fig(ytrue, ypred, step, criterion, p=None):
    # calc losses, and plot loss histogram for  energy (ytrue) ranges [0, 6], [6, 12] etc. You need to filter by ytrue!
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].scatter(ytrue, ypred, alpha=0.2)
    if p is not None:
        ax[0].scatter(ytrue, p, color="red")
    ax[0].set_ylabel("Predicted energy")
    ax[0].set_xlabel("True energy")
    acceptable_loss = 1e-2
    rages = [[0, 6], [6, 12], [12, 18], [18, 24], [24, 30], [30, 36], [36, 42], [42, 48], [48, 54], [54, 60]]
    for i, r in enumerate(rages):
        mask = (ytrue >= r[0]) & (ytrue < r[1])
        # % BELOW ACCEPTABLE LOSS
        #frac = torch.mean((((ypred[mask] - ytrue[mask]).abs() < acceptable_loss)))*100
        #print(frac)
        #print(torch.mean(((ypred[mask] - ytrue[mask]).abs() < acceptable_loss).float()).item())
        frac = torch.mean(((ypred[mask] - ytrue[mask]).abs() < acceptable_loss).float())
        # if is nan, change to 0
        if torch.isnan(frac):
            frac = 0
        if mask.sum() > 0:
            losses = [criterion(ypred[mask][i], ytrue[mask][i], step).detach().cpu() for i in range(mask.sum())]
            losses = torch.tensor(losses)
            ax[1].hist(torch.clamp(torch.log10(losses), -5, 5), bins=100, alpha=0.5, label=f"{r[0]}-{r[1]} ({str(int(frac*100))}%)")
    ax[0].plot([0, max(ytrue)], [0, max(ytrue)], "--", color="gray")
    ax[1].set_xlim([-5, 5])
    ax[1].set_xlabel("log10(loss)")
    # log scale
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Count")
    ax[1].legend()
    return fig

prefix = "/eos/home-g/gkrzmanc/2024/1_4_/piplus_bs512_demo_patience_50/"
os.makedirs(prefix, exist_ok=True)

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
                #nn.BatchNorm1d(13),
                nn.Linear(13, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                #nn.BatchNorm1d(64),
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
            total_step = 0
            self.model = Net()
            self.model.to(DEVICE)
            self.model.train()
            batch_size = 512
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            def criterion(ypred, ytrue, step):
                #if step < 12000:
                #    return F.mse_loss(ypred, ytrue)
                #elif step < 20000:
                #    return F.l1_loss(ypred, ytrue)
                #else:
                #    # relative loss
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
                    # if only one sample, skip
                    if xbatch.shape[0] == 1:
                        print("Skipping batch of size 1")
                        continue
                    optimizer.zero_grad()
                    ypred = self.model(xbatch)
                    loss = criterion(ypred.flatten(), ybatch, total_step)
                    loss.backward()
                    losses_all.append(loss.item())
                    losses_this_epoch.append(loss.item())
                    if i < 5000:
                        ytrue_epoch += ybatch.detach().cpu().numpy().tolist()
                        ypred_epoch += ypred.detach().cpu().numpy().flatten().tolist()
                        ps_epoch += xbatch[:, 3].detach().cpu().numpy().flatten().tolist()
                    #pbar.set_description(
                    #    "Loss: " + str(round(loss.item(), 3)) + " / running loss: " + str(round(loss_running_mean, 3)))
                    optimizer.step()
                if patience_counter > patience:
                    break
                epoch_losses.append(np.mean(losses_this_epoch))
                fname = prefix + "losses.pdf"
                fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                ax[0].plot(losses_all)
                ax[0].set_ylabel("Loss")
                ax[0].set_xlabel("Epoch")
                ax[1].plot(epoch_losses)
                ax[1].set_ylabel("Loss")
                ax[1].set_xlabel("Epoch")
                ax[1].set_yscale("log")
                fig.savefig(fname)
                plt.clf()
                if epoch % 10 == 0:
                    fig = get_eval_fig(torch.tensor(ytrue_epoch), torch.tensor(ypred_epoch), total_step, criterion, p=ps_epoch)
                    fig.savefig(prefix + f"epoch_{epoch}.pdf")
                if np.mean(losses_this_epoch) < best_loss - tolerance:
                    best_loss = np.mean(losses_this_epoch)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        print("Early stopping at running mean loss:", np.mean(losses_this_epoch))
                        break
                if epoch % 10 == 0:
                    print("Epoch", epoch, "loss:", np.mean(losses_this_epoch))
            return losses_all, epoch_losses
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
        CROP_DS = False
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
        fig, ax = plt.subplots()
        #(split[0][:, 3] / split[4])
        ax.hist(split[0][:, 3] / split[4], bins=100)
        ax.set_yscale("log")
        fig.savefig(prefix + "p_over_Etrue.pdf")
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
ds = list(get_dataset())


#ds[0], ds[2] = gen_synthetic_dataset(ds[0])
print("Loaded dataset")

# %%
yt, yp, en, _, model, split, result = main(ds=ds,
                                           train_energy_regression=True,
                                           train_only_on_PIDs=[211],
                                           remove_sum_e=False,
                                           patience=50)
#len(ds)

# PICKLE THE RESULT
#import pickle
#with open(prefix + "result_loss.pkl", "wb") as f:
#    pickle.dump((yt, yp, en, model, split, result), f)



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
plt.savefig(prefix + "final_corr_test.pdf")

#%%
epochs_loss = result[1]
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(epochs_loss)
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
# same but in log scale
ax[1].plot(epochs_loss)
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Epoch")
ax[1].set_yscale("log")
plt.savefig(prefix + "final_losses.pdf")

#%%






