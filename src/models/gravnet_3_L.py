import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_add
from torch.nn import Parameter
from src.layers.GravNetConv3 import GravNetConv, WeirdBatchNorm
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.layers.object_cond import (
    calc_LV_Lbeta,
    get_clustering,
    calc_LV_Lbeta_inference,
)
from src.layers.obj_cond_inf import calc_energy_loss
from src.layers.mlp_readout_layer import MLPReadout
from src.layers.inference_oc import create_and_store_graph_output
from src.models.gravnet_calibration import (
    object_condensation_loss2,
    obtain_batch_numbers,
)
from src.utils.save_features import save_features
import lightning as L
from src.utils.nn.tools import log_losses_wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.post_clustering_features import get_post_clustering_features
from src.models.GattedGCN_correction import GraphTransformerNet, GCNNet, LinearGNNLayer
from src.layers.inference_oc import hfdb_obtain_labels
import torch_cmspepr
from src.layers.inference_oc import match_showers
from lightning.pytorch.callbacks import BaseFinetuning
import os
import wandb

class GravnetModel(L.LightningModule):
    def __init__(
        self,
        args,
        dev,
        input_dim: int = 8,
        output_dim: int = 4,
        n_postgn_dense_blocks: int = 3,
        n_gravnet_blocks: int = 4,
        clust_space_norm: str = "twonorm",
        k_gravnet: int = 7,
        activation: str = "elu",
        weird_batchnom=False,
        use_correction=False
    ):

        super(GravnetModel, self).__init__()
        self.dev = dev
        self.loss_final = 100
        # self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []
        self.args = args
        self.validation_step_outputs = []
        assert activation in ["relu", "tanh", "sigmoid", "elu"]
        acts = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
        }
        self.act = acts[activation]

        N_NEIGHBOURS = [16, 128, 16, 128]
        TOTAL_ITERATIONS = len(N_NEIGHBOURS)
        self.return_graphs = False
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = TOTAL_ITERATIONS
        self.n_postgn_dense_blocks = n_postgn_dense_blocks
        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_1 = WeirdBatchNorm(self.input_dim)
        else:
            self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim)

        self.Dense_1 = nn.Linear(input_dim, 64, bias=False)
        self.Dense_1.weight.data.copy_(torch.eye(64, input_dim))
        assert clust_space_norm in ["twonorm", "tanh", "none"]
        self.clust_space_norm = clust_space_norm

        self.d_shape = 32
        self.gravnet_blocks = nn.ModuleList(
            [
                GravNetBlock(
                    64 if i == 0 else (self.d_shape * i + 64),
                    k=N_NEIGHBOURS[i],
                    weird_batchnom=weird_batchnom,
                )
                for i in range(self.n_gravnet_blocks)
            ]
        )

        # Post-GravNet dense layers
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend(
                [
                    nn.Linear(4 * self.d_shape + 64 if i == 0 else 64, 64),
                    self.act,  # ,
                ]
            )
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)

        # Output block
        # self.output = nn.Sequential(
        #     nn.Linear(64, 64),
        #     self.act,
        #     nn.Linear(64, 64),
        #     self.act,
        #     nn.Linear(64, 64),
        # )

        # self.post_pid_pool_module = nn.Sequential(  # to project pooled "particle type" embeddings to a common space
        #     nn.Linear(22, 64),
        #     self.act,
        #     nn.Linear(64, 64),
        #     self.act,
        #     nn.Linear(64, 22),
        #     nn.Softmax(dim=-1),
        # )
        self.clustering = nn.Linear(64, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(64, 1)

        # init_weights_ = True
        # if init_weights_:
        #     # init_weights(self.clustering)
        #     init_weights(self.beta)
        #     init_weights(self.postgn_dense)
        #     # init_weights(self.output)

        if weird_batchnom:
            self.ScaledGooeyBatchNorm2_2 = WeirdBatchNorm(64)
        else:
            self.ScaledGooeyBatchNorm2_2 = nn.BatchNorm1d(64)  # , momentum=0.01)
        if self.args.correction:
            self.strict_loading = False
            if not self.args.global_features:
                if self.args.graph_level_features:
                    self.GatedGCNNet = MLPReadout(8, 1, L=3)
                else:
                    self.GatedGCNNet = LinearGNNLayer(self.dev, activation="linear")
            else:
                print("Initializing!")
                self.GatedGCNNet = LinearGNNLayer(self.dev, activation="linear", in_dim_node=18)
                print("Len of NN params", len([param for param in self.GatedGCNNet.parameters() if param.requires_grad]))

    def forward(self, g, y, step_count):
        #print("Num. of trainable params", len([param for param in self.parameters() if param.requires_grad]))
        #for p in self.parameters():
        #    if p.requires_grad:
        #        print("--> param name: ", p.name, " dir: ", dir(p))
        #print("Num of trainable params in the sub NN" , len([param for param in self.GatedGCNNet.parameters() if param.requires_grad]))
        x = g.ndata["h"]
        original_coords = x[:, 0:3]
        g.ndata["original_coords"] = original_coords
        device = x.device
        batch = obtain_batch_numbers(x, g)
        x = self.ScaledGooeyBatchNorm2_1(x)
        x = self.Dense_1(x)
        assert x.device == device

        allfeat = []  # To store intermediate outputs
        allfeat.append(x)
        graphs = []
        loss_regularizing_neig = 0.0
        loss_ll = 0
        if self.trainer.is_global_zero and (step_count % 100 == 0):
            PlotCoordinates(g, path="input_coords", outdir=self.args.model_prefix)
        for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
            #! first time dim x is 64
            #! second time is 64+d
            x, graph, loss_regularizing_neig_block, loss_ll_ = gravnet_block(
                g,
                x,
                batch,
                original_coords,
                step_count,
                self.args.model_prefix,
                num_layer,
            )
            allfeat.append(x)
            graphs.append(graph)
            loss_regularizing_neig = (
                loss_regularizing_neig_block + loss_regularizing_neig
            )
            loss_ll = loss_ll_ + loss_ll
            if len(allfeat) > 1:
                x = torch.concatenate(allfeat, dim=1)

        x = torch.cat(allfeat, dim=-1)
        x = self.postgn_dense(x)
        x = self.ScaledGooeyBatchNorm2_2(x)
        x_cluster_coord = self.clustering(x)
        beta = self.beta(x)
        if self.args.tracks:
            mask = g.ndata["hit_type"] == 1
            beta[mask] = 9
        g.ndata["final_cluster"] = x_cluster_coord
        g.ndata["beta"] = beta.view(-1)
        if self.trainer.is_global_zero and (step_count % 100 == 0):
            PlotCoordinates(
                g,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        if self.args.correction:
            graphs_new, true_new, sum_e = obtain_clustering_for_matched_showers(
                g, x, y, self.trainer.global_rank
            )
            batch_num_nodes = graphs_new.batch_num_nodes()
            batch_idx = []
            for i, n in enumerate(batch_num_nodes):
                batch_idx.extend([i] * n)
            batch_idx = torch.tensor(batch_idx).to(device)
            graphs_new.ndata["h"][:, 0:3] = graphs_new.ndata["h"][:, 0:3] / 3300
            # TODO: add global features to each node here
            print("Add global features?")
            if self.args.global_features:
                print("Using global features of the graphs as well")
                # graphs_num_nodes = graphs_new.batch_num_nodes
                # add num_nodes for each node
                graphs_sum_features = scatter_add(graphs_new.ndata["h"], batch_idx, dim=0)
                # now multiply graphs_sum_features so the shapes match
                graphs_sum_features = graphs_sum_features[batch_idx]
                # append the new features to "h" (graphs_sum_features)
                shape0 = graphs_new.ndata["h"].shape
                graphs_new.ndata["h"] = torch.cat((graphs_new.ndata["h"], graphs_sum_features), dim=1)
                assert shape0[1] * 2 == graphs_new.ndata["h"].shape[1]
            if self.args.graph_level_features:
                #print("Also computing graph-level features")
                graphs_high_level_features = get_post_clustering_features(graphs_new, sum_e)
                #print("Computed graph-level features")
                #print("Shape", graphs_high_level_features)
                pred_energy_corr = self.GatedGCNNet(graphs_high_level_features)
                assert graphs_high_level_features.shape[0] == graphs_new.batch_num_nodes().shape[0]
                return x, pred_energy_corr, true_new, sum_e, graphs_new, batch_idx, graphs_high_level_features
            else:
                pred_energy_corr = self.GatedGCNNet(graphs_new)
                return x, pred_energy_corr, true_new, sum_e, graphs_new, batch_idx, None
        else:
            pred_energy_corr = torch.ones_like(beta.view(-1, 1))
            return x, pred_energy_corr, 0

    # def on_after_backward(self):
    #     for name, p in self.named_parameters():
    #         if p.grad is None:
    #             print(name)

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]
        if self.trainer.is_global_zero:
            if self.args.correction:
                model_output, e_cor1, true_e, sum_e, new_graphs, batch_idx, graph_level_features = self(batch_g, y, batch_idx)
                e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
                loss_ll = 0
            else:
                model_output, e_cor, loss_ll = self(batch_g, y, batch_idx)
        else:
            if self.args.correction:
                model_output, e_cor1, true_e, sum_e, new_graphs, batch_idx, graph_level_features = self(batch_g, y, 1)
                e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
                loss_ll = 0
            else:
                model_output, e_cor, loss_ll = self(batch_g, y, 1)
                e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        '''energies_sums_features = new_graphs.ndata["h"][:, 15]
        energies_sums = [sum_e[i] for i in batch_idx]
        energies_sums = torch.tensor(energies_sums).to(energies_sums_features.device).flatten()
        print(energies_sums[energies_sums != energies_sums_features])
        print(energies_sums_features[energies_sums != energies_sums_features])
        assert (torch.abs(energies_sums - energies_sums_features) < 0.001).all()'''
        # print(model_output.shape, e_cor.shape, e_cor1.shape)
        # e_cor1 += 1.  # We regress the number around zero!!! # TODO: uncomment this if needed
        (loss, losses, loss_E, loss_E_frac_true,) = object_condensation_loss2(
            batch_g,
            model_output,
            e_cor,
            y,
            clust_loss_only=True,
            add_energy_loss=False,
            calc_e_frac_loss=False,
            q_min=self.args.qmin,
            frac_clustering_loss=self.args.frac_cluster_loss,
            attr_weight=self.args.L_attractive_weight,
            repul_weight=self.args.L_repulsive_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            hgcalloss=self.args.hgcalloss,
        )
        if self.args.correction:
            debug_regress_e_sum = False
            def corr(array):
                return torch.clamp(array, min=0.000001)
            if debug_regress_e_sum:
                # This section is just for debugging!
                loss, loss_abs, loss_abs_nocali = loss_reco_sum_absolute(e_cor1, torch.log10(true_e), torch.log10(sum_e))
                data = [[x, y] for (x, y) in zip(torch.log10(sum_e), e_cor1)]
                table = wandb.Table(data=data, columns=["sum E (GT)", "sum E (regressed)"])
                wandb.log({"energy_corr": wandb.plot.scatter(table, "sum E (GT)", "sum E (regressed)",
                                                             title="energy correction correlation")})
                print("Logged!")
            else:
                loss, loss_abs, loss_abs_nocali = loss_reco_true(e_cor1, true_e, sum_e)
                true_e_corr = (true_e / sum_e) - 1
                model_e_corr = e_cor1
                data = [[x, y] for (x, y) in zip(true_e_corr, model_e_corr)]
                table = wandb.Table(data=data, columns=["true E corr factor", "regressed E corr factor"])
                wandb.log({"energy_corr": wandb.plot.scatter(table, "true E corr factor", "regressed E corr factor",
                                                                   title="energy correction correlation")})
                if self.args.graph_level_features:
                    # also plot graph level features correlations
                    data = [[x, y] for (x, y) in zip(true_e_corr, sum_e)]
                    table = wandb.Table(data=data, columns=["true E corr factor", "sum of E of hits"])
                    #wandb.log({"energy_corr_vs_E_hits": wandb.plot.scatter(table, "true E corr factor", "sum of E of hits",
                    #                                                       title="energy correction vs. sum of E hits")})
                    #  save graph-level features temporarily, to view later...
                    cluster_features_path = os.path.join(self.args.model_prefix, "cluster_features")
                    if not os.path.exists(cluster_features_path):
                        os.makedirs(cluster_features_path)
                    save_features(cluster_features_path, {"x": graph_level_features.detach().cpu(), "e_true": true_e.detach().cpu(),
                                                  "e_reco": model_e_corr.detach().cpu(), "true_e_corr": true_e_corr.detach().cpu()})
                    print("!!!temporarily saving features in an external file!!!!")
                print("Logged!")
        else:
            loss = loss  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
        if self.trainer.is_global_zero:
            log_losses_wandb(True, len(batch_idx), 0, losses, loss, loss_ll)

        self.loss_final = loss
        return loss

    def validation_step(self, batch, batch_idx):
        show_df_eval_path = os.path.join(self.args.model_prefix, "showers_df_evaluation")
        cluster_features_path = os.path.join(self.args.model_prefix, "cluster_features")
        if not os.path.exists(show_df_eval_path):
            os.makedirs(show_df_eval_path)
        if not os.path.exists(cluster_features_path):
            os.makedirs(cluster_features_path)
        self.validation_step_outputs = []
        y = batch[1]
        batch_g = batch[0]
        if self.args.correction:
            model_output, e_cor1, true_e, sum_e, new_graphs, batch_id, graph_level_features = self(batch_g, y, 1)
            loss_ll = 0
            e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        else:
            model_output, e_cor1, loss_ll = self(batch_g, y, 1)
            loss_ll = 0
            e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        preds = model_output.squeeze()
        (loss, losses, loss_E, loss_E_frac_true,) = object_condensation_loss2(
            batch_g,
            model_output,
            e_cor,
            y,
            clust_loss_only=True,
            add_energy_loss=False,
            calc_e_frac_loss=False,
            q_min=self.args.qmin,
            frac_clustering_loss=self.args.frac_cluster_loss,
            attr_weight=self.args.L_attractive_weight,
            repul_weight=self.args.L_repulsive_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            hgcalloss=self.args.hgcalloss,
        )
        if self.args.correction:
            loss, loss_abs, loss_abs_nocali = loss_reco_true(e_cor1, true_e, sum_e)
            loss_ec = 0
        else:
            print("Doing both correction and OC loss!!")
            loss_ec, loss_abs, loss_abs_nocali = loss_reco_true(e_cor1, true_e, sum_e)
            loss = loss + 0.05 * loss_ec
        print("starting validation step", batch_idx, loss)
        if self.trainer.is_global_zero:
            log_losses_wandb(True, batch_idx, 0, losses, loss, loss_ll, loss_ec, val=True)
        self.validation_step_outputs.append([model_output, e_cor, batch_g, y])
        if self.args.predict:
            model_output1 = torch.cat((model_output, e_cor.view(-1, 1)), dim=1)
            if self.args.correction:
                e_corr = e_cor1
            else:
                e_corr = None
            (df_batch_pandora, df_batch1,) = create_and_store_graph_output(
                batch_g,
                model_output1,
                y,
                0,
                batch_idx,
                0,
                path_save=show_df_eval_path,
                store=True,
                predict=True,
                e_corr=e_corr,
                tracks=self.args.tracks,
            )
            # self.df_showers.append(df_batch)
            self.df_showers_pandora.append(df_batch_pandora)
            self.df_showes_db.append(df_batch1)

    def on_train_epoch_end(self):
        # log epoch metric
        self.log("train_loss_epoch", self.loss_final)

    def on_train_epoch_start(self):
        # if self.args.correction:
        #     self.turn_grads_off()
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        # self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            self.ScaledGooeyBatchNorm2_1.momentum = 0
            # self.ScaledGooeyBatchNorm2_2.momentum = 0
            # for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
            #     gravnet_block.batchnorm_gravnet1.momentum = 0
            #     gravnet_block.batchnorm_gravnet2.momentum = 0

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            if self.args.predict:
                from src.layers.inference_oc import store_at_batch_end
                import pandas as pd

                # self.df_showers = pd.concat(self.df_showers)
                self.df_showers_pandora = pd.concat(self.df_showers_pandora)
                self.df_showes_db = pd.concat(self.df_showes_db)
                store_at_batch_end(
                    path_save=os.path.join(self.args.model_prefix, "showers_df_evaluation"),
                    # df_batch=self.df_showers,
                    df_batch_pandora=self.df_showers_pandora,
                    df_batch1=self.df_showes_db,
                    step=0,
                    predict=True,
                )
            else:
                model_output = self.validation_step_outputs[0][0]
                e_corr = self.validation_step_outputs[0][1]
                batch_g = self.validation_step_outputs[0][2]
                y = self.validation_step_outputs[0][3]
                model_output1 = torch.cat((model_output, e_corr.view(-1, 1)), dim=1)
                create_and_store_graph_output(
                    batch_g,
                    model_output1,
                    y,
                    0,
                    0,
                    0,
                    path_save=os.path.join(self.args.model_prefix, "showers_df_evaluation"),
                    store=True,
                    predict=False,
                    tracks=self.args.tracks,
                )
        self.validation_step_outputs = []

    def configure_optimizers(self):
        print("Configuring optimizer!")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3
        )
        print("Optimizer params:", filter(lambda p: p.requires_grad, self.parameters()))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=3),
                "interval": "epoch",
                "monitor": "train_loss_epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    # def turn_grads_off(self):
    #     for name, param in self.named_parameters():
    #         print("name", name)
    #         if name.split(".")[2] == "GatedGCNNet":
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False


class FreezeClustering(BaseFinetuning):
    def __init__(
        self,
    ):
        super().__init__()
        # self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        print("freezing the following module:", pl_module)
        # freeze any module you want
        # Here, we are freezing `feature_extractor`

        self.freeze(pl_module.ScaledGooeyBatchNorm2_1)
        self.freeze(pl_module.Dense_1)
        self.freeze(pl_module.gravnet_blocks)
        self.freeze(pl_module.postgn_dense)
        self.freeze(pl_module.ScaledGooeyBatchNorm2_2)
        self.freeze(pl_module.clustering)
        self.freeze(pl_module.beta)

        print("CLUSTERING HAS BEEN FROOOZEN")

    def finetune_function(self, pl_module, current_epoch, optimizer):
        print("Not finetunning")
        # # When `current_epoch` is 10, feature_extractor will start training.
        # if current_epoch == self._unfreeze_at_epoch:
        #     self.unfreeze_and_add_param_group(
        #         modules=pl_module.feature_extractor,
        #         optimizer=optimizer,
        #         train_bn=True,
        #     )


class GravNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 96,
        space_dimensions: int = 3,
        propagate_dimensions: int = 22,
        k: int = 40,
        # batchnorm: bool = True
        weird_batchnom=False,
    ):
        super(GravNetBlock, self).__init__()
        self.d_shape = 32
        out_channels = self.d_shape
        if weird_batchnom:
            self.batchnorm_gravnet1 = WeirdBatchNorm(self.d_shape)
        else:
            self.batchnorm_gravnet1 = nn.BatchNorm1d(self.d_shape, momentum=0.01)
        propagate_dimensions = self.d_shape
        self.gravnet_layer = GravNetConv(
            self.d_shape,
            out_channels,
            space_dimensions,
            propagate_dimensions,
            k,
            weird_batchnom,
        ).jittable()

        self.post_gravnet = nn.Sequential(
            nn.Linear(
                out_channels + space_dimensions + self.d_shape, self.d_shape
            ),  #! Dense 3
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 4
            nn.ELU(),
        )
        self.pre_gravnet = nn.Sequential(
            nn.Linear(in_channels, self.d_shape),  #! Dense 1
            nn.ELU(),
            nn.Linear(self.d_shape, self.d_shape),  #! Dense 2
            nn.ELU(),
        )
        # self.output = nn.Sequential(nn.Linear(self.d_shape, self.d_shape), nn.ELU())

        # init_weights(self.output)
        init_weights(self.post_gravnet)
        init_weights(self.pre_gravnet)

        if weird_batchnom:
            self.batchnorm_gravnet2 = WeirdBatchNorm(self.d_shape)
        else:
            self.batchnorm_gravnet2 = nn.BatchNorm1d(self.d_shape, momentum=0.01)

    def forward(
        self,
        g,
        x: Tensor,
        batch: Tensor,
        original_coords: Tensor,
        step_count,
        outdir,
        num_layer,
    ) -> Tensor:
        x = self.pre_gravnet(x)
        x = self.batchnorm_gravnet1(x)
        x_input = x
        xgn, graph, gncoords, loss_regularizing_neig, ll_r = self.gravnet_layer(
            g, x, original_coords, batch
        )
        g.ndata["gncoords"] = gncoords
        # if step_count % 50:
        #     PlotCoordinates(
        #         g, path="gravnet_coord", outdir=outdir, num_layer=str(num_layer)
        #     )
        # gncoords = gncoords.detach()
        x = torch.cat((xgn, gncoords, x_input), dim=1)
        x = self.post_gravnet(x)
        x = self.batchnorm_gravnet2(x)  #! batchnorm 2
        # x = global_exchange(x, batch)
        # x = self.output(x)
        return x, graph, loss_regularizing_neig, ll_r


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)


def obtain_clustering_for_matched_showers(batch_g, model_output, y, local_rank):
    graphs_showers_matched = []
    true_energy_showers = []
    reco_energy_showers = []
    batch_g.ndata["coords"] = model_output[:, 0:3]
    batch_g.ndata["beta"] = model_output[:, 3]
    graphs = dgl.unbatch(batch_g)
    batch_id = y[:, -1].view(-1)
    for i in range(0, len(graphs)):
        mask = batch_id == i
        dic = {}
        dic["graph"] = graphs[i]
        dic["part_true"] = y[mask]
        betas = torch.sigmoid(dic["graph"].ndata["beta"])
        X = dic["graph"].ndata["coords"]
        clustering_mode = "dbscan"
        if clustering_mode == "clustering_normal":
            clustering = get_clustering(betas, X)
        elif clustering_mode == "dbscan":
            labels = hfdb_obtain_labels(X, model_output.device)

            particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
            shower_p_unique = torch.unique(labels)
            shower_p_unique, row_ind, col_ind, i_m_w = match_showers(
                labels, dic, particle_ids, model_output, local_rank, i, None
            )
            row_ind = torch.Tensor(row_ind).to(model_output.device).long()
            col_ind = torch.Tensor(col_ind).to(model_output.device).long()
            index_matches = col_ind + 1
            index_matches = index_matches.to(model_output.device).long()
            for unique_showers_label in shower_p_unique:
                if torch.sum(unique_showers_label == index_matches) == 1:
                    index_in_matched = torch.argmax(
                        (unique_showers_label == index_matches) * 1
                    )
                    mask = labels == unique_showers_label
                    # non_graph = torch.sum(mask)
                    sls_graph = graphs[i].ndata["pos_hits_xyz"][mask][:, 0:3]
                    k = 7
                    edge_index = torch_cmspepr.knn_graph(sls_graph, k=k)
                    g = dgl.graph(
                        (edge_index[0], edge_index[1]), num_nodes=sls_graph.shape[0]
                    )
                    g = dgl.remove_self_loop(g)
                    # g = dgl.DGLGraph().to(graphs[i].device)
                    # g.add_nodes(non_graph.detach().cpu())
                    g.ndata["h"] = torch.cat(
                        (
                            graphs[i].ndata["h"][mask],
                            graphs[i].ndata["beta"][mask].view(-1, 1),
                        ),
                        dim=1,
                    )
                    energy_t = dic["part_true"][:, 3].to(model_output.device)
                    true_energy_shower = energy_t[row_ind[index_in_matched]]
                    reco_energy_shower = torch.sum(graphs[i].ndata["e_hits"][mask])
                    graphs_showers_matched.append(g)
                    true_energy_showers.append(true_energy_shower.view(-1))
                    reco_energy_showers.append(reco_energy_shower.view(-1))
    graphs_showers_matched = dgl.batch(graphs_showers_matched)
    true_energy_showers = torch.cat(true_energy_showers, dim=0)
    reco_energy_showers = torch.cat(reco_energy_showers, dim=0)
    return graphs_showers_matched, true_energy_showers, reco_energy_showers


def loss_reco_true(e_cor, true_e, sum_e):
    # m = nn.ELU()
    # e_cor = m(e_cor)
    print("corection", e_cor[0:5])
    print("sum_e", sum_e[0:5])
    print("true_e", true_e[0:5])
    # true_e = -1 * sum_e  # Temporarily, to debug - so the model would have to learn corr. factor of -1 for each particle...
    loss = torch.square(((e_cor ) * sum_e - true_e) / true_e)
    loss_abs = torch.mean(torch.abs(e_cor * sum_e - true_e) / true_e)
    loss_abs_nocali = torch.mean(torch.abs(sum_e - true_e) / true_e)
    loss = torch.mean(loss)
    return loss, loss_abs, loss_abs_nocali

def loss_reco_sum_absolute(e_cor, true_e, sum_e):
    # implementation of a loss that regresses the sum of the hits instead of the corr. factor ( just for debugging )
    # m = nn.ELU()
    # e_cor = m(e_cor)
    print("corection", e_cor[0:5])
    print("sum_e", sum_e[0:5])
    print("true_e", true_e[0:5])
    # true_e = -1 * sum_e  # Temporarily, to debug - so the model would have to learn corr. factor of -1 for each particle...
    loss = torch.square(e_cor - sum_e)
    loss_abs = torch.mean(torch.abs(e_cor * sum_e - true_e) / true_e)
    loss_abs_nocali = torch.mean(torch.abs(sum_e - true_e) / true_e)
    loss = torch.mean(loss)
    return loss, loss_abs, loss_abs_nocali


