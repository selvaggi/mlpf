
from lightning.pytorch.callbacks import BaseFinetuning
import torch
import torch.nn as nn


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
        # self.freeze(pl_module.Dense_1)
        self.freeze(pl_module.gatr)
        # self.freeze(pl_module.postgn_dense)
        # self.freeze(pl_module.ScaledGooeyBatchNorm2_2)
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
