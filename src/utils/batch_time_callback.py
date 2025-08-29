# src/utils/batch_time_callback.py
import time
import torch
import lightning as L
import numpy as np

class BatchTimeCallback(L.Callback):
    def __init__(self, sync_cuda: bool = True, log_histogram: bool = False):
        self.sync_cuda = sync_cuda
        self.log_histogram = log_histogram
        self._t0 = None
        self._train_times = []
        self._val_times = []

    # ---- TRAIN ----
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(device=pl_module.device)
        dt = time.perf_counter() - self._t0
        self._train_times.append(dt)
        pl_module.log("train/batch_time_ms", dt * 1000.0, on_step=True, prog_bar=True, logger=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._train_times:
            arr = np.array(self._train_times)
            metrics = {
                "train/batch_time_ms_avg": arr.mean() * 1000.0,
                "train/batch_time_ms_p50": np.percentile(arr, 50) * 1000.0,
                "train/batch_time_ms_p95": np.percentile(arr, 95) * 1000.0,
            }
            pl_module.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True, batch_size=10)
            if self.log_histogram and trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                # W&B supports histograms
                trainer.logger.experiment.log({"train/batch_time_ms_hist": arr * 1000.0})
        self._train_times = []

    # ---- VALIDATION (optional) ----
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._t0 = time.perf_counter()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize(device=pl_module.device)
        dt = time.perf_counter() - self._t0
        self._val_times.append(dt)
        pl_module.log("val/batch_time_ms", dt * 1000.0, on_step=True, prog_bar=False, logger=True, batch_size=10)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._val_times:
            arr = np.array(self._val_times)
            metrics = {
                "val/batch_time_ms_avg": arr.mean() * 1000.0,
                "val/batch_time_ms_p50": np.percentile(arr, 50) * 1000.0,
                "val/batch_time_ms_p95": np.percentile(arr, 95) * 1000.0,
            }
            pl_module.log_dict(metrics, prog_bar=False, logger=True, on_epoch=True)
        self._val_times = []
