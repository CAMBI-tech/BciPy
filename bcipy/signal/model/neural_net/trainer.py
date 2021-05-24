import csv
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
from dotted_dict import DottedDict
from loguru import logger
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from bcipy.signal.model import SignalModel
from bcipy.signal.model.neural_net.utils import get_decay_rate


class Trainer:
    def __init__(self, cfg: DottedDict, model: SignalModel):
        self.cfg = cfg
        self.model = model
        self.optim: Union[Adam, AdamW]

        logger.debug("Setup optim...")
        if self.cfg.optimizer == "Adam":
            self.optim = Adam(
                self.model.parameters(), lr=self.cfg.initial_lr, betas=(self.cfg.adam_beta1, self.cfg.adam_beta2)
            )
        elif self.cfg.optimizer == "AdamW":
            self.optim = AdamW(
                self.model.parameters(), lr=self.cfg.initial_lr, betas=(self.cfg.adam_beta1, self.cfg.adam_beta2)
            )
        else:
            raise ValueError(self.cfg.optimizer)

        logger.debug("Setup sched...")
        self.sched: Union[CosineAnnealingLR, ExponentialLR]
        if self.cfg.scheduler == "CosineAnnealingLR":
            self.sched = CosineAnnealingLR(self.optim, T_max=cfg.epochs, eta_min=cfg.final_lr)
        elif self.cfg.scheduler == "ExponentialLR":
            self.sched = ExponentialLR(self.optim, gamma=get_decay_rate(cfg.initial_lr, cfg.final_lr, cfg.epochs))
        else:
            raise ValueError(self.cfg.scheduler)

        logger.debug("Setup counters and paths...")
        self.epoch = 0
        self.global_step = 0  # number of training batches seen
        self.normal_checkpoint = self.cfg.output_dir / "normal_checkpoint.pt"  # checkpoint after every train epoch
        self.best_checkpoint = self.cfg.output_dir / "best_checkpoint.pt"  # best_val_loss checkpoint
        self.final_checkpoint = self.cfg.output_dir / "model_weights.pt"  # the final checkpoint that should be loaded
        if self.cfg.use_early_stop:
            self.best_val_loss = np.Inf
            self.early_stop_now = False

        logger.debug("Setup tensorboard...")
        self.writer = SummaryWriter(self.cfg.output_dir)

    def fit(self, train_set: Dataset, val_set: Dataset):
        example_input = next(iter(train_set))[0]  # type: ignore
        self.model._trace(example_input)  # TODO - where does tracing belong?
        self.train_loader = self._get_loader(train_set)
        self.val_loader = self._get_loader(val_set)

        logger.debug("Begin fit")
        for _ in trange(self.cfg.epochs, desc="Epochs", leave=True):  # TODO - start from 0 or from resumed epoch
            self.train_epoch()
            self.val_epoch()
            if self.cfg.use_early_stop and self.early_stop_now:
                break
        success_flag = self.cfg.output_dir / self.cfg.success_flag
        logger.debug(f"Fitting complete - touch success flag at {success_flag}...")
        success_flag.touch()

        if self.best_checkpoint.exists():
            target = self.best_checkpoint
        else:
            target = self.normal_checkpoint
        logger.debug(f"Symlink {self.final_checkpoint} to {target}")
        self.final_checkpoint.symlink_to(target)

    def _get_loader(self, dataset, shuffle=True):
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % (2 ** 32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.cfg.loader_num_workers,
            worker_init_fn=worker_init_fn,
        )

    def train_epoch(self):
        self.model.train()
        for data, labels in tqdm(self.train_loader, leave=False, desc="Train"):
            self.optim.zero_grad()
            outputs = self.model.get_outputs(data, labels)
            outputs["loss"].backward()
            self.optim.step()
            self.writer.add_scalar("Train/Loss", outputs["loss"], self.global_step)
            self.writer.add_scalar("Train/Acc", outputs["acc"], self.global_step)
            self.global_step += 1

        self.save(self.normal_checkpoint)
        self.sched.step()
        self.epoch += 1

    def val_epoch(self):
        outputs, _, _ = self._test(self.val_loader, "Val")

        if self.cfg.use_early_stop:
            avg_val_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
            if avg_val_loss < self.best_val_loss:  # If we improve, then save & record value & reset counter
                self.best_val_loss = avg_val_loss
                self.save(self.best_checkpoint)
                self.early_stop_counter = 0
            else:  # After `early_stop_patience` epochs without improvement, stop
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.cfg.early_stop_patience:
                    self.early_stop_now = True

        return outputs

    def test(self, test_set):
        # TODO - if acc calculation is unused, remove it
        logger.debug("Begin test")
        self.test_loader = self._get_loader(test_set, shuffle=False)
        outputs, avg_loss, avg_acc = self._test(self.test_loader, "Test", leave_tqdm=True)

        self._save_cfg_and_metrics(avg_loss, avg_acc)
        return outputs

    @torch.no_grad()
    def _test(self, loader, desc: str, leave_tqdm=False):
        """Shared code for val and test"""
        self.model.eval()

        outputs = []
        for data, labels in tqdm(loader, leave=leave_tqdm, desc=desc):
            outputs.append(self.model.get_outputs(data, labels))

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        avg_acc = torch.stack([x["acc"] for x in outputs]).mean().item()
        self.writer.add_scalar(f"{desc}/Avg_Loss", avg_loss, self.global_step)
        self.writer.add_scalar(f"{desc}/Avg_Acc", avg_acc, self.global_step)

        return outputs, avg_loss, avg_acc

    def _save_cfg_and_metrics(self, avg_test_loss, avg_test_acc):
        """Saves test results and config into CSV"""
        fieldnames = sorted(self.cfg.keys()) + ["avg_test_loss", "avg_test_acc"]
        with open(self.cfg.output_dir / "results.csv", "w") as f:
            csvwriter = csv.DictWriter(f, fieldnames=fieldnames)
            csvwriter.writeheader()
            csvwriter.writerow(dict({"avg_test_loss": avg_test_loss, "avg_test_acc": avg_test_acc}, **self.cfg))

    def save(self, checkpoint_path: Path):
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "sched_state_dict": self.sched.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def load(self, checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path)
        logger.info(f"Load checkpoint: {checkpoint_path} from epoch: {ckpt['epoch']}")
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optim.load_state_dict(ckpt["optim_state_dict"])
        self.sched.load_state_dict(ckpt["sched_state_dict"])
