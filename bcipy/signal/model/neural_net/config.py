import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from loguru import logger

from typing import Optional
import random
import numpy as np
import torch

import jsonpickle


def get_git_hash():
    h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")

    exitcode, _ = subprocess.getstatusoutput(["git", "diff-index", "--quiet", "HEAD"])
    if exitcode != 0:
        h += "+"  # Add suffix if local files modified
    return "git" + h


def default_run_name():
    t = datetime.now().isoformat("_", "seconds")
    h = get_git_hash()
    return t + "_" + h


def seed_everything(s: Optional[int] = None) -> int:
    if s is None:
        s = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)

    logger.info(f"Using seed: {s}")
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    return s


@dataclass
class Config:
    activation: str = 'SiLU'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    alphabet_len: int = 28
    batch_size: int = 128
    data_device: str = 'dsi'
    data_dir: Path = Path(__file__).absolute().parent / '../../../../../VisualERPDatasets/Data/Extracted'
    data_mode: str = 'sequences'
    device: str = field(default_factory=lambda: 'cuda:0' if torch.cuda.is_available() else 'cpu')
    early_stop_patience: int = 15
    epochs: int = 300
    fake_data: bool = False
    final_lr: float = 1e-08
    initial_lr: float = 0.0005
    loader_num_workers: int = 6
    optimizer: str = 'AdamW'
    output_dir: Path = Path(__file__).absolute().parent / "runs" / default_run_name()
    quick_test: bool = False
    scheduler: str = 'CosineAnnealingLR'
    seed: int = field(default_factory=seed_everything)
    test_frac: float = 0.05
    use_cuda: bool = True
    use_early_stop: bool = True
    val_frac: float = 0.05
    n_classes: int = 14 + 1  # inquiry_length + 1
    n_channels: int = 20  # DSI channels
    length: int = 465  # number of samples per inquiry
    length_tol: int = 5  # allow +/- this amount of length

    def __post_init__(self):
        if self.quick_test:
            logger.warning("QUICK TEST")
            self.output_dir = Path(__file__).absolute().parent / "test_runs_deleteme" / default_run_name()
            self.epochs = 1

    def save(self, path: Path):
        with open(path, "w") as f:
            json_str = jsonpickle.encode(asdict(self))
            f.write(json_str)

    @classmethod
    def load(cls, path: Path):
        with open(path, "r") as f:
            contents = jsonpickle.decode(f.read())
            return cls(**contents)
