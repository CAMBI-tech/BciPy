from pathlib import Path
from typing import Callable, Tuple, Union

import torch
from loguru import logger
from torch.utils.data import Dataset

from bcipy.signal.model.neural_net.utils import adjust_length


class EEGDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, transform: Callable = lambda x: x):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.labels[idx]


def get_fake_data(N, channels, classes, length) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(N, channels, length).float()
    y = torch.randint(classes, (N,)).long()
    return x, y


def get_data_from_folder(
    folder: Union[str, Path], subfolder: str, length: int, length_tol: int, shuffle=True, quick_test=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_list = []
    labels_list = []

    subjects = [x for x in Path(folder).iterdir() if x.is_dir()]
    for s in subjects:
        try:
            d = torch.load(s / subfolder / "data.pt")
        except FileNotFoundError:
            continue  # TODO - for now, ignore empty folders

        if d.shape[-1] != length:
            if abs(length - d.shape[-1]) <= length_tol:
                d = adjust_length(d, length)
            else:
                continue  # TODO - for now, just skipping other data...

        lab = torch.load(s / subfolder / "labels.pt")
        if lab.shape[0] != d.shape[0]:
            logger.warning(f"SHAPE MISMATCH! subject: {s}, data: {d.shape}, labels: {lab.shape}")
            continue

        data_list.append(d)
        labels_list.append(lab)

        if quick_test and len(data_list) > 10:  # During quick test, break after locating several usable samples
            break

    data = torch.cat(data_list, dim=0).float()
    labels = torch.cat(labels_list, dim=0).long()

    if shuffle:
        r = torch.randperm(data.shape[0])
        data = data[r, ...]
        labels = labels[r, ...]

    return data, labels
