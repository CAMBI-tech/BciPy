from .datasets import EEGDataset, FolderDataset
from .data_utils import get_fake_data, setup_datasets

__all__ = [
    "setup_datasets",
    "EEGDataset",
    "FolderDataset",
    "get_fake_data",
]
