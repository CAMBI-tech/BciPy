from .datasets import EEGDataset
from .data_utils import get_fake_data, setup_datasets

__all__ = [
    "setup_datasets",
    "EEGDataset",
    "get_fake_data",
]
