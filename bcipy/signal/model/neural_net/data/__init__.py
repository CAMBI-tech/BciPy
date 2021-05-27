from .data_config import data_config
from .datasets import EEGDataset
from .utils import get_fake_data, setup_datasets

__all__ = [
    "data_config",
    "setup_datasets",
    "EEGDataset",
    "get_fake_data",
]
