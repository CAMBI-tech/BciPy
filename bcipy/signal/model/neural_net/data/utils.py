from pathlib import Path

from sklearn.model_selection import train_test_split

from .data_config import data_config
from .datasets import get_data_from_folder, get_fake_data

wd = Path(__file__).absolute().parent.parent.parent.parent


def load_data(cfg):
    x, y = get_data_from_folder(
        folder=cfg.data_dir,
        subfolder=cfg.data_mode,
        length=data_config[cfg.data_device][cfg.data_mode]["length"],
        length_tol=data_config[cfg.data_device][cfg.data_mode]["length_tol"],
        shuffle=True,
        quick_test=cfg.quick_test,
    )
    return x, y


def setup_datasets(cfg):
    all_x, all_y = load_data(cfg)
    x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=cfg.test_frac, random_state=cfg.seed)

    if cfg.fake_data:  # Fake test data, to demonstrate that classification fails
        x_test, y_test = get_fake_data(
            N=100,
            channels=data_config[cfg.data_device]["n_channels"],
            classes=data_config[cfg.data_device][cfg.data_mode]["n_classes"],
            length=data_config[cfg.data_device][cfg.data_mode]["length"],
        )

    if cfg.quick_test:
        x_train, y_train = x_train[:100, ...], y_train[:100, ...]
        x_test, y_test = x_test[:100, ...], y_test[:100, ...]

    return x_train, x_test, y_train, y_test
