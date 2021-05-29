import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bcipy.signal.model.neural_net.config import Config

DEFAULT_CONFIG_PATH = Path(__file__).absolute().parent / "config.yaml"


def parse_args() -> Config:
    # Begin with hard-coded defaults
    config = Config()

    # Allow overrides on command line
    parser = ArgumentParser()
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--data-mode", choices=["trials", "sequences"], default="sequences")
    parser.add_argument("--data-device", choices=["dsi", "gtec"], default="dsi")
    parser.add_argument("--test-frac", type=float)
    args = parser.parse_args()
    for key, val in vars(args).items():
        if not hasattr(config, key):
            raise ValueError(f"Invalid arg: {key} with value: {val}")
        setattr(config, key, val)

    return config


def get_activation(name: str):
    try:
        return getattr(nn, name)(inplace=True)
    except TypeError:
        return getattr(nn, name)()


def get_decay_rate(initial_lr, final_lr, epochs):
    """
    >>> 0.1 == get_decay_rate(1e-4, 1e-8, 4)
    True
    """
    if final_lr == 0:
        final_lr = initial_lr / 1e8

    # final_lr = initial_lr * gamma ** epochs
    # final_lr / initial_lr = gamma ** epochs
    # log ( final_lr / initial_lr ) / epochs = log ( gamma )
    # gamma = exp [ log ( final_lr / initial_lr ) / epochs ]

    gamma = math.exp(math.log(final_lr / initial_lr) / epochs)
    assert abs(gamma ** epochs * initial_lr - final_lr) < 1e-8
    return gamma


def adjust_length(data, length):
    if data.shape[-1] > length:
        return trim(data, length)
    else:
        return pad(data, length)


def trim(data: np.ndarray, length: int, which="begin"):
    if which == "begin":
        return data[..., -length:]
    elif which == "end":
        return data[..., :length]
    else:
        raise ValueError("Invalid choice:", which)


def pad(data: np.ndarray, length: int, which="end", pad_mode="edge") -> torch.Tensor:
    """
    See options for `pad_mode` at: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    """
    ndim = len(data.shape)
    if which == "begin":
        return torch.from_numpy(
            np.pad(
                data,
                (
                    *[(0, 0) for _ in range(ndim - 1)],  # no padding for first D-1 dimensions
                    (length - data.shape[-1], 0),  # pad final dim at front
                ),
                pad_mode,
            )
        )
    elif which == "end":
        return torch.from_numpy(
            np.pad(
                data,
                (
                    *[(0, 0) for _ in range(ndim - 1)],
                    (0, length - data.shape[-1]),
                ),
                pad_mode,
            )
        )
    else:
        raise ValueError("Invalid choice:", which)
