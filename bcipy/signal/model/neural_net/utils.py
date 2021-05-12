import math
import random
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from dotted_dict import DottedDict
from loguru import logger
from torch import nn
from yaml import safe_load

default_config_path = Path(__file__).absolute().parent / "config.yaml"


def parse_args() -> DottedDict:
    # Parse location of config file
    conf_parser = ArgumentParser(add_help=False)
    conf_parser.add_argument(
        "--config",
        help="Path to config YAML",
        metavar="FILE",
        default=default_config_path,
    )
    known_args, remaining_args = conf_parser.parse_known_args()

    # Get defaults from config file
    default_config = parse_config_from_file(known_args.config)

    # Allow overrides on command line
    cli_parser = ArgumentParser(parents=[conf_parser])
    cli_parser.set_defaults(**default_config)
    cli_parser.add_argument("--quick-test", action="store_true")
    cli_parser.add_argument("--fake-data", action="store_true")
    cli_parser.add_argument("--data-mode", choices=["trials", "sequences"], default="sequences")
    cli_parser.add_argument("--data-device", choices=["dsi", "gtec"], default="dsi")
    cli_parser.add_argument("--test-frac", type=float)
    args = cli_parser.parse_args(remaining_args)
    cfg = DottedDict(vars(args))
    return postprocess_args(cfg)


def parse_config_from_file(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        contents = safe_load(f)

    return dict((k.replace("-", "_"), v) for k, v in contents.items())


def postprocess_args(cfg: DottedDict) -> DottedDict:
    module_dir = Path(__file__).absolute().parent

    if cfg.output_dir is None:
        cfg.output_dir = module_dir / "runs" / default_run_name()

    cfg.seed = seed_everything(cfg.seed)
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # Last step - downgrade options for quick test mode
    if cfg.quick_test:
        logger.warning("QUICK TEST")
        cfg.epochs = 1
        cfg.output_dir = module_dir / "test_runs_deleteme" / default_run_name()

    return cfg


def get_default_cfg():
    default_config = parse_config_from_file(default_config_path)
    cfg = DottedDict(default_config)
    return postprocess_args(cfg)


def get_activation(name: str):
    try:
        return getattr(nn, name)(inplace=True)
    except TypeError:
        return getattr(nn, name)()


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


def seed_everything(s: Optional[int] = None) -> int:
    if s is None:
        s = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)

    logger.info(f"Using seed: {s}")
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    return s
