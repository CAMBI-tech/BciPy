from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat

import colored_traceback
from loguru import logger

from bcipy.signal.model.neural_net.data import setup_datasets
from bcipy.signal.model.neural_net.model_wrapper import EegClassifierModel, model_from_checkpoint
from bcipy.signal.model.neural_net.utils import parse_args

colored_traceback.add_hook(always=True)


def train():
    cfg = parse_args()
    logger.add(cfg.output_dir / "log.txt")
    logger.info(pformat(cfg))

    x_train, x_test, y_train, y_test = setup_datasets(cfg)
    logger.info(f"Data sizes. Train: {len(x_train)}, Test: {len(x_test)}")

    eeg_model = EegClassifierModel(cfg)
    eeg_model.fit(x_train, y_train)
    eeg_model.save(cfg.output_dir)
    report = eeg_model.evaluate(x_test, y_test)
    logger.info(report)


def evaluate():
    parser = ArgumentParser()
    parser.add_argument("--path", required=True, type=Path, help="Path to trained model folder")
    parser.add_argument("--fake-data", action="store_true")
    args = parser.parse_args()
    eeg_model, cfg = model_from_checkpoint(args.path, fake_data=args.fake_data)

    # TODO - ensure dataset creation uses known fixed seed, to get same AUROC
    _, x_test, _, y_test = setup_datasets(cfg)
    logger.info(eeg_model)
    report = eeg_model.evaluate(x_test, y_test)
    logger.info(report)


if __name__ == "__main__":
    train()
