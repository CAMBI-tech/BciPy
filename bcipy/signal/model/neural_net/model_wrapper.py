from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from dotted_dict import DottedDict
from loguru import logger
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

from bcipy.signal.model.neural_net.data import EEGDataset, data_config, get_transform
from bcipy.signal.model.neural_net.models import ResNet1D
from bcipy.signal.model.neural_net.probability import compute_log_likelihoods
from bcipy.signal.model.neural_net.trainer import Trainer

from bcipy.signal.model import ModelEvaluationReport, SignalModel
from bcipy.signal.exceptions import SignalException


def model_from_checkpoint(path: Path, **cfg_overrides):
    config_yaml = path / "config.yaml"
    if not config_yaml.exists():
        raise ValueError("config yaml not found!")

    with open(config_yaml, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    for key, val in cfg_overrides.items():
        try:
            cfg[key] = val
        except KeyError:
            raise ValueError(f"Invalid override: key {key} not in config")

    model = EegClassifierModel(cfg)
    model.load(path)
    return model, cfg


class EegClassifierModel(SignalModel):
    """Discriminative signal model for EEG"""

    def __init__(self, cfg: DottedDict):
        self.cfg = cfg
        self.model = ResNet1D(
            layers=[2, 2, 2, 2],
            num_classes=data_config[cfg.data_device][cfg.data_mode]["n_classes"],
            in_channels=data_config[cfg.data_device]["n_channels"],
            act_name=cfg.activation,
            device=cfg.device,
        )
        self.trainer = Trainer(cfg, self.model)
        self._ready_to_predict = False

    def __repr__(self):
        return f"{self.__class__}" + f"\n\nConfig: {self.cfg}" + f"\n\nModel: {self.model}"

    def fit(self, train_data: np.ndarray, train_labels: np.ndarray):
        """Train model using `train_data` and `train_labels`

        Args:
            train_data (np.array): shape (sequences, samples) - EEG of each sequence
            train_labels (np.array): shape (sequences) - integer label of each sequence
        """

        logger.info("Begin fit")
        # Split off val set for early stopping
        x_train, x_val, y_train, y_val = train_test_split(
            train_data, train_labels, test_size=self.cfg.val_frac, random_state=self.cfg.seed
        )
        transform = get_transform(self.cfg)
        self.trainer.fit(EEGDataset(x_train, y_train, transform), EEGDataset(x_val, y_val, transform))
        self._ready_to_predict = True

    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> ModelEvaluationReport:
        """Compute model AUROC

        Args:
            test_data (np.array): shape (sequences, samples) - EEG of each sequence
            test_labels (np.array): shape (sequences) - integer label of each sequence

        Returns:
            ModelEvaluationReport: stores AUROC
        """
        if not self._ready_to_predict:
            raise SignalException("call model.fit() before model.evaluate()")

        logger.info("Begin evaluate")
        outputs = self.trainer.test(EEGDataset(test_data, test_labels, get_transform(self.cfg)))
        predicted_probs = torch.cat([x["log_probs"].exp() for x in outputs]).cpu()

        logger.info(f"Confusion matrix: \n{confusion_matrix(test_labels, predicted_probs.argmax(1))}")

        if self.cfg.data_mode == "trials":
            result = ModelEvaluationReport(roc_auc_score(test_labels, predicted_probs[..., -1]))

        else:  # "sequences"
            # NOTE
            # - sklearn determine whether this is "multiclass" data by checking number of distinct items:
            #   https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py#L523
            #   Thus this crashes if only 2 classes are present (because it assumes binary classification, and then
            #   y_pred shape is wrong)
            # - roc_auc_score requires that each class appears at least once.
            #   A single fake data item is added to fulfill this requirement, until training
            #   data representing all classes is available.
            logger.warning("ADDING A FAKE DATA ITEM (necessary for AUC due to missing class)")
            fake_label = torch.tensor([0])
            fake_prediction = torch.zeros(1, predicted_probs.shape[-1])
            fake_prediction[0, -1] = 1

            all_labels = torch.cat([torch.tensor(test_labels), fake_label])
            predicted_probs = torch.cat([predicted_probs, fake_prediction])

            # TODO - ovr or ovo?
            result = ModelEvaluationReport(roc_auc_score(all_labels, predicted_probs, multi_class="ovr"))

        return result

    @torch.no_grad()
    def predict(self, data: np.ndarray, inquiry: List[str], symbol_set: List[str]) -> np.ndarray:
        """Compute log likelihood updates after an inquiry

        Args:
            data (np.array): shape (channels, samples) - user's EEG response
            inquiry (List[str]): presented letters
            symbol_set (List[str]): full alphabet

        Returns:
            np.array: log likelihoods for full alphabet
        """
        if not self._ready_to_predict:
            raise SignalException("call model.fit() before model.predict()")

        logger.info("Begin predict")
        # distinct letters
        assert len(set(inquiry)) == len(inquiry)
        assert len(set(symbol_set)) == len(symbol_set)

        self.model.eval()
        return compute_log_likelihoods(
            model_log_probs=self.model(torch.tensor(data).unsqueeze(0)),
            alphabet_len=len(symbol_set),
            presented_seq_idx=torch.tensor([symbol_set.index(letter) for letter in inquiry]).unsqueeze(0),
            none_class_idx=0,
            alpha=0.8,  # TODO
        )

    def save(self, path: Path):
        # Save config
        config_yaml = path / "config.yaml"
        logger.debug(f"Save config to {config_yaml}")
        with open(config_yaml, "w") as f:
            yaml.dump(self.cfg, f)

        # Save weights
        # TODO
        logger.warning("Need to figure out how this interacts with Trainer.save()")
        # weights_ckpt = path / "model_weights.pt"
        # logger.debug(f"Save weights to {weights_ckpt}")
        # self.trainer.save(weights_ckpt)

        # Save model description
        model_arch = path / "model_arch.txt"
        logger.debug(f"Save model arch description to {model_arch}")
        with open(model_arch, "w") as f:
            f.write(str(self.model))

    def load(self, path: Path):
        """
        Args:
            path (Path): Folder containing trained model outputs
        """
        self.trainer.load(self.trainer.final_checkpoint)
        self._ready_to_predict = True
