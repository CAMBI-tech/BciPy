from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from bcipy.helpers.task import InquiryReshaper
from bcipy.signal.exceptions import SignalException
from bcipy.signal.model import ModelEvaluationReport, SignalModel
from bcipy.signal.model.neural_net.data import EEGDataset
from bcipy.signal.model.neural_net.models import ResNet1D
from bcipy.signal.model.neural_net.probability import compute_log_likelihoods
from bcipy.signal.model.neural_net.trainer import Trainer
from loguru import logger
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import jsonpickle
from bcipy.signal.model.neural_net.config import Config

MODEL_CHECKPOINT_NAME = "model_checkpoint.pt"


class EegClassifierModel(SignalModel):
    """Discriminative signal model for EEG"""
    reshaper = InquiryReshaper()

    def __init__(self, cfg: Config):
        """
        Args:
            cfg (Config): <TODO>
            n_classes (int): number of stimuli in inquiry.
                class [0, K-1] represent symbols presented, and [K] represents all other possible symbols
            n_channels (int): number of channels in input EEG data
        """
        self.cfg = cfg
        self.model = ResNet1D(
            layers=[2, 2, 2, 2],
            num_classes=cfg.n_classes,
            in_channels=cfg.n_channels,
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
        self.trainer.fit(EEGDataset(x_train, y_train), EEGDataset(x_val, y_val))
        self._ready_to_predict = True

    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> ModelEvaluationReport:
        """Compute model AUROC

        Args:
            test_data (np.array): shape (sequences, samples) - EEG of each sequence
            test_labels (np.array): shape (sequences) - integer label of each sequence

        Raises:
            SignalException: if model was not fit first.

        Returns:
            ModelEvaluationReport: stores AUROC
        """
        if not self._ready_to_predict:
            raise SignalException("call model.fit() before model.evaluate()")

        logger.info("Begin evaluate")
        outputs = self.trainer.test(EEGDataset(test_data, test_labels))
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
        logger.warning("TODO - estimate alpha parameter")
        return compute_log_likelihoods(
            model_log_probs=self.model(torch.tensor(data).unsqueeze(0)),
            alphabet_len=len(symbol_set),
            presented_seq_idx=torch.tensor([symbol_set.index(letter) for letter in inquiry]).unsqueeze(0),
            none_class_idx=0,
            alpha=0.8,
        )

    def save(self, folder: Path):
        # Save config
        config_path = folder / "config.json"
        logger.debug(f"Save config to {config_path}")
        self.cfg.save(config_path)

        # Save weights, optim, sched
        weights_ckpt = folder / MODEL_CHECKPOINT_NAME
        logger.debug(f"Save weights to {weights_ckpt}")
        self.trainer.save(weights_ckpt)

        # Save model description
        model_arch = folder / "model_arch.txt"
        logger.debug(f"Save model arch description to {model_arch}")
        with open(model_arch, "w") as f:
            f.write(str(self.model))

    def load(self, folder: Path):
        # Load config and re-init
        config_path = folder / "config.json"
        config = Config.load(config_path)
        self.__init__(config)

        # Load trainer state, including model weights
        self.trainer.load(folder / MODEL_CHECKPOINT_NAME)
        self._ready_to_predict = True
