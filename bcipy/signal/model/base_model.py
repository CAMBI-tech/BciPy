from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, NamedTuple

import numpy as np

from bcipy.acquisition.devices import DeviceSpec
from bcipy.helpers.stimuli import Reshaper
from bcipy.signal.process import Composition


class SignalModelMetadata(NamedTuple):
    """Metadata about the SignalModel, including how the model was trained
    (device, filters, etc)."""

    device_spec: DeviceSpec  # device used to train the model
    transform: Composition  # data preprocessing steps
    evidence_type: str = None  # optional; type of evidence produced
    auc: float = None  # optional; area under the curve
    balanced_accuracy: float = None  # optional; balanced accuracy


class SignalModel(ABC):

    name = "undefined"

    @property
    def metadata(self) -> SignalModelMetadata:
        """Information regarding the data and parameters used to train the
        model."""
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """Set the metadata"""
        self._metadata = value

    @property
    @abstractmethod
    def reshaper(self) -> Reshaper:
        """Reshapes data into trials or inquiry as needed for each model."""
        ...

    @abstractmethod
    def fit(self, training_data: np.ndarray, training_labels: np.ndarray):
        """
        Train the model using the provided data and labels.
        Return self for convenience.
        """
        ...

    @abstractmethod
    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray):
        """Compute model performance characteristics on the provided test data and labels."""
        ...

    @abstractmethod
    def predict(self, data: np.ndarray, inquiry: List[str], symbol_set: List[str]) -> np.ndarray:
        """
        Using the provided data, compute log likelihoods over the entire symbol set.
        Args:
            inquiry - the subset of symbols presented
            symbol_set - the entire alphabet of symbols
        """
        ...

    def compute_likelihood_ratio(self, data: np.array, inquiry: List[str], symbol_set: List[str]) -> np.array:
        """
        For each trial in `data`, compute a likelihood ratio to update that symbol's probability.
        Rather than just computing an update p(data|l=+) for the seen symbol and p(data|l=-) for all unseen symbols,
        we compute a likelihood ratio p(data | l=+) / p(data | l=-) to update the seen symbol, and all other symbols
        can receive a multiplicative update of 1.

        Args:
            data (np.array): data with shape (n_channel, n_trial, n_sample).
            inquiry (List[str]): List describing the symbol shown in each trial.
            symbol_set (List[str]): The set of all possible symbols.

        Raises:
            SignalException: error if called before model is fit.

        Returns:
            np.array: multiplicative update term (likelihood ratios) for each symbol in the `symbol_set`.
        """
        ...

    def compute_class_probabilities(self, data: np.ndarray) -> np.ndarray:
        """Converts log likelihoods from model into class probabilities."""
        ...

    def evaluate_likelihood(self, data: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model state to the provided checkpoint"""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model state from the provided checkpoint"""
        ...


class ModelEvaluationReport:
    """
    Describes model performance characteristics.
    """

    def __init__(self, auc: float):
        self.auc = auc

    def __eq__(self, other):
        return np.allclose(self.auc, other.auc)
