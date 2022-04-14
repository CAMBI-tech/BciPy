from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import numpy as np
from bcipy.helpers.stimuli import Reshaper


class SignalModel(ABC):
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
