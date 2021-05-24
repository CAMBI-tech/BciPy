from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from enum import Enum

import numpy as np


class InputDataType(Enum):
    """Controls how data is reshaped before being passed to a model"""
    TRIAL = 1
    INQUIRY = 2


class SignalModel(ABC):
    @property
    @abstractmethod
    def input_data_type(self) -> InputDataType:
        ...

    @abstractmethod
    def fit(self, training_data: np.array, training_labels: np.array):
        """
        Train the model using the provided data and labels.
        Return self for convenience.
        """
        ...

    @abstractmethod
    def evaluate(self, test_data: np.array, test_labels: np.array):
        """Compute model performance characteristics on the provided test data and labels."""
        ...

    @abstractmethod
    def predict(self, data: np.array, inquiry: List[str], symbol_set: List[str]) -> np.array:
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
