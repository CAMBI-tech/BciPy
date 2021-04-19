from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from dataclasses import dataclass

import numpy as np


class SignalModel(ABC):
    @abstractmethod
    def fit(self, training_data: np.array, training_labels: np.array) -> SignalModel:
        """
        Train the model using the provided data and labels.
        Return self for convenience.
        """
        ...

    @abstractmethod
    def evaluate(self, test_data: np.array, test_labels: np.array) -> ModelEvaluationReport:
        """Compute model performance characteristics on the provided test data and labels."""
        ...

    @abstractmethod
    def predict(self, data: np.array, presented_symbols: List[str], all_symbols: List[str]) -> np.array:
        """Using the provided data, compute log likelihoods over the entire symbol set."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model state to the provided checkpoint"""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model state from the provided checkpoint"""
        ...

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Might be useful for confirming that model.load(model.save) == model"""
        ...


@dataclass
class ModelEvaluationReport:
    """
    Describes model performance characteristics.

    TODO - add more attributes as needed. Note that the model
    may not know about how its outputs are applied to compute a decision rule,
    so the model alone cannot compute AUC unless `model.evaluate` is also invoked
    with a callback or something. (This would get a bit finnicky).
    """

    auc: float
