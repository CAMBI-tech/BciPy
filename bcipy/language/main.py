from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List
from enum import Enum


class ResponseType(Enum):
    SYMBOL = 'Symbol'
    WORD = 'Word'


class LanguageModel(ABC):

    response_type: ResponseType
    symbol_set: List[str]
    normalized: bool = False  # normalized to probability domain

    @abstractmethod
    def predict(self, evidence: List[str]) -> List[Tuple]:
        """
        Using the provided data, compute log likelihoods over the entire symbol set.
        Args:
            evidence - ['A', 'B']

        Response:
            probability - dependant on response type, a list of words or symbols with probability
        """
        ...

    @abstractmethod
    def update(self) -> None:
        """Update the model state"""
        ...

    @abstractmethod
    def load(self) -> None:
        """Restore model state from the provided checkpoint"""
        ...

    def reset(self) -> None:
        """Reset language model state"""
        ...

    def state_update(self, evidence: List[str]) -> List[Tuple]:
        """Update state by predicting and updating"""
        ...
