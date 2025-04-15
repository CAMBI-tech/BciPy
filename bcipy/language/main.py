"""Defines the language model base class."""
from abc import abstractmethod
from typing import List, Tuple, Protocol, Union


class LanguageModel(Protocol):
    """Protocol for BciPy Language Models."""

    symbol_set: List[str] = None

    @abstractmethod
    def set_symbol_set(self, symbol_set: List[str]) -> None:
        """Updates the symbol set of the model. Must be called prior to prediction"""
        ...


class CharacterLanguageModel(LanguageModel, Protocol):

    @abstractmethod
    def predict_character(self, evidence: Union[str, List[str]]) -> List[Tuple]:
        """
        Using the provided data, compute the probability distribution over the entire symbol set.
        Args:
            evidence - ['H', 'E']

        Response:
            probability - a list of symbols with probability
        """
        ...

class WordLanguageModel(LanguageModel, Protocol):

    @abstractmethod
    def predict_word(self, evidence: Union[str, List[str]], num_predictions: int) -> List[Tuple]:
        """
        Using the provided data, compute log likelihoods of word completions
        in the symbol set.
        Args:
            evidence - ['H', 'E']
            
        Response:
            a list of words with associated log likelihoods
        """
        ...