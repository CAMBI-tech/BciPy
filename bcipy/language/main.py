"""Defines the language model base class."""
from abc import abstractmethod
from typing import List, Optional, Protocol, Tuple, Union, runtime_checkable


class UsesSymbols(Protocol):
    """A protocol for classes in which symbols can be set."""
    symbol_set: Optional[List[str]] = None

    @abstractmethod
    def set_symbol_set(self, symbol_set: List[str]) -> None:
        """Updates the symbol set of the model. Must be called prior to prediction"""


@runtime_checkable
class CharacterLanguageModel(UsesSymbols, Protocol):
    """Protocol for BciPy Language models that predict characters."""

    @abstractmethod
    def predict_character(self, evidence: Union[str,
                                                List[str]]) -> List[Tuple]:
        """
        Using the provided data, compute the probability distribution over the entire symbol set.
        Args:
            evidence - ['H', 'E']

        Response:
            probability - a list of symbols with probability
        """


@runtime_checkable
class WordLanguageModel(UsesSymbols, Protocol):
    """Protocol for BciPy Language models that predict words."""

    @abstractmethod
    def predict_word(self, evidence: Union[str, List[str]],
                     num_predictions: int) -> List[Tuple]:
        """
        Using the provided data, compute log likelihoods of word completions
        in the symbol set.
        Args:
            evidence - ['H', 'E']

        Response:
            a list of words with associated log likelihoods
        """


LanguageModel = Union[CharacterLanguageModel, WordLanguageModel]
