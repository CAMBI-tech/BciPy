"""Defines the language model adapter base class."""
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from bcipy.exceptions import InvalidSymbolSetException
from bcipy.core.symbols import SPACE_CHAR, BACKSPACE_CHAR
from bcipy.config import DEFAULT_LM_PARAMETERS_PATH
import json

class LanguageModelAdapter(ABC):
    """Abstract base class for textslinger language model adapters."""

    symbol_set: List[str] = None
    model = None

    def predict_character(self, evidence: Union[str, List[str]]) -> List[Tuple]:
        """
        Using the provided data, compute the probability distribution over the entire symbol set.
        Args:
            evidence - ['H', 'E']

        Response:
            probability - a list of symbols with probability
        """

        if self.symbol_set is None:
            raise InvalidSymbolSetException("symbol set must be set prior to requesting predictions.")
        
        assert self.model is not None, "language model does not exist!"

        context = "".join(evidence)
        converted_context = context.replace(SPACE_CHAR, ' ')

        # TODO: If toolkit dependency is updated to >=1.0.0, this will need to change to predict_character()
        next_char_pred = dict(self.model.predict(list(converted_context)))

        # Replace space with special space
        if ' ' in next_char_pred:
            next_char_pred[SPACE_CHAR] = next_char_pred[' ']
            del next_char_pred[' ']

        # Add backspace, but return prob 0 from the lm
        next_char_pred[BACKSPACE_CHAR] = 0.0

        return list(sorted(next_char_pred.items(),
                    key=lambda item: item[1], reverse=True))
    

    def _load_parameters(self) -> None:
        with open(DEFAULT_LM_PARAMETERS_PATH, 'r') as params_file:
            self.parameters = json.load(params_file)


    @abstractmethod
    def _load_model(self) -> None:
        """Load the model itself using stored parameters"""
        ...


    def set_symbol_set(self, symbol_set: List[str]) -> None:
        """Update the symbol set and call for the model to be loaded"""
        
        self.symbol_set = symbol_set
        
        # LM doesn't care about backspace, needs literal space
        self.model_symbol_set = [' ' if ch is SPACE_CHAR else ch for ch in self.symbol_set]
        self.model_symbol_set.remove(BACKSPACE_CHAR)

        self._load_model()