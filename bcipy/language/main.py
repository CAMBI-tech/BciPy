"""Defines the language model base class."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple
import json

from bcipy.exceptions import UnsupportedResponseType
from bcipy.core.symbols import SPACE_CHAR, BACKSPACE_CHAR
from bcipy.config import DEFAULT_LM_PARAMETERS_PATH


class ResponseType(Enum):
    """Language model response type options."""
    SYMBOL = 'Symbol'
    WORD = 'Word'

    def __str__(self):
        return self.value


class LanguageModelAdapter(ABC):
    """Parent class for Language Model Adapters."""

    _response_type: ResponseType = None
    symbol_set: List[str] = None

    def __init__(self,
                 response_type: Optional[ResponseType] = None):
        self.response_type = response_type or ResponseType.SYMBOL
        with open(DEFAULT_LM_PARAMETERS_PATH, 'r') as params_file:
            self.parameters = json.load(params_file)

    @classmethod
    def name(cls) -> str:
        """Model name used for configuration"""
        suffix = 'LanguageModelAdapter'
        if cls.__name__.endswith(suffix):
            return cls.__name__[0:-len(suffix)].upper()
        return cls.__name__.upper()

    @abstractmethod
    def supported_response_types(self) -> List[ResponseType]:
        """Returns a list of response types supported by this language model."""

    @property
    def response_type(self) -> ResponseType:
        """Returns the current response type"""
        return self._response_type

    @response_type.setter
    def response_type(self, value: ResponseType):
        """Attempts to set the response type to the given value"""
        if value not in self.supported_response_types():
            raise UnsupportedResponseType(
                f"{value} responses are not supported by this model")
        self._response_type = value

    def predict(self, evidence: List[str]) -> List[Tuple]:
        """
        Using the provided data, compute log likelihoods over the entire symbol set.
        Args:
            evidence - ['A', 'B']

        Response:
            probability - dependant on response type, a list of words or symbols with probability
        """
        assert self.model is not None, "language model does not exist!"

        context = "".join(evidence)
        converted_context = context.replace(SPACE_CHAR, ' ')

        next_char_pred = dict(self.model.predict(list(converted_context)))

        # Replace space with special space
        if ' ' in next_char_pred:
            next_char_pred[SPACE_CHAR] = next_char_pred[' ']
            del next_char_pred[' ']

        # Add backspace, but return prob 0 from the lm
        next_char_pred[BACKSPACE_CHAR] = 0.0

        return list(sorted(next_char_pred.items(),
                    key=lambda item: item[1], reverse=True))
