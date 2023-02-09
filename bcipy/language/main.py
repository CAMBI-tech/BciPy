"""Defines the language model base class."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple
from string import ascii_uppercase

import os

from bcipy.helpers.exceptions import UnsupportedResponseType

SPACE_CHAR = '_'
BACKSPACE_CHAR = '<'

def alphabet(parameters=None, include_path=True):
    """Alphabet.

    Function used to standardize the symbols we use as alphabet.

    Returns
    -------
        array of letters.
    """
    if parameters and not parameters['is_txt_stim']:
        # construct an array of paths to images
        path = parameters['path_to_presentation_images']
        stimulus_array = []
        for stimulus_filename in sorted(os.listdir(path)):
            # PLUS.png is reserved for the fixation symbol
            if stimulus_filename.endswith(
                    '.png') and not stimulus_filename.endswith('PLUS.png'):
                if include_path:
                    img = os.path.join(path, stimulus_filename)
                else:
                    img = os.path.splitext(stimulus_filename)[0]
                stimulus_array.append(img)
        return stimulus_array

    return list(ascii_uppercase) + [BACKSPACE_CHAR, SPACE_CHAR]

    
DEFAULT_SYMBOL_SET = alphabet()


class ResponseType(Enum):
    """Language model response type options."""
    SYMBOL = 'Symbol'
    WORD = 'Word'

    def __str__(self):
        return self.value


class LanguageModel(ABC):
    """Parent class for Language Models."""

    _response_type: ResponseType = None
    symbol_set: List[str] = None

    def __init__(self,
                 response_type: Optional[ResponseType] = None,
                 symbol_set: Optional[List[str]] = None):
        self.response_type = response_type or ResponseType.SYMBOL
        self.symbol_set = symbol_set or DEFAULT_SYMBOL_SET

    @classmethod
    def name(cls) -> str:
        """Model name used for configuration"""
        suffix = 'LanguageModel'
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
