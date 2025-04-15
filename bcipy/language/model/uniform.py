"""Uniform language model"""
from typing import List, Optional, Union, Tuple, Dict

from bcipy.language.main import CharacterLanguageModel
from bcipy.core.symbols import BACKSPACE_CHAR
from bcipy.exceptions import InvalidSymbolSetException

import numpy as np


class UniformLanguageModel(CharacterLanguageModel):
    """Language model in which probabilities for symbols are uniformly
    distributed.

    Parameters
    ----------
        None
    """

    def __init__(self):
        self.symbol_set = None

    def set_symbol_set(self, symbol_set: List[str]) -> None:
        self.symbol_set = symbol_set

        self.model_symbol_set = [ch for ch in symbol_set]
        self.model_symbol_set.remove(BACKSPACE_CHAR)
    
    def predict_character(self, evidence: Union[str, List[str]]) -> List[Tuple]:
        """
        Using the provided data, compute probabilities over the entire symbol.
        set.

        Parameters
        ----------
            evidence  - list of previously typed symbols

        Returns
        -------
            list of (symbol, probability) tuples
        """

        if self.symbol_set is None:
            raise InvalidSymbolSetException("symbol set must be set prior to requesting predictions.")

        probs = equally_probable(self.model_symbol_set)
        return list(zip(self.model_symbol_set, probs)) + [(BACKSPACE_CHAR, 0.0)]

def equally_probable(alphabet: List[str],
                     specified: Optional[Dict[str, float]] = None) -> List[float]:
    """Returns a list of probabilities which correspond to the provided
    alphabet. Unless overridden by the specified values, all items will
    have the same probability. All probabilities sum to 1.0.

    Parameters:
    ----------
        alphabet - list of symbols; a probability will be generated for each.
        specified - dict of symbol => probability values for which we want to
            override the default probability.
    Returns:
    --------
        list of probabilities (floats)
    """
    n_letters = len(alphabet)
    if not specified:
        return np.full(n_letters, 1 / n_letters)

    # copy specified dict ignoring non-alphabet items
    overrides = {k: specified[k] for k in alphabet if k in specified}
    assert sum(overrides.values()) < 1

    prob = (1 - sum(overrides.values())) / (n_letters - len(overrides))
    # override specified values
    return [overrides[sym] if sym in overrides else prob for sym in alphabet]