"""Uniform language model"""
from typing import Dict, List, Tuple, Union, Optional

import numpy as np

from bcipy.language.main import LanguageModel, ResponseType


class UniformLanguageModel(LanguageModel):
    """Language model in which probabilities for symbols are uniformly
    distributed.

    Parameters
    ----------
        response_type - SYMBOL only
        symbol_set - optional specify the symbol set, otherwise uses DEFAULT_SYMBOL_SET
    """

    def __init__(self,
                 response_type: Optional[ResponseType] = None,
                 symbol_set: Optional[List[str]] = None):
        super().__init__(response_type=response_type, symbol_set=symbol_set)

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    def predict(self, evidence: Union[str, List[str]]) -> List[Tuple]:
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
        probs = equally_probable(self.symbol_set)
        return list(zip(self.symbol_set, probs))

    def update(self) -> None:
        """Update the model state"""

    def load(self) -> None:
        """Restore model state from the provided checkpoint"""


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
