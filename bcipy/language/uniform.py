"""Uniform language model"""
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from bcipy.helpers.task import BACKSPACE_CHAR, alphabet
from bcipy.language.main import LanguageModel, ResponseType


class UniformLanguageModel(LanguageModel):
    """Language model in which probabilities for symbols are uniformly
    distributed. Symbols can be text or images. If images are used the
    path must be provided.

    Parameters
    ----------
        lm_backspace_prob - optionally used to set a fixed probability for the
            backspace symbol. Probabilities for other symbols will be adjusted.
        is_txt_stim - Symbols can be text or images. Set to False for images.
        path_to_presentation_images - path to images
    """

    def __init__(self,
                 lm_backspace_prob: float = None,
                 is_txt_stim: bool = True,
                 path_to_presentation_images: str = None):
        if lm_backspace_prob:
            assert 0 <= lm_backspace_prob < 1, "Backspace probability must be between 0 and 1"
        if not is_txt_stim:
            assert path_to_presentation_images, "Path must be provided for images."

        self.backspace_prob = lm_backspace_prob
        self.response_type = ResponseType.SYMBOL
        params = {
            'is_txt_stim': is_txt_stim,
            'path_to_presentation_images': path_to_presentation_images
        }
        self.symbol_set = alphabet(params)
        self.normalized = True

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
        overrides = None
        if self.backspace_prob:
            overrides = {BACKSPACE_CHAR: self.backspace_prob}
        probs = equally_probable(self.symbol_set, overrides)
        return list(zip(self.symbol_set, probs))

    def update(self) -> None:
        """Update the model state"""

    def load(self, path: Path) -> None:
        """Restore model state from the provided checkpoint"""


def equally_probable(alphabet: List[str],
                     specified: Dict[str, float] = None) -> List[float]:
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
