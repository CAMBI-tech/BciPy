"""Signal model for button presses."""
from pathlib import Path
from typing import List

import numpy as np

from bcipy.core.stimuli import InquiryReshaper
from bcipy.signal.model.base_model import SignalModel
from bcipy.signal.model.inquiry_preview import compute_probs_after_preview


class ButtonPressModel(SignalModel):
    """Signal model that classifies button presses.

    This is a demo model. The provided data should be comprised of ones and zeros,
    where a 1 indicates a button press occurred. If a 1.0 occurs any time within
    an inquiry, all symbols in the inquiry are supported and all non-inquiry symbols
    are downgraded. Otherwise the opposite happens. See tests for
    inquiry_preview.compute_probs_after_preview.

    Parameters
    ----------
        error_prob - Specifies the probability of a button press error.
    """

    name = "ButtonPressModel"
    reshaper: InquiryReshaper = InquiryReshaper()

    def __init__(self, error_prob: float = 0.05):
        self.error_prob = error_prob

    def fit(self, training_data: np.ndarray, training_labels: np.ndarray):
        """
        @override
        """
        return self

    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray):
        """@override"""

    def predict(self, data: np.ndarray, inquiry: List[str],
                symbol_set: List[str]) -> np.ndarray:
        """@override"""
        return np.ones(data.shape)

    def compute_likelihood_ratio(self, data: np.array, inquiry: List[str],
                                 symbol_set: List[str]) -> np.array:
        """
        For each trial in `data`, compute a likelihood ratio to update that symbol's probability.

        Args:
            data (np.array): button press data data a single element of 0 or 1; shape (1,)
            inquiry (List[str]): List describing the symbol shown in each trial.
            symbol_set (List[str]): The set of all possible symbols.

        Returns:
            np.array: multiplicative update term (likelihood ratios) for each symbol in the `symbol_set`.
        """

        proceed = np.any(data)
        return compute_probs_after_preview(inquiry=inquiry,
                                           symbol_set=symbol_set,
                                           user_error_prob=self.error_prob,
                                           proceed=proceed)

    def compute_class_probabilities(self, data: np.ndarray) -> np.ndarray:
        """@override"""
        return np.ones(0)

    def evaluate_likelihood(self, data: np.ndarray) -> np.ndarray:
        """@override"""
        return np.ones(0)

    def save(self, path: Path) -> None:
        """@override"""

    def load(self, path: Path) -> None:
        """@override"""
