"""Oracle language model"""
import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from bcipy.config import SESSION_LOG_FILENAME
from bcipy.core.symbols import BACKSPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.language.model.uniform import equally_probable

logger = logging.getLogger(SESSION_LOG_FILENAME)

TARGET_BUMP_MIN = 0.0
TARGET_BUMP_MAX = 0.95


class OracleLanguageModel(LanguageModel):
    """Language model which knows the target phrase the user is attempting to
    spell.

    Probabilities for symbols are uniformly distributed with the exception of
    the target letter, which has a slightly higher probability. How much higher
    depends on the configured parameter.

    After the target text has been correctly spelled subsequent predictions
    yield all symbols with equal probability.

    Parameters
    ----------
        response_type - SYMBOL only
        symbol_set - optional specify the symbol set, otherwise uses DEFAULT_SYMBOL_SET
        task_text - the phrase the user is attempting to spell (ex. 'HELLO_WORLD')
        target_bump - the amount by which the probability of the target letter
            is increased.
    """

    def __init__(self,
                 response_type: Optional[ResponseType] = None,
                 symbol_set: Optional[List[str]] = None,
                 task_text: str = None,
                 target_bump: float = 0.1):
        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.task_text = task_text
        self.target_bump = target_bump
        logger.debug(
            f"Initialized OracleLanguageModel(task_text='{task_text}', target_bump={target_bump})"
        )

    @property
    def task_text(self):
        """Get the task_text property"""
        return self._task_text

    @task_text.setter
    def task_text(self, value: str):
        """Setter for task_text"""
        assert value, "task_text is required"
        self._task_text = value

    @property
    def target_bump(self):
        """Get the target_bump property"""
        return self._target_bump

    @target_bump.setter
    def target_bump(self, value: float):
        """Setter for target_bump"""
        msg = f"target_bump should be between {TARGET_BUMP_MIN} and {TARGET_BUMP_MAX}"
        assert TARGET_BUMP_MIN <= value <= TARGET_BUMP_MAX, msg
        self._target_bump = value

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
        spelled_text = ''.join(evidence)
        probs = equally_probable(self.symbol_set)
        symbol_probs = list(zip(self.symbol_set, probs))
        target = self._next_target(spelled_text)

        if target:
            sym = (target, probs[0] + self.target_bump)
            updated_symbol_probs = with_min_prob(symbol_probs, sym)
        else:
            updated_symbol_probs = symbol_probs

        return sorted(updated_symbol_probs,
                      key=lambda pair: self.symbol_set.index(pair[0]))

    def update(self) -> None:
        """Update the model state"""

    def load(self) -> None:
        """Restore model state from the provided checkpoint"""

    def _next_target(self, spelled_text: str) -> Optional[str]:
        """Computes the next target letter based on the currently spelled_text.
        """
        len_spelled = len(spelled_text)
        len_task = len(self.task_text)

        if len_spelled >= len_task and spelled_text[
                0:len_task] == self.task_text:
            return None

        if len_spelled < len_task and self.task_text[
                0:len_spelled] == spelled_text:
            # correctly spelled so far, get the next letter.
            return self.task_text[len(spelled_text)]
        return BACKSPACE_CHAR


# TODO: Cleanup; This method is copied from helpers/language_model.py but can't
# be imported from there due to circular dependencies. We should either create
# a separate module for lm utilities, or refactor this method to only extract
# the needed functionality.
def with_min_prob(symbol_probs: List[Tuple[str, float]],
                  sym_prob: Tuple[str, float]) -> List[Tuple[str, float]]:
    """Returns a new list of symbol-probability pairs where the provided
    symbol has a minimum probability given in the sym_prob.

    If the provided symbol is already in the list with a greater probability,
    the list of symbol_probs will be returned unmodified.

    If the new probability is added or modified, existing values are adjusted
    equally.

    Parameters:
    -----------
        symbol_probs - list of symbol, probability pairs
        sym_prob - (symbol, min_probability) defines the minimum probability
            for the given symbol in the returned list.

    Returns:
    -------
        list of (symbol, probability) pairs such that the sum of the
        probabilities is approx. 1.0.
    """
    new_sym, new_prob = sym_prob

    # Split out symbols and probabilities into separate lists, excluding the
    # symbol to be adjusted.
    symbols = []
    probs = []
    for sym, prob in symbol_probs:
        if sym != new_sym:
            symbols.append(sym)
            probs.append(prob)
        elif prob >= new_prob:
            # symbol prob in list is larger than minimum.
            return symbol_probs

    probabilities = np.array(probs)

    # Add new symbol and its probability
    all_probs = np.append(probabilities, new_prob / (1 - new_prob))
    all_symbols = symbols + [new_sym]

    normalized = all_probs / sum(all_probs)

    return list(zip(all_symbols, normalized))
