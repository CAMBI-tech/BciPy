"""Helper functions for language model use."""
import math
import inspect
from typing import Dict, List, Tuple
import numpy as np
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.language.main import alphabet
# pylint: disable=unused-import
# flake8: noqa
from bcipy.language.uniform import UniformLanguageModel
# flake8: noqa
from bcipy.language.model.gpt2 import GPT2LanguageModel
from bcipy.language.model.causal import CausalLanguageModel
from bcipy.language.model.mixture import MixtureLanguageModel
from bcipy.language.model.unigram import UnigramLanguageModel


def language_models_by_name() -> Dict[str, LanguageModel]:
    """Returns available language models indexed by name."""
    return {lm.name(): lm for lm in LanguageModel.__subclasses__()}


def init_language_model(parameters: dict) -> LanguageModel:
    """
    Init Language Model configured in the parameters.

    Parameters
    ----------
        parameters : dict
            configuration details and path locations

    Returns
    -------
        instance of a LanguageModel
    """
    language_models = language_models_by_name()
    model = language_models[parameters.get("lang_model_type", "UNIFORM")]

    # introspect the model arguments to determine what parameters to pass.
    args = inspect.signature(model).parameters.keys()

    # select the relevant parameters into a dict.
    params = {key: parameters[key] for key in args & parameters.keys()}
    return model(response_type=ResponseType.SYMBOL,
                 symbol_set=alphabet(parameters),
                 **params)


def norm_domain(priors: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Convert a list of (symbol, likelihood) values from negative log
    likelihood to the probability domain (between 0 and 1)

    Parameters:
        priors - list of (symbol, likelihood) values.
            assumes that the units are in the negative log likelihood where
            the lowest value is the most likely.
    Returns:
        list of values in the probability domain (between 0 and 1),
            where the highest value is the most likely.
    """
    return [(sym, math.exp(-prob)) for sym, prob in priors]


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


def histogram(letter_prior: List[Tuple[str, float]]) -> str:
    """Given a list of letter, prob tuples, generate a histogram that can be
    output to the console.

    Parameters:
    -----------
        letter_prior - list of letter, probability pairs
    Returns:
    --------
        printable string which contains a histogram with the letter and probability as the label.
    """
    margin = "\t"
    star = '*'
    lines = []
    for letter, prob in sorted(letter_prior):
        units = int(round(prob * 100))
        lines.append(letter + ' (' + "%03.2f" % (prob) + ") :" + margin +
                     (units * star))
    return '\n'.join(lines)
