"""Helper functions for language model use."""
import math
import inspect
from typing import Dict, List, Tuple
import numpy as np
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.helpers.system_utils import import_submodules
from bcipy.helpers.task import alphabet

import_submodules('bcipy.language')
# TODO: remove this import: https://www.pivotaltracker.com/story/show/178695472
import_submodules('bcipy.language_model')


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
    if not parameters['lang_model_enabled']:
        model = language_models['UNIFORM']
    else:
        model = language_models[parameters.get("lang_model_type", "GPT2")]

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


def sym_appended(symbol_probs: List[Tuple[str, float]],
                 sym_prob: Tuple[str, float]) -> List[Tuple[str, float]]:
    """Returns a new list of probabilities with the addition of a new symbol
    with the given probability for that symbol. Existing values are adjusted
    equally such that the sum of the probabilities in the resulting list sum
    to approx. 1.0. Only adds the symbol if it is not already in the list.

    Used to add the backspace symbol to the LM output.

    Parameters:
    -----------
        symbol_probs - list of symbol, probability pairs
        sym_prob - (symbol, probability) pair to append
    """
    if sym_prob[0] in dict(symbol_probs):
        return symbol_probs

    # Slit out symbols and probabilities into separate lists
    symbols = [prob[0] for prob in symbol_probs]
    probabilities = np.array([prob[1] for prob in symbol_probs])

    # Add new symbol and its probability
    all_probs = np.append(probabilities, sym_prob[1] / (1 - sym_prob[1]))
    all_symbols = symbols + [sym_prob[0]]

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
