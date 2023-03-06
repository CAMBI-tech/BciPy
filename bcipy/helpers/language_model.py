"""Helper functions for language model use."""
import math
import inspect
from typing import Dict, List, Tuple
import numpy as np
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.language.main import alphabet
# pylint: disable=unused-import
# flake8: noqa
from bcipy.language.model.uniform import UniformLanguageModel
# flake8: noqa
from bcipy.language.model.causal import CausalLanguageModel
from bcipy.language.model.mixture import MixtureLanguageModel
from bcipy.language.model.unigram import UnigramLanguageModel
from bcipy.helpers.exceptions import InvalidModelException


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

#   "lm_types": {
#     "value": "",
#     "section": "bci_config",
#     "readableName": "Mixture Language Model Types",
#     "helpTip": "Defines the types of models to be used by the mixture model. Default: None",
#     "recommended_values": [
#       "",
#       "causal, unigram",
#       "causal, kenlm"
#     ],
#     "type": "List[str]"
#   },
#   "lm_weights": {
#     "value": "",
#     "section": "bci_config",
#     "readableName": "Mixture Language Model Weights",
#     "helpTip": "Defines the weights of models to be used by the mixture model. Must sum to 1. Default: None",
#     "recommended_values": [
#       "",
#       "0.5, 0.5"
#     ],
#     "type": "List[float]"
#   },
#   "lm_params": {
#     "value": "",
#     "section": "bci_config",
#     "readableName": "Mixture Language Model Parameters",
#     "helpTip": "Defines the extra parameters of models to be used by the mixture model. Default: None",
#     "recommended_values": [
#       "",
#       "{'lang_model_name': 'gpt2'}, {}",
#       "{'lang_model_name': 'gpt2'}, {'lm_path': './bcipy/language/lms/lm_dec19_char_large_12gram.kenlm'}"
#     ],
#     "type": "List[Dict[str, str]]"
#   },

    language_models = language_models_by_name()
    model = language_models[parameters.get("lang_model_type", "UNIFORM")]

    # introspect the model arguments to determine what parameters to pass.
    args = inspect.signature(model).parameters.keys()

    if model == MixtureLanguageModel:
        mixture = parameters.get("lm_mixture", "gpt2_unigram")

        if mixture == "gpt2_opt":
            lm_types = ["causal", "causal"]
            lm_weights = [0.32, 0.68]
            lm_params = [{"lang_model_name": "gpt2"}, {"lang_model_name": "facebook/opt-125m"}]
        elif mixture == "gpt2_kenlm":
            kenlm_path = "./bcipy/language/lms/lm_dec19_char_large_12gram.kenlm"
            lm_types = ["causal", "kenlm"]
            lm_weights = [0.17, 0.83]
            lm_params = [{"lang_model_name": "gpt2"}, {"lm_path": kenlm_path}]
        elif mixture == "opt_kenlm":
            kenlm_path = "./bcipy/language/lms/lm_dec19_char_large_12gram.kenlm"
            lm_types = ["causal", "kenlm"]
            lm_weights = [0.30, 0.70]
            lm_params = [{"lang_model_name": "facebook/opt-125m"}, {"lm_path": kenlm_path}]
        else:
            raise InvalidModelException("The specified lm_mixture is not a defined configuration.")

        return model(response_type=ResponseType.SYMBOL,
                     symbol_set=alphabet(parameters),
                     lm_types=lm_types, lm_weights=lm_weights, lm_params=lm_params)

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
