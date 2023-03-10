from collections import Counter
from typing import Dict, List, Tuple
from math import isclose

from bcipy.language.main import LanguageModel, ResponseType

from bcipy.helpers.exceptions import InvalidModelException

# pylint: disable=unused-import
# flake8: noqa
"""All supported models must be imported"""
from bcipy.language.model.causal import CausalLanguageModel
from bcipy.language.model.kenlm import KenLMLanguageModel
from bcipy.language.model.unigram import UnigramLanguageModel


class MixtureLanguageModel(LanguageModel):
    """
        Character language model that mixes any combination of other models
    """

    supported_lm_types = ["CAUSAL", "UNIGRAM", "KENLM"]

    @staticmethod
    def language_models_by_name() -> Dict[str, LanguageModel]:
        """Returns available language models indexed by name."""
        return {lm.name(): lm for lm in LanguageModel.__subclasses__()}

    def __init__(self,
                 response_type: ResponseType,
                 symbol_set: List[str],
                 lm_types: List[str] = None,
                 lm_weights: List[float] = None,
                 lm_params: List[Dict[str, str]] = None):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type - SYMBOL only
            symbol_set - list of symbol strings
            lm_types - list of types of models to mix
            lm_weights - list of weights to use when mixing the models
            lm_params - list of dictionaries to pass as parameters for each model's instantiation
        """

        if lm_params is not None:
            if (lm_types is None) or (len(lm_types) != len(lm_params)):
                raise InvalidModelException("Length of parameters does not match length of types")

        if lm_weights is not None:
            if (lm_types is None) or (len(lm_types) != len(lm_weights)):
                raise InvalidModelException("Length of weights does not match length of types")
            if not isclose(sum(lm_weights), 1.0, abs_tol=1e-05):
                raise InvalidModelException("Weights do not sum to 1")

        if lm_types is not None:
            if lm_weights is None:
                raise InvalidModelException("Model weights not provided")
            if lm_params is None:
                raise InvalidModelException("Model parameters not provided")
            if not all(x in MixtureLanguageModel.supported_lm_types for x in lm_types):
                raise InvalidModelException(f"Supported model types: {MixtureLanguageModel.supported_lm_types}")

        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.models = list()
        self.response_type = response_type
        self.symbol_set = symbol_set
        self.lm_types = lm_types or self.parameters.get("mixture_types")
        self.lm_weights = lm_weights or self.parameters.get("mixture_weights")
        self.lm_params = lm_params or self.parameters.get("mixture_params")

        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    @staticmethod
    def interpolate_language_models(lms: List[Dict[str, float]], coeffs: List[float]) -> List[Tuple]:
        """
        interpolate two or more language models
        Args:
            lms - output from the language models (a list of dicts with char as keys and prob as values)
            coeffs - list of rescale coefficients, lms[0] will be scaled by coeffs[0] and so on
        Response:
            a list of (char, prob) tuples representing an interpolated language model
        """
        combined_lm = Counter()

        for i, lm in enumerate(lms):
            for char in lm:
                combined_lm[char] += lm[char] * coeffs[i]

        return list(sorted(combined_lm.items(), key=lambda item: item[1], reverse=True))

    def predict(self, evidence: List[str]) -> List[Tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of
        the next symbol
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbols with probability
        """

        pred_list = list()

        # Generate predictions from each component language model
        for model in self.models:
            pred = model.predict(evidence)
            pred_list.append(dict(pred))

        # Mix the component models
        next_char_pred = MixtureLanguageModel.interpolate_language_models(pred_list, self.lm_weights)

        return next_char_pred

    def update(self) -> None:
        """Update the model state"""
        ...

    def load(self) -> None:
        """
            Load the language models to be mixed
        """

        language_models = MixtureLanguageModel.language_models_by_name()
        for lm_type, params in zip(self.lm_types, self.lm_params):
            model = language_models[lm_type]
            lm = None
            try:
                lm = model(self.response_type, self.symbol_set, **params)
            except InvalidModelException as e:
                raise InvalidModelException(f"Error in creation of model type {lm_type}: {e.message}")

            self.models.append(lm)

    def state_update(self, evidence: List[str]) -> List[Tuple]:
        """
            Wrapper method that takes in evidence text, and output probability distribution
            of next character
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbol with probability
        """
        next_char_pred = self.predict(evidence)

        return next_char_pred
