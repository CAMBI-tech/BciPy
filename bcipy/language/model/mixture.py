from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from scipy.special import softmax
from math import isclose

from bcipy.language.main import BACKSPACE_CHAR, SPACE_CHAR, alphabet
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.language.model.unigram import UnigramLanguageModel
from bcipy.language.model.kenlm import KenLMLanguageModel
from bcipy.language.model.causal import CausalLanguageModel

from bcipy.helpers.exceptions import InvalidModelException

class MixtureLanguageModel(LanguageModel):
    """
        Character language model that mixes any combination of other models
        By default, 80% Causal GPT-2 with 20% Unigram
    """

    supported_lm_types = ["causal", "unigram", "kenlm"]

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

        if lm_params != None:
            if (lm_types == None) or (len(lm_types) != len(lm_params)):
                raise InvalidModelException("Length of parameters does not match length of types")
        
        if lm_weights != None:
            if (lm_types == None) or (len(lm_types) != len(lm_weights)):
                raise InvalidModelException("Length of weights does not match length of types")
            if not isclose(sum(lm_weights), 1.0, abs_tol=1e-05):
                raise InvalidModelException("Weights do not sum to 1")

        if lm_types != None:
            if lm_weights == None:
                raise InvalidModelException("Model weights not provided")
            if lm_params == None:
                raise InvalidModelException("Model parameters not provided")
            if not all(x in MixtureLanguageModel.supported_lm_types for x in lm_types):
                raise InvalidModelException(f"Supported model types: {MixtureLanguageModel.supported_lm_types}")

        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.models = list()
        self.response_type = response_type
        self.symbol_set = symbol_set
        self.lm_types = lm_types or ["causal", "unigram"]
        self.lm_weights = lm_weights or [0.8, 0.2]
        self.lm_params = lm_params or [{"lang_model_name": "gpt2"}, {}]

        # rescale coefficient
        self.rescale_coeff = 0.5

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

    @staticmethod
    def rescale(lm: Dict[str, float], coeff: float):
        """
        rescale a languge model with exponential coefficient
        Args:
            lm - the language model (a dict with char as keys and prob as values)
            coeff - rescale coefficient
        Response:
            a list of (char, prob) tuples representing a rescaled language model
        """
        rescaled_lm = Counter()

        # scale
        for char in lm:
            rescaled_lm[char] = lm[char] ** coeff

        # normalize
        sum_char_prob = sum(rescaled_lm.values())
        for char in rescaled_lm:
            rescaled_lm[char] /= sum_char_prob

        return list(sorted(rescaled_lm.items(), key=lambda item: item[1], reverse=True))

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

        # Exponentially rescale the mixed predictions
        # next_char_pred = MixtureLanguageModel.rescale(dict(next_char_pred), self.rescale_coeff)

        return next_char_pred

    def update(self) -> None:
        """Update the model state"""
        ...

    def load(self) -> None:
        """
            Load the language models to be mixed
        """

        for lm_type, params in zip(self.lm_types, self.lm_params):
            try:
                if lm_type == "causal":
                    model = CausalLanguageModel(self.response_type, self.symbol_set, **params)
                elif lm_type == "unigram":
                    model = UnigramLanguageModel(self.response_type, self.symbol_set, **params)
                elif lm_type == "kenlm":
                    model = KenLMLanguageModel(self.response_type, self.symbol_set, **params)
            except InvalidModelException as e:
                raise InvalidModelException(f"Error in creation of model type {lm_type}: {e.message}")
            
            self.models.append(model)

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