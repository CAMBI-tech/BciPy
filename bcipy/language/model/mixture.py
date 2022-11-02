from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from scipy.special import softmax

from bcipy.helpers.task import BACKSPACE_CHAR, SPACE_CHAR, alphabet
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.language.model.gpt2 import GPT2LanguageModel
from bcipy.language.model.unigram import UnigramLanguageModel

class MixtureLanguageModel(LanguageModel):
    """Character language model that mixes GPT2 with a Unigram"""

    supported_lm_types = ["gpt2", "unigram"]

    def __init__(self, response_type: ResponseType, symbol_set: List[str], lm_types: List[str] = None, 
        lm_paths: List[str] = None, lm_weights: List[float] = None):
        
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type - SYMBOL only
            symbol_set - list of symbol strings
            lm_path - path to language model files
        """

        assert (lm_types == None and lm_paths == None) or (len(lm_types) == len(lm_paths)), "invalid model paths!"
        assert (lm_paths == None and lm_weights == None) or (len(lm_paths) == len(lm_weights)), "invalid model weights!"

        assert (lm_types == None or all(x in MixtureLanguageModel.supported_lm_types for x in lm_types)), "invalid model types!"

        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.models = list()
        self.lm_types = lm_types or ["gpt2", "unigram"]
        self.lm_paths = lm_paths or ["gpt2", "unigram"]
        self.lm_weights = lm_weights or [0.8, 0.2]

        # rescale coefficient
        self.rescale_coeff = 0.5

        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    @staticmethod
    def interpolate_language_models(lms: List[Dict[str, float]], coeffs: List[float]) -> List[Tuple]:
        """
        interpolate two language models
        Args:
            lm1 - the first language model (a dict with char as keys and prob as values)
            lm2 - the second language model (same type as lm1)
            coeff - rescale coefficient, lm1 will be scaled by coeff and lm2 will be
            scaled by (1-coeff)
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

        for model in self.models:
            pred = model.predict(evidence)
            pred_list.append(dict(pred))
        
        # interpolate language models to smooth the probability distribution returned
        # by GPT2 language model
        next_char_pred = MixtureLanguageModel.interpolate_language_models(pred_list, self.lm_weights)

        # exponentially rescale the language model
        next_char_pred = MixtureLanguageModel.rescale(dict(next_char_pred), self.rescale_coeff)

        return next_char_pred

    def update(self) -> None:
        """Update the model state"""
        ...

    def load(self) -> None:
        """
            Load the language models to be mixed
        """

        symbol_set = alphabet()
        response_type = ResponseType.SYMBOL

        for lm_type, path in zip(self.lm_types, self.lm_paths):
            if lm_type == "gpt2":
                model = GPT2LanguageModel(response_type, symbol_set, path)
            elif lm_type == "unigram":
                model = UnigramLanguageModel(response_type, symbol_set, path)
            
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