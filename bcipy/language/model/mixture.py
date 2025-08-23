from typing import Dict, List, Optional

from textslinger.mixture import MixtureLanguageModel

from bcipy.config import LM_PATH
from bcipy.language.model.adapter import LanguageModelAdapter


class MixtureLanguageModelAdapter(LanguageModelAdapter):
    """Character language model that mixes any combination of other models."""

    supported_lm_types = MixtureLanguageModel.supported_lm_types

    def __init__(self,
                 lm_types: Optional[List[str]] = None,
                 lm_weights: Optional[List[float]] = None,
                 lm_params: Optional[List[Dict[str, str]]] = None):
        """
        Initialize instance variables and load parameters
        Args:
            lm_types - list of types of models to mix
            lm_weights - list of weights to use when mixing the models
            lm_params - list of dictionaries to pass as parameters for each model's instantiation
        """

        MixtureLanguageModel.validate_parameters(
            lm_types, lm_weights, lm_params)

        self._load_parameters()

        mixture_params = self.parameters['mixture']
        self.lm_types = lm_types or mixture_params['model_types']['value']
        self.lm_weights = lm_weights or mixture_params['model_weights']['value']
        self.lm_params = lm_params or mixture_params['model_params']['value']

        for type, params in zip(self.lm_types, self.lm_params):
            if type == "NGRAM":
                params["lm_path"] = f"{LM_PATH}/{params['lm_path']}"

        MixtureLanguageModel.validate_parameters(
            self.lm_types, self.lm_weights, self.lm_params)

    def _load_model(self) -> None:
        """Load the model itself using stored parameters"""
        self.model = MixtureLanguageModel(self.model_symbol_set, self.lm_types,
                                          self.lm_weights, self.lm_params)
