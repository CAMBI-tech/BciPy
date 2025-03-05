from typing import Optional, Dict, List

from bcipy.language.main import LanguageModelAdapter, ResponseType
from bcipy.core.symbols import SPACE_CHAR, BACKSPACE_CHAR, DEFAULT_SYMBOL_SET
from bcipy.config import LM_PATH

from aactextpredict.mixture import MixtureLanguageModel


class MixtureLanguageModelAdapter(LanguageModelAdapter):
    """
        Character language model that mixes any combination of other models
    """

    supported_lm_types = MixtureLanguageModel.supported_lm_types

    def __init__(self,
                 response_type: ResponseType,
                 symbol_set: List[str] = DEFAULT_SYMBOL_SET,
                 lm_types: Optional[List[str]] = None,
                 lm_weights: Optional[List[float]] = None,
                 lm_params: Optional[List[Dict[str, str]]] = None):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type - SYMBOL only
            symbol_set - list of symbol strings
            lm_types - list of types of models to mix
            lm_weights - list of weights to use when mixing the models
            lm_params - list of dictionaries to pass as parameters for each model's instantiation
        """

        MixtureLanguageModel.validate_parameters(lm_types, lm_weights, lm_params)

        super().__init__(response_type=response_type)

        # LM doesn't care about backspace, needs literal space
        self.symbol_set = [' ' if ch is SPACE_CHAR else ch for ch in symbol_set]
        self.symbol_set.remove(BACKSPACE_CHAR)

        mixture_params = self.parameters['mixture']
        self.lm_types = lm_types or mixture_params['model_types']['value']
        self.lm_weights = lm_weights or mixture_params['model_weights']['value']
        self.lm_params = lm_params or mixture_params['model_params']['value']

        for type, params in zip(self.lm_types, self.lm_params):
            if type == "NGRAM":
                params["lm_path"] = f"{LM_PATH}/{params['lm_path']}"

        MixtureLanguageModel.validate_parameters(self.lm_types, self.lm_weights, self.lm_params)

        self.model = MixtureLanguageModel(self.symbol_set, self.lm_types, self.lm_weights, self.lm_params)

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]
