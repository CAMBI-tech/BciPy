"""Uniform language model"""
from typing import Dict, List, Tuple, Union, Optional

import numpy as np

from bcipy.language.main import LanguageModelAdapter, ResponseType
from bcipy.core.symbols import SPACE_CHAR, BACKSPACE_CHAR, DEFAULT_SYMBOL_SET

from aactextpredict.uniform import UniformLanguageModel


class UniformLanguageModelAdapter(LanguageModelAdapter):
    """Language model in which probabilities for symbols are uniformly
    distributed.

    Parameters
    ----------
        response_type - SYMBOL only
        symbol_set - optional specify the symbol set, otherwise uses DEFAULT_SYMBOL_SET
    """

    def __init__(self,
                 response_type: Optional[ResponseType] = None,
                 symbol_set: Optional[List[str]] = DEFAULT_SYMBOL_SET):
        super().__init__(response_type=response_type)

        # LM doesn't care about backspace, needs literal space
        self.symbol_set = [' ' if ch is SPACE_CHAR else ch for ch in symbol_set]
        self.symbol_set.remove(BACKSPACE_CHAR)

        self.model = UniformLanguageModel(symbol_set=self.symbol_set)


    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]