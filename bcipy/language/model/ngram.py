from typing import Optional, List
from bcipy.core.symbols import BACKSPACE_CHAR, SPACE_CHAR, DEFAULT_SYMBOL_SET
from bcipy.language.main import LanguageModelAdapter, ResponseType
from aactextpredict.ngram import NGramLanguageModel
from bcipy.config import LM_PATH


class NGramLanguageModelAdapter(LanguageModelAdapter):
    """Character n-gram language model using the KenLM library for querying"""

    def __init__(self,
                 response_type: ResponseType,
                 symbol_set: List[str] = DEFAULT_SYMBOL_SET,
                 lm_path: Optional[str] = None):

        super().__init__(response_type=response_type)
        ngram_params = self.parameters['ngram']
        ngram_model = ngram_params['model_file']['value']
        self.lm_path = lm_path or f"{LM_PATH}/{ngram_model}"

        self.symbol_set = symbol_set
        # LM doesn't care about backspace, needs literal space
        self.model_symbol_set = [' ' if ch is SPACE_CHAR else ch for ch in symbol_set]
        self.model_symbol_set.remove(BACKSPACE_CHAR)

        self.model = NGramLanguageModel(symbol_set=self.model_symbol_set, lm_path=self.lm_path)

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]
