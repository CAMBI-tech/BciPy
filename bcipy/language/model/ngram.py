from typing import Optional

from bcipy.language.main import CharacterLanguageModel
from bcipy.language.model.adapter import LanguageModelAdapter
from aactextpredict.ngram import NGramLanguageModel
from bcipy.config import LM_PATH


class NGramLanguageModelAdapter(LanguageModelAdapter, CharacterLanguageModel):
    """Character n-gram language model using the KenLM library for querying"""

    def __init__(self,
                 lm_path: Optional[str] = None):
        """
        Initialize instance variables and load parameters
        Args:
            lm_path - location of local ngram model - loaded from parameters if None
        """

        self._load_parameters()

        ngram_params = self.parameters['ngram']
        ngram_model = ngram_params['model_file']['value']
        self.lm_path = lm_path or f"{LM_PATH}/{ngram_model}"


    def _load_model(self) -> None:
        """Load the model itself using stored parameters"""
        self.model = NGramLanguageModel(symbol_set=self.model_symbol_set, lm_path=self.lm_path)
