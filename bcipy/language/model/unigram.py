from typing import Dict, List, Tuple
from bcipy.language.main import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType
import json, os

class UnigramLanguageModel(LanguageModel):
    """Character language model based on trained unigram weights"""

    def __init__(self, response_type: ResponseType, symbol_set: List[str], lm_path: str = None):
        
        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.model = None
        dirname = os.path.dirname(__file__) or '.'
        self.lm_path = lm_path or f"{dirname}/../lms/unigram.json"

        with open(self.lm_path) as json_file:
            self.unigram_lm = json.load(json_file)

        self.unigram_lm[SPACE_CHAR] = self.unigram_lm.pop("SPACE_CHAR")
        self.unigram_lm[BACKSPACE_CHAR] = self.unigram_lm.pop("BACKSPACE_CHAR")

        self.unigram_lm = dict(sorted(self.unigram_lm.items(), key=lambda item: item[1], reverse=True))

        assert set(self.unigram_lm.keys()) == set(self.symbol_set), "invalid unigram model symbol set!"

        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    def predict(self, evidence: List[str]) -> List[Tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of
        the next symbol
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbols with probability
        """

        return self.unigram_lm

    def update(self) -> None:
        """Update the model state"""
        ...

    def load(self) -> None:
        """
            Load the language model and tokenizer, initialize class variables
        Args:
            path: language model file path, can be just "unigram"
        """

    def state_update(self, evidence: List[str]) -> List[Tuple]:
        """
            Wrapper method that takes in evidence text, and output probability distribution
            of next character - evidence does not matter for unigram model
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbol with probability
        """
        return self.unigram_lm