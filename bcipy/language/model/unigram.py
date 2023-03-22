from typing import List, Tuple
from bcipy.language.main import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.helpers.exceptions import InvalidLanguageModelException
import json
from bcipy.config import LM_PATH


class UnigramLanguageModel(LanguageModel):
    """Character language model based on trained unigram weights"""

    def __init__(self, response_type: ResponseType, symbol_set: List[str], lm_path: str = None):

        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.model = None
        self.lm_path = lm_path or f"{LM_PATH}/unigram.json"

        try:
            with open(self.lm_path) as json_file:
                self.unigram_lm = json.load(json_file)
        except BaseException:
            raise InvalidLanguageModelException("Unable to load Unigram model from file")

        self.unigram_lm[SPACE_CHAR] = self.unigram_lm.pop("SPACE_CHAR")
        self.unigram_lm[BACKSPACE_CHAR] = self.unigram_lm.pop("BACKSPACE_CHAR")

        if not set(self.unigram_lm.keys()) == set(self.symbol_set):
            raise InvalidLanguageModelException("Invalid unigram model symbol set!")

        self.unigram_lm = sorted(self.unigram_lm.items(), key=lambda item: item[1], reverse=True)

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
