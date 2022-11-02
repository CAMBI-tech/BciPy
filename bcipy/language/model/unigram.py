from typing import Dict, List, Tuple
from bcipy.helpers.task import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType

class UnigramLanguageModel(LanguageModel):
    """Character language model based on trained unigram weights"""

    def __init__(self, response_type: ResponseType, symbol_set: List[str], lm_path: str = None):
        
        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.model = None
        self.tokenizer = None
        self.vocab_size = 0
        self.idx_to_word = None
        self.lm_path = lm_path or None

        # Hard coding a unigram language model trained on ALS phrase dataset
        # for smoothing purpose
        # TODO: Load this from file
        self.unigram_lm = {'E': 0.0998, 'T': 0.096, 'O': 0.0946, 'I': 0.0835,
                           'A': 0.0774, 'N': 0.0554, SPACE_CHAR: 0.0523, 'H': 0.0504,
                           'S': 0.0435, 'L': 0.0408, 'R': 0.0387, 'U': 0.0379,
                           'D': 0.0358, 'Y': 0.0324, 'W': 0.0288, 'M': 0.0266,
                           'G': 0.0221, 'C': 0.018, 'K': 0.016, 'P': 0.0145,
                           'F': 0.0117, 'B': 0.0113, 'V': 0.0091, 'J': 0.0016,
                           'X': 0.0008, 'Z': 0.0005, 'Q': 0.0002, BACKSPACE_CHAR: 0.0}

                           
        assert set(self.unigram_lm.keys()) == set(self.symbol_set), "invalid unigram model symbol set!"

        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    # may not be needed
    
    # def __build_vocab(self) -> Dict[int, str]:
    #     """
    #     Build a vocabulary table mapping token index to word strings
    #     Response:
    #         a dict that maps token index to word strings
    #     """

    #     idx_to_word = {}
    #     for idx in range(self.vocab_size):
    #         word = self.tokenizer.decode([idx])
    #         idx_to_word[idx] = word

    #     return idx_to_word

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