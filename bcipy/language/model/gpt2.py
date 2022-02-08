from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from bcipy.helpers.task import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType


class GPT2LanguageModel(LanguageModel):
    """Character language model based on GPT2."""

    def __init__(self, response_type, symbol_set, lm_path=None):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type - SYMBOL only
            symbol_set - list of symbol strings
            lm_path - path to language model files
        """
        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.normalized = True
        self.model = None
        self.tokenizer = None
        self.is_start_of_word = True
        self.vocab_size = 0
        self.idx_to_word = None
        self.curr_word_predicted_prob = None
        self.lm_path = lm_path or "gpt2"
        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    def __get_char_predictions(self, word_prefix: str) -> List[tuple]:
        """
        Given a word prefix typed by user, predict the probability distribution
        of next character
        Args:
            word_prefix - string
        Response:
            A list of tuples - each tuple consists a pair of symbol and probability
        """

        prefix_len = len(word_prefix)
        char_to_prob = Counter()

        # iterate over the entire vocabulary
        for idx in range(self.vocab_size):
            # change word and word_prefix to upper case
            word = self.idx_to_word[idx].upper()
            word_prefix = word_prefix.upper()

            # if a word in the vocabulary matches the word prefix
            if word.startswith(word_prefix):
                # obtain the character following the prefix in this word
                if len(word) == prefix_len:
                    next_char = SPACE_CHAR
                else:
                    next_char = word[prefix_len]
                # make sure the next character is in the symbol set
                if next_char in self.symbol_set and next_char != BACKSPACE_CHAR:
                    char_to_prob[next_char] += self.curr_word_predicted_prob[idx]

        # normalize char_to_prob
        sum_char_prob = sum(char_to_prob.values())
        for char in char_to_prob:
            char_to_prob[char] /= sum_char_prob

        # build a list of tuples (char, prob)
        char_prob_tuples = list(
            sorted(char_to_prob.items(),
                   key=lambda item: item[1],
                   reverse=True))

        return char_prob_tuples

    def __build_vocab(self) -> Dict[int, str]:
        """
        Build a vocabulary table mapping token index to word strings
        Response:
            a dict that maps token index to word strings
        """

        idx_to_word = {}
        for idx in range(self.vocab_size):
            word = self.tokenizer.decode([idx]).strip()
            idx_to_word[idx] = word

        return idx_to_word

    def __model_infer(self, text: str) -> List[float]:
        """
        Use the transformer language model to infer the distribution of next word
        Args:
            text - a string of text representing the text input so far
        Response:
            An numpy array representing next word's probability distribution over
            the entire vocabulary
        """

        # empty text indicates the beginning of a phrase, attach the BOS symbol
        if text == "":
            text = "<|endoftext|>"

        # tokenize the text
        indexed_tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens])

        # predict the next word
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]

            predicted_logit = predictions[0, -1, :]
            predicted_prob = F.softmax(predicted_logit, dim=0).numpy()

        return predicted_prob

    def predict(self, evidence: List[str]) -> List[Tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of
        the next symbol
        Args:
            evidence - a list of characters (typed by the user)
        Response:
            A list of symbols with probability
        """
        assert self.model is not None, "language model does not exist!"

        evidence_str = "".join(evidence)

        # mark the start-of-word indicator
        if evidence_str == "" or evidence[-1] == SPACE_CHAR:
            self.is_start_of_word = True
        else:
            self.is_start_of_word = False

        # predict the next character
        evidence_str = evidence_str.replace(SPACE_CHAR, ' ')

        # if we are at the start of a word, let the language model predict the next word
        # then we get the probability distribution of the first character of the next word
        if self.is_start_of_word is True:
            self.curr_word_predicted_prob = self.__model_infer(evidence_str)
            next_char_pred = self.__get_char_predictions("")

        # if we are at the middle of a word, just get the marginal distribution of
        # the next character given current word prefix
        else:
            # first we need the language model predict the probability distribution of the
            # current word
            self.curr_word_predicted_prob = self.__model_infer(" ".join(evidence_str.split()[:-1]))
            cur_word_prefix = evidence_str.split()[-1]
            next_char_pred = self.__get_char_predictions(cur_word_prefix)

        return next_char_pred

    def update(self) -> None:
        """Update the model state"""
        ...

    def load(self) -> None:
        """
            Load the language model and tokenizer, initialize class variables
        Args:
            path: language model file path, can be just "gpt2"
        """
        self.model = GPT2LMHeadModel.from_pretrained(self.lm_path)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.lm_path)
        self.vocab_size = self.tokenizer.vocab_size
        self.idx_to_word = self.__build_vocab()

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
