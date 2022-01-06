from typing import List, Tuple, Dict
from pathlib import Path
from bcipy.helpers.task import alphabet
from bcipy.language import LanguageModel
from bcipy.language.base import ResponseType

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import Counter


class TransformerLanguageModel(LanguageModel):

    def __init__(self, response_type, symbol_set):
        self.response_type = response_type
        self.symbol_set = symbol_set

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
                    next_char = "_"
                else:
                    next_char = word[prefix_len]
                # make sure the next character is in the symbol set
                if next_char in self.symbol_set and next_char != "<":
                    char_to_prob[next_char] += self.curr_predicted_prob[idx]

        # normalize char_to_prob
        sum_char_prob = sum(char_to_prob.values())
        for char in char_to_prob:
            char_to_prob[char] /= sum_char_prob

        # build a list of tuples (char, prob)
        char_prob_tuples = [ (k,v) for k, v in sorted(char_to_prob.items(), key = lambda item: item[1], reverse=True)]
        
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

        # tokenize the text
        indexed_tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens])

        # predict the next word
        with torch.no_grad():
            model = self.model
            outputs = model(tokens_tensor)
            predictions = outputs[0]

            predicted_logit = predictions[0, -1, :]
            predicted_prob = F.softmax(predicted_logit, dim = 0).numpy()

        return predicted_prob

    def predict(self, evidence: str) -> List[tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of 
        the next symbol
        Args:
            evidence - a string of evidence text
        Response:
            A list of symbols with probability
        """
        assert self.model is not None, "language model does not exist!"

        # if we are at the start of a word, let the language model predict the next word
        # then we get the probability distribution of the first character of the next word
        if self.is_start_of_word is True:
            self.curr_predicted_prob = self.__model_infer(evidence)

            next_char_pred = self.__get_char_predictions("")

        # if we are at the middle of a word, just get the marginal distribution of
        # the next character given current word prefix
        else:
            # make sure we have the word level distribution
            if self.curr_predicted_prob is None:
                self.curr_predicted_prob = self.__model_infer(" ".join(evidence.split()[:-1]))

            next_char_pred = self.update(evidence)

        return next_char_pred

    def update(self, evidence: str) -> List[tuple]:
        """
        Obtain the marginal distribution of next character given current word prefix
        Args:
            evidence - a string of evidence text
        Response:
            A list of symbol with probability
        """
        cur_word_prefix = evidence.split()[-1]
        next_char_pred = self.__get_char_predictions(cur_word_prefix)

        return next_char_pred


    def load(self, path: Path) -> None:
        """
            Load the language model and tokenizer, initialize class variables
        Args:
            path: language model file path, can be just "gpt2"
        """
        self.model = GPT2LMHeadModel.from_pretrained(path)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        self.is_start_of_word = True
        self.vocab_size = self.tokenizer.vocab_size
        self.idx_to_word = self.__build_vocab()
        self.curr_predicted_prob = None

    def state_update(self, evidence: str) -> List[Tuple]:
        """
            Wrapper method that takes in evidence text, and output probability distribution
            of next character
        Args:
            evidence - a string of evidence text
        Response:
            A list of symbol with probability
        """

        # empty evidence indicates the beginning of a phrase, attach the BOS symbol
        if evidence == "":
            evidence = "<|endoftext|>"

        # mark the start-of-word indicator
        if evidence == "<|endoftext|>" or evidence[-1] == "_":
            self.is_start_of_word = True
        else:
            self.is_start_of_word = False

        # predict the next character
        evidence = evidence.replace('_', ' ')

        next_char_pred = self.predict(evidence)

        return next_char_pred

    def reset(self):
        """
        Erase the current prediction
        """
        self.curr_predicted_prob = None


# test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--data_folder', default=None)
    # args = parser.parse_args()
    # data_folder = args.data_folder
    symbol_set = alphabet()
    response_type = ResponseType.SYMBOL
    lm = TransformerLanguageModel(response_type, symbol_set)
    lm.load('gpt2')
    next_char_pred = lm.state_update("")
    print(next_char_pred)
    next_char_pred = lm.state_update("peanut_butter_and_")
    print(next_char_pred)
    next_char_pred = lm.state_update("j")
    print(next_char_pred)
    next_char_pred = lm.state_update("e")
    print(next_char_pred)
    next_char_pred = lm.state_update("l")
    print(next_char_pred)
    next_char_pred = lm.state_update("l")
    print(next_char_pred)
    next_char_pred = lm.state_update("y")
    print(next_char_pred)
    next_char_pred = lm.state_update("_")
    print(next_char_pred)
    lm.reset()
    next_char_pred = lm.state_update("")
    print(next_char_pred)
    next_char_pred = lm.state_update("peanut_but")
    print(next_char_pred)
    next_char_pred = lm.state_update("peanut_butte")
    print(next_char_pred)
