from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from scipy.special import softmax

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from bcipy.helpers.task import BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.main import LanguageModel, ResponseType
from bcipy.language.uniform import equally_probable


class GPT2LanguageModel(LanguageModel):
    """Character language model based on GPT2."""

    def __init__(self, response_type: ResponseType, symbol_set: List[str], lm_path: str = None):
        """
        Initialize instance variables and load the language model with given path
        Args:
            response_type - SYMBOL only
            symbol_set - list of symbol strings
            lm_path - path to language model files
        """
        super().__init__(response_type=response_type, symbol_set=symbol_set)
        self.model = None
        self.tokenizer = None
        self.is_start_of_word = True
        self.vocab_size = 0
        self.idx_to_word = None
        self.curr_word_predicted_prob = None
        self.lm_path = lm_path or "gpt2"

        # Hard coding a unigram language model trained on ALS phrase dataset
        # for smoothing purpose
        self.unigram_lm = {'E': 0.0998, 'T': 0.096, 'O': 0.0946, 'I': 0.0835,
                           'A': 0.0774, 'N': 0.0554, SPACE_CHAR: 0.0523, 'H': 0.0504,
                           'S': 0.0435, 'L': 0.0408, 'R': 0.0387, 'U': 0.0379,
                           'D': 0.0358, 'Y': 0.0324, 'W': 0.0288, 'M': 0.0266,
                           'G': 0.0221, 'C': 0.018, 'K': 0.016, 'P': 0.0145,
                           'F': 0.0117, 'B': 0.0113, 'V': 0.0091, 'J': 0.0016,
                           'X': 0.0008, 'Z': 0.0005, 'Q': 0.0002, BACKSPACE_CHAR: 0.0}

        # A uniform language model
        self.uniform_lm = dict(zip(self.symbol_set, equally_probable(self.symbol_set, {BACKSPACE_CHAR: 0.0})))

        self.load()

    def supported_response_types(self) -> List[ResponseType]:
        return [ResponseType.SYMBOL]

    def __build_vocab(self) -> Dict[int, str]:
        """
        Build a vocabulary table mapping token index to word strings
        Response:
            a dict that maps token index to word strings
        """

        idx_to_word = {}
        for idx in range(self.vocab_size):
            word = self.tokenizer.decode([idx])
            idx_to_word[idx] = word

        return idx_to_word

    def __model_infer(self, text: str, is_initial: bool) -> List[float]:
        """
        Use the transformer language model to infer the distribution of next word
        Args:
            text - a string of text representing the text input so far
            is_initial - a boolean indicating whether we are infering the first wordpiece in
            beam search
        Response:
            An numpy array representing next wordpiece's log likelihood distribution over
            the entire vocabulary
        """

        # tokenize the text
        if is_initial is True:
            # if infering the first wordpiece in beam search, treat the last input token as partially
            # complete and exclude it from the input to the language model
            indexed_tokens = self.tokenizer.encode(text)[:-1]
            if len(indexed_tokens) == 0:
                # attach the BOS token if there is no wordpiece prior to the last input token
                indexed_tokens = self.tokenizer.encode("<|endoftext|>")
        else:
            indexed_tokens = self.tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens])

        # predict the next wordpiece
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]

            predicted_logit = predictions[0, -1, :].numpy()

        return predicted_logit

    def __prefix_match(self, sorted_idx: List[float], evidence_str: str) -> Tuple[List[float], str]:
        """
        Filter the sorted_idx, keep the indices whose tokens have
        prefix matches with the last wordpiece of evidence_str
        Args:
            sorted_idx - indices of vocabulary returned by GPT2 language model
            sorted by log likelihood
            evidence_str - evidence input by user
        Response:
            matched_sorted_idx - a list of indices whose tokens have prefix
            matches with the last wordpiece of evidence_str
            new_evidence_str - evidence_str without the last wordpiece
        """
        matched_sorted_idx = []
        if evidence_str == "":
            evidence_str_last_token_text = ""
            new_evidence_str = ""
        else:
            evidence_str_tokens = self.tokenizer.encode(evidence_str)
            evidence_str_last_token_text = self.idx_to_word[evidence_str_tokens[-1]]
            new_evidence_str = self.tokenizer.decode(evidence_str_tokens[:-1])

        for idx in sorted_idx:
            idx_token_text = self.idx_to_word[idx]
            if idx_token_text.startswith(evidence_str_last_token_text):
                matched_sorted_idx.append(idx)

        return matched_sorted_idx, new_evidence_str

    def __populate_beam_candidates(self, evidence_str: str,
                                   beam: List[Tuple], all_beam_candidates: List[Tuple]) -> None:
        """
        populate all_beam_candidates with candidates from beam
        Args:
            evidence_str - input text from user
            beam - list of candidates with text and log likelihood
            all_beam_candidates - the list we are going to populate with candidates from beam
        """
        for candidate in beam:
            candidate_text = evidence_str + self.tokenizer.decode(candidate[0])
            candidate_log_prob = candidate[1]
            all_beam_candidates.append((candidate_text, candidate_log_prob))

    def __get_char_predictions_beam_search(self, evidence_str: str, all_beam_candidates: List[Tuple]) -> List[Tuple]:
        """
        marginalize the word probabilities to generate character level probability distribution
        Args:
            evidence_str - input text from user
            all_beam_candidates - list of candidates with text and log likelihood
        Response:
            A list of tuples - each tuple consists a pair of symbol and probability
        """
        char_to_prob = Counter()
        candidates_log_likelihood = np.array([beam_candidate[1] for beam_candidate in all_beam_candidates])
        candidates_prob = softmax(candidates_log_likelihood)

        for idx, beam_candidate in enumerate(all_beam_candidates):
            candidate_text = beam_candidate[0]
            # obtain the next character following evidence_str in the candidate
            if len(evidence_str) == len(candidate_text):
                next_char = SPACE_CHAR
            else:
                next_char = candidate_text[len(evidence_str)].upper()

            # make sure the next character is in the symbol set
            if next_char in self.symbol_set and next_char != BACKSPACE_CHAR:
                char_to_prob[next_char] += candidates_prob[idx]

        # normalize char_to_prob
        sum_char_prob = sum(char_to_prob.values())
        for char in char_to_prob:
            char_to_prob[char] /= sum_char_prob

        # assign probability of 0.0 for symbols not returned by the language model
        for char in self.symbol_set:
            if char not in char_to_prob:
                char_to_prob[char] = 0.0

        # build a list of tuples (char, prob)
        char_prob_tuples = list(
            sorted(char_to_prob.items(),
                   key=lambda item: item[1],
                   reverse=True))

        return char_prob_tuples

    def __interpolate_language_models(self, lm1: Dict[str, float], lm2: Dict[str, float], coeff: float) -> List[Tuple]:
        """
        interpolate two language models
        Args:
            lm1 - the first language model (a dict with char as keys and prob as values)
            lm2 - the second language model (same type as lm1)
            coeff - rescale coefficient, lm1 will be scaled by coeff and lm2 will be
            scaled by (1-coeff)
        Response:
            a list of (char, prob) tuples representing an interpolated language model
        """
        combined_lm = Counter()

        for char in lm1:
            combined_lm[char] = lm1[char] * coeff + lm2[char] * (1 - coeff)

        for char in lm2:
            if char not in lm1:
                combined_lm[char] = lm2[char] * (1 - coeff)

        return list(sorted(combined_lm.items(), key=lambda item: item[1], reverse=True))

    def __rescale(self, lm: Dict[str, float], coeff: float):
        """
        rescale a languge model with exponential coefficient
        Args:
            lm - the language model (a dict with char as keys and prob as values)
            coeff - rescale coefficient
        Response:
            a list of (char, prob) tuples representing a rescaled language model
        """

        rescaled_lm = Counter()

        # scale
        for char in lm:
            rescaled_lm[char] = lm[char] ** coeff

        # normalize
        sum_char_prob = sum(rescaled_lm.values())
        for char in rescaled_lm:
            rescaled_lm[char] /= sum_char_prob

        return list(sorted(rescaled_lm.items(), key=lambda item: item[1], reverse=True))

    def predict(self, evidence: List[str], beam_width: int = 20, search_depth: int = 2) -> List[Tuple]:
        """
        Given an evidence of typed string, predict the probability distribution of
        the next symbol
        Args:
            evidence - a list of characters (typed by the user)
            beam_width - size of the beam used for beam search (hyperparameter)
            search_depth - max number of wordpieces to predict (hyperparameter)
        Response:
            A list of symbols with probability
        """

        assert self.model is not None, "language model does not exist!"

        evidence_str = "".join(evidence)

        evidence_str = evidence_str.replace(SPACE_CHAR, ' ')
        evidence_str = evidence_str.lower()

        # infer the first wordpiece
        predicted_logit = self.__model_infer(evidence_str, True)
        sorted_idx = predicted_logit.argsort()[::-1]

        # get the indices of wordpieces whose have prefix match with the last wordpiece of the
        # evidence_str
        matched_sorted_idx, new_evidence_str = self.__prefix_match(sorted_idx, evidence_str)

        all_beam_candidates = []

        # construct the initial beam
        beam = []
        for i in range(min(beam_width, len(matched_sorted_idx))):
            candidate = ([matched_sorted_idx[i]], predicted_logit[matched_sorted_idx[i]])
            beam.append(candidate)

        # populate all_beam_candidates with tuples of (text, log_likelihood)
        self.__populate_beam_candidates(new_evidence_str, beam, all_beam_candidates)

        # further beam search with more depth
        for depth in range(search_depth - 1):
            new_candidates = []
            for candidate in beam:
                candidate_tokens_idx = candidate[0]
                candidate_tokens_log_prob = candidate[1]
                candidate_text = new_evidence_str + self.tokenizer.decode(candidate_tokens_idx)
                predicted_logit = self.__model_infer(candidate_text, False)
                sorted_idx = predicted_logit.argsort()[::-1]
                for i in range(beam_width):
                    # make sure that wordpieces contain only letters
                    if self.idx_to_word[sorted_idx[i]].isalpha():
                        new_candidate = (candidate_tokens_idx + [sorted_idx[i]],
                                         candidate_tokens_log_prob + predicted_logit[sorted_idx[i]])
                        new_candidates.append(new_candidate)

            # sort the new candidates based on likelihood and populate the beam
            ordered_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
            beam = ordered_candidates[:beam_width]

            self.__populate_beam_candidates(new_evidence_str, beam, all_beam_candidates)

        all_beam_candidates = sorted(all_beam_candidates, key=lambda x: x[1], reverse=True)

        # get marginalized characger-level probabilities from beam search results
        next_char_pred = self.__get_char_predictions_beam_search(evidence_str, all_beam_candidates)

        # interpolate with unigram language model to smooth the probability distribution returned
        # by GPT2 language model
        next_char_pred = self.__interpolate_language_models(dict(next_char_pred), self.unigram_lm, 0.8)

        # exponentially rescale the language model
        next_char_pred = self.__rescale(dict(next_char_pred), 0.5)

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
