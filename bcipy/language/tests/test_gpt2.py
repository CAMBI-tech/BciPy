"""Tests for GPT2 Language Model"""

import pytest
import unittest
from operator import itemgetter

from bcipy.helpers.exceptions import UnsupportedResponseType
from bcipy.helpers.task import alphabet
from bcipy.language.model.gpt2 import GPT2LanguageModel, ResponseType
from bcipy.helpers.task import BACKSPACE_CHAR, SPACE_CHAR


@pytest.mark.slow
class TestGPT2LanguageModel(unittest.TestCase):
    """Tests for language model"""
    @classmethod
    def setUpClass(cls):
        cls.lmodel = GPT2LanguageModel(response_type=ResponseType.SYMBOL,
                                       symbol_set=alphabet())

    def test_init(self):
        """Test default parameters"""
        self.assertEqual(self.lmodel.response_type, ResponseType.SYMBOL)
        self.assertEqual(self.lmodel.symbol_set, alphabet())
        self.assertTrue(
            ResponseType.SYMBOL in self.lmodel.supported_response_types())

    def test_name(self):
        """Test model name."""
        self.assertEqual("GPT2", GPT2LanguageModel.name())

    def test_unsupported_response_type(self):
        """Unsupported responses should raise an exception"""
        with self.assertRaises(UnsupportedResponseType):
            GPT2LanguageModel(response_type=ResponseType.WORD,
                              symbol_set=alphabet())

    def test_predict_start_of_word(self):
        """Test the predict method with no prior evidence."""
        symbol_probs = self.lmodel.predict(evidence=[])
        probs = [prob for sym, prob in symbol_probs]

        self.assertTrue(
            len(set(probs)) > 1,
            "All values should not be the same probability")
        # Consider whether the following assertion should be True
        # backspace_prob = next(prob for sym, prob in probs if sym == BACKSPACE_CHAR)
        # self.assertEqual(0, backspace_prob)
        for prob in probs:
            self.assertTrue(0 <= prob < 1)
        self.assertAlmostEqual(sum(probs), 1)

    def test_predict_middle_of_word(self):
        """Test the predict method in the middle of a word."""
        symbol_probs = self.lmodel.predict(evidence=list("TH"))
        probs = [prob for sym, prob in symbol_probs]

        self.assertTrue(
            len(set(probs)) > 1,
            "All values should not be the same probability")
        for prob in probs:
            self.assertTrue(0 <= prob < 1)
        self.assertAlmostEqual(sum(probs), 1)

        most_likely_sym, _prob = sorted(symbol_probs,
                                        key=itemgetter(1),
                                        reverse=True)[0]
        self.assertEqual('E', most_likely_sym,
                         "Should predict 'E' as the next most likely symbol")

    def test_phrase(self):
        """Test that a phrase can be used for input"""
        symbol_probs = self.lmodel.predict(list("does_it_make_sen"))
        most_likely_sym, _prob = sorted(symbol_probs,
                                        key=itemgetter(1),
                                        reverse=True)[0]
        self.assertEqual('S', most_likely_sym)

    def test_multiple_spaces(self):
        """Test that the probability of space after a space is smaller than before the space"""
        symbol_probs_before = self.lmodel.predict(list("the"))
        symbol_probs_after = self.lmodel.predict(list("the_"))
        space_prob_before = (dict(symbol_probs_before))[SPACE_CHAR]
        space_prob_after = (dict(symbol_probs_after))[SPACE_CHAR]
        self.assertTrue(space_prob_before > space_prob_after)

    def test_nonzero_prob(self):
        """Test that all letters in the alphabet have nonzero probability except for backspace"""
        symbol_probs = self.lmodel.predict(list("does_it_make_sens"))
        prob_values = [item[1] for item in symbol_probs if item[0] != BACKSPACE_CHAR]
        for value in prob_values:
            self.assertTrue(value > 0)
