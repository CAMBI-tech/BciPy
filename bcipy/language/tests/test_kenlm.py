"""Tests for KENLM Language Model"""

import pytest
import unittest
import os
from operator import itemgetter

from bcipy.helpers.exceptions import UnsupportedResponseType, InvalidModelException
from bcipy.language.main import alphabet
from bcipy.language.model.kenlm import KenLMLanguageModel
from bcipy.language.main import BACKSPACE_CHAR, SPACE_CHAR, ResponseType


@pytest.mark.slow
class TestKenLMLanguageModel(unittest.TestCase):
    """Tests for language model"""
    @classmethod
    def setUpClass(cls):
        dirname = os.path.dirname(__file__) or '.'
        cls.lm_path = f"{dirname}/resources/lm_dec19_char_tiny_12gram.arpa"
        cls.lmodel = KenLMLanguageModel(response_type=ResponseType.SYMBOL,
                                        symbol_set=alphabet(), lm_path=cls.lm_path)

    def test_init(self):
        """Test default parameters"""
        self.assertEqual(self.lmodel.response_type, ResponseType.SYMBOL)
        self.assertEqual(self.lmodel.symbol_set, alphabet())
        self.assertTrue(
            ResponseType.SYMBOL in self.lmodel.supported_response_types())

    def test_name(self):
        """Test model name."""
        self.assertEqual("KENLM", KenLMLanguageModel.name())

    def test_unsupported_response_type(self):
        """Unsupported responses should raise an exception"""
        with self.assertRaises(UnsupportedResponseType):
            KenLMLanguageModel(response_type=ResponseType.WORD,
                               symbol_set=alphabet(), lm_path=self.lm_path)

    def test_invalid_model_path(self):
        """Test that the proper exception is thrown if given an invalid lm_path"""
        with self.assertRaises(InvalidModelException):
            KenLMLanguageModel(response_type=ResponseType.SYMBOL, symbol_set=alphabet(),
                               lm_path="phonymodel.txt")

    def test_non_mutable_evidence(self):
        """Test that the model does not change the evidence variable passed in.
           This could impact the mixture model if failed"""
        evidence = list("Test_test")
        evidence2 = list("Test_test")
        self.lmodel.predict(evidence)
        self.assertEqual(evidence, evidence2)

    def test_identical(self):
        """Ensure predictions are the same for subsequent queries with the same evidence."""
        query1 = self.lmodel.predict(list("evidenc"))
        query2 = self.lmodel.predict(list("evidenc"))
        for ((sym1, prob1), (sym2, prob2)) in zip(query1, query2):
            self.assertAlmostEqual(prob1, prob2, places=5)
            self.assertEqual(sym1, sym2)

    def test_upper_lower_case(self):
        """Ensure predictions are the same for upper or lower case evidence."""
        lc = self.lmodel.predict(list("EVIDENC"))
        uc = self.lmodel.predict(list("evidenc"))
        for ((l_sym, l_prob), (u_sym, u_prob)) in zip(lc, uc):
            self.assertAlmostEqual(l_prob, u_prob, places=5)
            self.assertEqual(l_sym, u_sym)

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
        self.assertAlmostEqual(sum(probs), 1, places=3)

    def test_predict_middle_of_word(self):
        """Test the predict method in the middle of a word."""
        symbol_probs = self.lmodel.predict(evidence=list("TH"))
        probs = [prob for sym, prob in symbol_probs]

        self.assertTrue(
            len(set(probs)) > 1,
            "All values should not be the same probability")
        for prob in probs:
            self.assertTrue(0 <= prob < 1)
        self.assertAlmostEqual(sum(probs), 1, places=3)

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
