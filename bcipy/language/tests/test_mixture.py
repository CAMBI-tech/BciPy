"""Tests for MIXTURE Language Model"""

import pytest
import unittest
import os
from operator import itemgetter

from bcipy.exceptions import InvalidSymbolSetException
from textslinger.exceptions import InvalidLanguageModelException
from bcipy.core.symbols import DEFAULT_SYMBOL_SET, BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.model.mixture import MixtureLanguageModelAdapter
from bcipy.language.main import CharacterLanguageModel


@pytest.mark.slow
class TestMixtureLanguageModelAdapter(unittest.TestCase):
    """Tests for language model"""
    @classmethod
    def setUpClass(cls):
        dirname = os.path.dirname(__file__) or '.'
        cls.kenlm_path = "lm_dec19_char_tiny_12gram.kenlm"
        print(cls.kenlm_path)
        cls.lm_params = [{"lm_path": cls.kenlm_path}, {"lang_model_name": "gpt2"}]
        cls.lmodel = MixtureLanguageModelAdapter(lm_types=["NGRAM", "CAUSAL"], lm_weights=[0.5, 0.5],
                                                 lm_params=cls.lm_params)
        cls.lmodel.set_symbol_set(DEFAULT_SYMBOL_SET)


    @pytest.mark.slow
    def test_default_load(self):
        """Test loading model with parameters from json
        This test requires a valid lm_params.json file and all referenced models"""
        lm = MixtureLanguageModelAdapter()
        lm.set_symbol_set(DEFAULT_SYMBOL_SET)


    def test_init(self):
        """Test default parameters"""
        self.assertEqual(self.lmodel.symbol_set, DEFAULT_SYMBOL_SET)
        self.assertTrue(isinstance(self.lmodel, CharacterLanguageModel))


    def test_invalid_symbol_set(self):
        """Should raise an exception if predict is called without setting symbol set"""
        with self.assertRaises(InvalidSymbolSetException):
            lm = MixtureLanguageModelAdapter()
            lm.predict_character("this_should_fail")

    def test_invalid_model_type(self):
        """Test that the proper exception is thrown if given an invalid lm_type"""
        with self.assertRaises(InvalidLanguageModelException):
            lm = MixtureLanguageModelAdapter(lm_types=["PHONY", "CAUSAL"], lm_weights=[0.5, 0.5],
                                             lm_params=[{}, {"lang_model_name": "gpt2"}])
            lm.set_symbol_set(DEFAULT_SYMBOL_SET)

        with self.assertRaises(InvalidLanguageModelException):
            lm = MixtureLanguageModelAdapter(lm_types=["CAUSAL", "PHONY"], lm_weights=[0.5, 0.5],
                                             lm_params=[{"lang_model_name": "gpt2"}, {}])
            lm.set_symbol_set(DEFAULT_SYMBOL_SET)


    def test_invalid_model_weights(self):
        """Test that the proper exception is thrown if given an improper number of lm_weights"""
        with self.assertRaises(InvalidLanguageModelException, msg="Exception not thrown when too few weights given"):
            lm = MixtureLanguageModelAdapter(lm_types=["NGRAM", "CAUSAL"], lm_weights=[0.5],
                                             lm_params=[{"lm_path": self.kenlm_path}, {"lang_model_name": "gpt2"}])
            lm.set_symbol_set(DEFAULT_SYMBOL_SET)

        with self.assertRaises(InvalidLanguageModelException, msg="Exception not thrown when no weights given"):
            lm = MixtureLanguageModelAdapter(lm_types=["NGRAM", "CAUSAL"], lm_weights=None,
                                             lm_params=[{"lm_path": self.kenlm_path}, {"lang_model_name": "gpt2"}])
            lm.set_symbol_set(DEFAULT_SYMBOL_SET)

        with self.assertRaises(InvalidLanguageModelException, msg="Exception not thrown when too many weights given"):
            lm = MixtureLanguageModelAdapter(lm_types=["NGRAM", "CAUSAL"], lm_weights=[0.2, 0.3, 0.5],
                                             lm_params=[{"lm_path": self.kenlm_path}, {"lang_model_name": "gpt2"}])
            lm.set_symbol_set(DEFAULT_SYMBOL_SET)

        with self.assertRaises(InvalidLanguageModelException, msg="Exception not thrown when weights given do not \
                                 sum to 1"):
            lm = MixtureLanguageModelAdapter(lm_types=["NGRAM", "CAUSAL"], lm_weights=[0.5, 0.8],
                                             lm_params=[{"lm_path": self.kenlm_path}, {"lang_model_name": "gpt2"}])
            lm.set_symbol_set(DEFAULT_SYMBOL_SET)


    def test_non_mutable_evidence(self):
        """Test that the model does not change the evidence variable passed in."""
        evidence = list("Test_test")
        evidence2 = list("Test_test")
        self.lmodel.predict_character(evidence)
        self.assertEqual(evidence, evidence2)

    def test_identical(self):
        """Ensure predictions are the same for subsequent queries with the same evidence."""
        query1 = self.lmodel.predict_character(list("evidenc"))
        query2 = self.lmodel.predict_character(list("evidenc"))
        for ((sym1, prob1), (sym2, prob2)) in zip(query1, query2):
            self.assertAlmostEqual(prob1, prob2, places=5)
            self.assertEqual(sym1, sym2)

    def test_upper_lower_case(self):
        """Ensure predictions are the same for upper or lower case evidence."""
        lc = self.lmodel.predict_character(list("EVIDENC"))
        uc = self.lmodel.predict_character(list("evidenc"))
        for ((l_sym, l_prob), (u_sym, u_prob)) in zip(lc, uc):
            self.assertAlmostEqual(l_prob, u_prob, places=5)
            self.assertEqual(l_sym, u_sym)

    def test_predict_start_of_word(self):
        """Test the predict method with no prior evidence."""
        symbol_probs = self.lmodel.predict_character(evidence=[])
        probs = [prob for sym, prob in symbol_probs]

        self.assertTrue(
            len(set(probs)) > 1,
            "All values should not be the same probability")
        # Consider whether the following assertion should be True
        # backspace_prob = next(prob for sym, prob in probs if sym == BACKSPACE_CHAR)
        # self.assertEqual(0, backspace_prob)
        for prob in probs:
            self.assertTrue(0 <= prob < 1)
        self.assertAlmostEqual(sum(probs), 1, places=5)

    def test_predict_middle_of_word(self):
        """Test the predict method in the middle of a word."""
        symbol_probs = self.lmodel.predict_character(evidence=list("TH"))
        probs = [prob for sym, prob in symbol_probs]

        self.assertTrue(
            len(set(probs)) > 1,
            "All values should not be the same probability")
        for prob in probs:
            self.assertTrue(0 <= prob < 1)
        self.assertAlmostEqual(sum(probs), 1, places=5)

        most_likely_sym, _prob = sorted(symbol_probs,
                                        key=itemgetter(1),
                                        reverse=True)[0]
        self.assertEqual('E', most_likely_sym,
                         "Should predict 'E' as the next most likely symbol")

    def test_phrase(self):
        """Test that a phrase can be used for input"""
        symbol_probs = self.lmodel.predict_character(list("does_it_make_sen"))
        most_likely_sym, _prob = sorted(symbol_probs,
                                        key=itemgetter(1),
                                        reverse=True)[0]
        self.assertEqual('S', most_likely_sym)

    def test_multiple_spaces(self):
        """Test that the probability of space after a space is smaller than before the space"""
        symbol_probs_before = self.lmodel.predict_character(list("the"))
        symbol_probs_after = self.lmodel.predict_character(list("the_n"))
        space_prob_before = (dict(symbol_probs_before))[SPACE_CHAR]
        space_prob_after = (dict(symbol_probs_after))[SPACE_CHAR]
        self.assertTrue(space_prob_before > space_prob_after)

    def test_nonzero_prob(self):
        """Test that all letters in the alphabet have nonzero probability except for backspace"""
        symbol_probs = self.lmodel.predict_character(list("does_it_make_sens"))
        prob_values = [item[1] for item in symbol_probs if item[0] != BACKSPACE_CHAR]
        for value in prob_values:
            self.assertTrue(value > 0)
