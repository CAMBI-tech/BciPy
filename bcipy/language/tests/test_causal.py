"""Tests for CAUSAL Language Model"""

import pytest
import unittest
from operator import itemgetter

from bcipy.exceptions import UnsupportedResponseType
from bcipy.core.symbols import DEFAULT_SYMBOL_SET, BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.model.causal import CausalLanguageModelAdapter
from bcipy.language.main import ResponseType

from aactextpredict.exceptions import InvalidLanguageModelException

@pytest.mark.slow
class TestCausalLanguageModelAdapter(unittest.TestCase):
    """Tests for language model"""
    @classmethod
    def setUpClass(cls):
        cls.gpt2_model = CausalLanguageModelAdapter(response_type=ResponseType.SYMBOL, lang_model_name="gpt2")
        cls.opt_model = CausalLanguageModelAdapter(response_type=ResponseType.SYMBOL, lang_model_name="facebook/opt-125m")

    @pytest.mark.slow
    def test_default_load(self):
        """Test loading model with parameters from json
        This test requires a valid lm_params.json file and all requisite models"""
        lm = CausalLanguageModelAdapter(response_type=ResponseType.SYMBOL)

    def test_gpt2_init(self):
        """Test default parameters for GPT-2 model"""
        self.assertEqual(self.gpt2_model.response_type, ResponseType.SYMBOL)
        self.assertEqual(self.gpt2_model.symbol_set, DEFAULT_SYMBOL_SET)
        self.assertTrue(
            ResponseType.SYMBOL in self.gpt2_model.supported_response_types())
        self.assertEqual(self.gpt2_model.model.left_context, "<|endoftext|>")
        self.assertEqual(self.gpt2_model.model.device, "cpu")

    def test_opt_init(self):
        """Test default parameters for Facebook OPT model"""
        self.assertEqual(self.opt_model.response_type, ResponseType.SYMBOL)
        self.assertEqual(self.opt_model.symbol_set, DEFAULT_SYMBOL_SET)
        self.assertTrue(
            ResponseType.SYMBOL in self.opt_model.supported_response_types())
        self.assertEqual(self.opt_model.model.left_context, "</s>")
        self.assertEqual(self.opt_model.model.device, "cpu")

    def test_name(self):
        """Test model name."""
        self.assertEqual("CAUSAL", CausalLanguageModelAdapter.name())

    def test_unsupported_response_type(self):
        """Unsupported responses should raise an exception"""
        with self.assertRaises(UnsupportedResponseType):
            CausalLanguageModelAdapter(response_type=ResponseType.WORD,
                                lang_model_name="gpt2")

    def test_invalid_model_name(self):
        """Test that the proper exception is thrown if given an invalid lang_model_name"""
        with self.assertRaises(InvalidLanguageModelException):
            CausalLanguageModelAdapter(response_type=ResponseType.SYMBOL,
                                lang_model_name="phonymodel")

    def test_invalid_model_path(self):
        """Test that the proper exception is thrown if given an invalid lm_path"""
        with self.assertRaises(InvalidLanguageModelException):
            CausalLanguageModelAdapter(response_type=ResponseType.SYMBOL,
                                lang_model_name="gpt2", lm_path="./phonypath/")

    def test_non_mutable_evidence(self):
        """Test that the model does not change the evidence variable passed in.
           This could impact the mixture model if failed"""
        evidence = list("Test_test")
        evidence2 = list("Test_test")
        self.gpt2_model.predict(evidence)
        self.assertEqual(evidence, evidence2)

        self.opt_model.predict(evidence)
        self.assertEqual(evidence, evidence2)

    def test_gpt2_identical(self):
        """Ensure predictions are the same for subsequent queries with the same evidence."""
        query1 = self.gpt2_model.predict(list("evidenc"))
        query2 = self.gpt2_model.predict(list("evidenc"))
        for ((sym1, prob1), (sym2, prob2)) in zip(query1, query2):
            self.assertAlmostEqual(prob1, prob2, places=5)
            self.assertEqual(sym1, sym2)

    def test_opt_identical(self):
        """Ensure predictions are the same for subsequent queries with the same evidence."""
        query1 = self.opt_model.predict(list("evidenc"))
        query2 = self.opt_model.predict(list("evidenc"))
        for ((sym1, prob1), (sym2, prob2)) in zip(query1, query2):
            self.assertAlmostEqual(prob1, prob2, places=5)
            self.assertEqual(sym1, sym2)

    def test_gpt2_upper_lower_case(self):
        """Ensure predictions are the same for upper or lower case evidence."""
        lc = self.gpt2_model.predict(list("EVIDENC"))
        uc = self.gpt2_model.predict(list("evidenc"))
        for ((l_sym, l_prob), (u_sym, u_prob)) in zip(lc, uc):
            self.assertAlmostEqual(l_prob, u_prob, places=5)
            self.assertEqual(l_sym, u_sym)

    def test_opt_upper_lower_case(self):
        """Ensure predictions are the same for upper or lower case evidence."""
        lc = self.opt_model.predict(list("EVIDENC"))
        uc = self.opt_model.predict(list("evidenc"))
        for ((l_sym, l_prob), (u_sym, u_prob)) in zip(lc, uc):
            self.assertAlmostEqual(l_prob, u_prob, places=5)
            self.assertEqual(l_sym, u_sym)

    def test_gpt2_predict_start_of_word(self):
        """Test the gpt2 predict method with no prior evidence."""
        symbol_probs = self.gpt2_model.predict(evidence=[])
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

    def test_opt_predict_start_of_word(self):
        """Test the Facebook opt predict method with no prior evidence."""
        symbol_probs = self.opt_model.predict(evidence=[])
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

    def test_gpt2_predict_middle_of_word(self):
        """Test the predict method in the middle of a word with gpt2 model."""
        symbol_probs = self.gpt2_model.predict(evidence=list("TH"))
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

    def test_opt_predict_middle_of_word(self):
        """Test the predict method in the middle of a word with Facebook opt model."""
        symbol_probs = self.opt_model.predict(evidence=list("TH"))
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

    def test_gpt2_phrase(self):
        """Test that a phrase can be used for input with gpt2 model"""
        symbol_probs = self.gpt2_model.predict(list("does_it_make_sen"))
        most_likely_sym, _prob = sorted(symbol_probs,
                                        key=itemgetter(1),
                                        reverse=True)[0]
        self.assertEqual('S', most_likely_sym)

    def test_opt_phrase(self):
        """Test that a phrase can be used for input with Facebook opt model"""
        symbol_probs = self.opt_model.predict(list("does_it_make_sen"))
        most_likely_sym, _prob = sorted(symbol_probs,
                                        key=itemgetter(1),
                                        reverse=True)[0]
        self.assertEqual('S', most_likely_sym)

    def test_gpt2_multiple_spaces(self):
        """Test that the probability of space after a space is smaller than before the space"""
        symbol_probs_before = self.gpt2_model.predict(list("the"))
        symbol_probs_after = self.gpt2_model.predict(list("the_"))
        space_prob_before = (dict(symbol_probs_before))[SPACE_CHAR]
        space_prob_after = (dict(symbol_probs_after))[SPACE_CHAR]
        self.assertTrue(space_prob_before > space_prob_after)

    def test_opt_multiple_spaces(self):
        """Test that the probability of space after a space is smaller than before the space"""
        symbol_probs_before = self.opt_model.predict(list("the"))
        symbol_probs_after = self.opt_model.predict(list("the_"))
        space_prob_before = (dict(symbol_probs_before))[SPACE_CHAR]
        space_prob_after = (dict(symbol_probs_after))[SPACE_CHAR]
        self.assertTrue(space_prob_before > space_prob_after)

    def test_gpt2_nonzero_prob(self):
        """Test that all letters in the alphabet have nonzero probability except for backspace"""
        symbol_probs = self.gpt2_model.predict(list("does_it_make_sens"))
        prob_values = [item[1] for item in symbol_probs if item[0] != BACKSPACE_CHAR]
        for value in prob_values:
            self.assertTrue(value > 0)

    def test_opt_nonzero_prob(self):
        """Test that all letters in the alphabet have nonzero probability except for backspace"""
        symbol_probs = self.opt_model.predict(list("does_it_make_sens"))
        prob_values = [item[1] for item in symbol_probs if item[0] != BACKSPACE_CHAR]
        for value in prob_values:
            self.assertTrue(value > 0)
