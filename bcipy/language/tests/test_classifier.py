"""Tests for CAUSAL Language Model"""

import pytest
import unittest
from operator import itemgetter

from bcipy.helpers.exceptions import UnsupportedResponseType, InvalidLanguageModelException
from bcipy.helpers.symbols import alphabet, BACKSPACE_CHAR, SPACE_CHAR
from bcipy.language.model.classifier import ClassifierLanguageModel
from bcipy.language.main import ResponseType


@pytest.mark.slow
class TestClassifierLanguageModel(unittest.TestCase):
    """Tests for language model"""
    @classmethod
    def setUpClass(cls):
        cls.model = ClassifierLanguageModel(response_type=ResponseType.SYMBOL,
                                             symbol_set=alphabet(), lang_model_name="microsoft/deberta-v3-xsmall",
                                             lm_path="../lms/deberta-char-classifier")

    # @pytest.mark.slow
    # def test_default_load(self):
    #     """Test loading model with parameters from json
    #     This test requires a valid lm_params.json file and all requisite models"""
    #     lm = CausalLanguageModel(response_type=ResponseType.SYMBOL, symbol_set=alphabet())

    def test_model_init(self):
        """Test init of Deberta model"""
        self.assertEqual(self.model.response_type, ResponseType.SYMBOL)
        self.assertEqual(self.model.symbol_set, alphabet())
        self.assertTrue(
            ResponseType.SYMBOL in self.model.supported_response_types())
        self.assertEqual(self.model.device, "cpu")

    def test_name(self):
        """Test model name."""
        self.assertEqual("CLASSIFIER", ClassifierLanguageModel.name())

    def test_unsupported_response_type(self):
        """Unsupported responses should raise an exception"""
        with self.assertRaises(UnsupportedResponseType):
            ClassifierLanguageModel(response_type=ResponseType.WORD,
                                symbol_set=alphabet(), lang_model_name="microsoft/deberta-v3-xsmall")

    def test_invalid_model_name(self):
        """Test that the proper exception is thrown if given an invalid lang_model_name"""
        with self.assertRaises(InvalidLanguageModelException):
            ClassifierLanguageModel(response_type=ResponseType.SYMBOL, symbol_set=alphabet(),
                                lang_model_name="phonymodel")

    def test_invalid_model_path(self):
        """Test that the proper exception is thrown if given an invalid lm_path"""
        with self.assertRaises(InvalidLanguageModelException):
            ClassifierLanguageModel(response_type=ResponseType.SYMBOL, symbol_set=alphabet(),
                                lang_model_name="microsoft/deberta-v3-xsmall", lm_path="./phonypath/")

    def test_non_mutable_evidence(self):
        """Test that the model does not change the evidence variable passed in.
           This could impact the mixture model if failed"""
        evidence = list("Test_test")
        evidence2 = list("Test_test")
        self.model.predict(evidence)
        self.assertEqual(evidence, evidence2)

    def test_identical(self):
        """Ensure predictions are the same for subsequent queries with the same evidence."""
        query1 = self.model.predict(list("evidenc"))
        query2 = self.model.predict(list("evidenc"))
        for ((sym1, prob1), (sym2, prob2)) in zip(query1, query2):
            self.assertAlmostEqual(prob1, prob2, places=5)
            self.assertEqual(sym1, sym2)


    def test_predict_start_of_word(self):
        """Test the predict method with no prior evidence."""
        symbol_probs = self.model.predict(evidence=[])
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
        """Test the predict method in the middle of a word with gpt2 model."""
        symbol_probs = self.model.predict(evidence=list("TH"))
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
        """Test that a phrase can be used for input with classifier model"""
        symbol_probs = self.gpt2_model.predict(list("does_it_make_sen"))
        most_likely_sym, _prob = sorted(symbol_probs,
                                        key=itemgetter(1),
                                        reverse=True)[0]
        self.assertEqual('S', most_likely_sym)

    def test_multiple_spaces(self):
        """Test that the probability of space after a space is smaller than before the space"""
        symbol_probs_before = self.model.predict(list("the"))
        symbol_probs_after = self.model.predict(list("the_"))
        space_prob_before = (dict(symbol_probs_before))[SPACE_CHAR]
        space_prob_after = (dict(symbol_probs_after))[SPACE_CHAR]
        self.assertTrue(space_prob_before > space_prob_after)

    def test_nonzero_prob(self):
        """Test that all letters in the alphabet have nonzero probability except for backspace"""
        symbol_probs = self.model.predict(list("does_it_make_sens"))
        prob_values = [item[1] for item in symbol_probs if item[0] != BACKSPACE_CHAR]
        for value in prob_values:
            self.assertTrue(value > 0)
