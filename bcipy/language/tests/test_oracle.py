"""Tests for Oracle Language Model"""

import unittest

from bcipy.language.model.oracle import (BACKSPACE_CHAR, OracleLanguageModel,
                                         ResponseType)


class TestOracleLanguageModel(unittest.TestCase):
    """Tests for language model"""

    def test_init(self):
        """Test default parameters"""
        with self.assertRaises(AssertionError):
            OracleLanguageModel()

    def test_init_with_text(self):
        """Test with task_text provided"""
        lmodel = OracleLanguageModel(task_text="HELLO_WORLD")
        self.assertEqual(lmodel.response_type, ResponseType.SYMBOL)
        self.assertEqual(
            len(lmodel.symbol_set), 28,
            "Should be the alphabet plus the backspace and space chars")

    def test_predict(self):
        """Test the predict method"""
        lm = OracleLanguageModel(task_text="HELLO_WORLD")
        symbol_probs = lm.predict(evidence=[])
        probs = [prob for sym, prob in symbol_probs]

        self.assertEqual(len(set(probs)), 2,
                         "All non-target values should be the same")
        self.assertTrue(0 < probs[0] < 1)
        self.assertAlmostEqual(sum(probs), 1)

        probs_dict = dict(symbol_probs)
        self.assertTrue(probs_dict['H'] > probs_dict['A'],
                        "Target should have a higher value")
        self.assertAlmostEqual(lm.target_bump,
                               probs_dict['H'] - probs_dict['A'],
                               places=1)

    def test_predict_with_spelled_text(self):
        """Test predictions with previously spelled symbols"""
        lm = OracleLanguageModel(task_text="HELLO_WORLD")
        symbol_probs = lm.predict(evidence=list("HELLO_"))

        probs = [prob for sym, prob in symbol_probs]
        self.assertEqual(len(set(probs)), 2,
                         "All non-target values should be the same")

        probs_dict = dict(symbol_probs)
        self.assertTrue(probs_dict['W'] > probs_dict['A'])

    def test_predict_with_incorrectly_spelled_text(self):
        """Test predictions with incorrectly spelled prior."""
        lm = OracleLanguageModel(task_text="HELLO_WORLD")
        symbol_probs = lm.predict(evidence=list("HELP"))

        probs = [prob for sym, prob in symbol_probs]
        self.assertEqual(len(set(probs)), 2)

        probs_dict = dict(symbol_probs)
        self.assertTrue(probs_dict[BACKSPACE_CHAR] > probs_dict['A'])

    def test_target_bump_parameter(self):
        """Test setting the target_bump parameter."""
        lm = OracleLanguageModel(task_text="HELLO_WORLD", target_bump=0.2)
        symbol_probs = lm.predict(evidence=[])
        probs_dict = dict(symbol_probs)
        self.assertTrue(probs_dict['H'] > probs_dict['A'],
                        "Target should have a higher value")
        self.assertAlmostEqual(0.2,
                               probs_dict['H'] - probs_dict['A'],
                               places=1)
