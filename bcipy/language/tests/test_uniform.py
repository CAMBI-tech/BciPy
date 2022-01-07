"""Tests for Uniform Language Model"""

import unittest
from bcipy.language.uniform import UniformLanguageModel, ResponseType, BACKSPACE_CHAR


class TestUniformLanguageModel(unittest.TestCase):
    """Tests for language model"""

    def test_init(self):
        """Test default parameters"""
        lmodel = UniformLanguageModel()
        self.assertEqual(lmodel.response_type, ResponseType.SYMBOL)
        self.assertTrue(lmodel.normalized)
        self.assertIsNone(lmodel.backspace_prob)
        self.assertEqual(
            len(lmodel.symbol_set), 28,
            "Should be the alphabet plus the backspace and space chars")

    def test_config(self):
        """Test configuration parameters."""
        lmodel = UniformLanguageModel(lm_backspace_prob=0.05)
        self.assertEqual(lmodel.backspace_prob, 0.05)

    def test_required_parameters(self):
        """Test that missing parameters raise an exception"""
        with self.assertRaises(AssertionError):
            UniformLanguageModel(is_txt_stim=False)

        with self.assertRaises(AssertionError):
            UniformLanguageModel(lm_backspace_prob=-3)

        with self.assertRaises(AssertionError):
            UniformLanguageModel(lm_backspace_prob=1.1)

    def test_predict(self):
        """Test the predict method"""
        symbol_probs = UniformLanguageModel().predict(evidence=[])
        probs = [prob for sym, prob in symbol_probs]

        self.assertEqual(len(set(probs)), 1, "All values should be the same")
        self.assertTrue(0 < probs[0] < 1)
        self.assertAlmostEqual(sum(probs), 1)

    def test_predict_with_fixed_backspace(self):
        """Test predict with a fixed backspace probability."""
        lmodel = UniformLanguageModel(lm_backspace_prob=0.05)
        symbol_probs = lmodel.predict(evidence=[])
        self.assertTrue((BACKSPACE_CHAR, 0.05) in symbol_probs)

        probs = [prob for sym, prob in symbol_probs]
        self.assertEqual(len(set(probs)), 2,
                         "Backspace probability should be different")
        self.assertTrue(0.05 in probs)
        self.assertAlmostEqual(sum(probs), 1)
