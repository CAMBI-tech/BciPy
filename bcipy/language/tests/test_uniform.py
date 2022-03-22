"""Tests for Uniform Language Model"""

import unittest

from bcipy.language.uniform import (ResponseType, UniformLanguageModel,
                                    equally_probable)


class TestUniformLanguageModel(unittest.TestCase):
    """Tests for language model"""

    def test_init(self):
        """Test default parameters"""
        lmodel = UniformLanguageModel()
        self.assertEqual(lmodel.response_type, ResponseType.SYMBOL)
        self.assertEqual(
            len(lmodel.symbol_set), 28,
            "Should be the alphabet plus the backspace and space chars")

    def test_predict(self):
        """Test the predict method"""
        symbol_probs = UniformLanguageModel().predict(evidence=[])
        probs = [prob for sym, prob in symbol_probs]

        self.assertEqual(len(set(probs)), 1, "All values should be the same")
        self.assertTrue(0 < probs[0] < 1)
        self.assertAlmostEqual(sum(probs), 1)

    def test_equally_probable(self):
        """Test generation of equally probable values."""

        # no overrides
        alp = ['A', 'B', 'C', 'D']
        probs = equally_probable(alp)
        self.assertEqual(len(alp), len(probs))
        for prob in probs:
            self.assertEqual(0.25, prob)

        # test with override
        alp = ['A', 'B', 'C', 'D']
        probs = equally_probable(alp, {'A': 0.4})
        self.assertEqual(len(alp), len(probs))
        self.assertAlmostEqual(1.0, sum(probs))
        self.assertEqual(probs[0], 0.4)
        self.assertAlmostEqual(probs[1], 0.2)
        self.assertAlmostEqual(probs[2], 0.2)
        self.assertAlmostEqual(probs[3], 0.2)

        # test with 0.0 override
        alp = ['A', 'B', 'C', 'D', 'E']
        probs = equally_probable(alp, {'E': 0.0})
        self.assertEqual(len(alp), len(probs))
        self.assertAlmostEqual(1.0, sum(probs))
        self.assertEqual(probs[0], 0.25)
        self.assertAlmostEqual(probs[1], 0.25)
        self.assertAlmostEqual(probs[2], 0.25)
        self.assertAlmostEqual(probs[3], 0.25)
        self.assertAlmostEqual(probs[4], 0.0)

        # test with multiple overrides
        alp = ['A', 'B', 'C', 'D']
        probs = equally_probable(alp, {'B': 0.2, 'D': 0.3})
        self.assertEqual(len(alp), len(probs))
        self.assertAlmostEqual(1.0, sum(probs))
        self.assertEqual(probs[0], 0.25)
        self.assertAlmostEqual(probs[1], 0.2)
        self.assertAlmostEqual(probs[2], 0.25)
        self.assertAlmostEqual(probs[3], 0.3)

        # test with override that's not in the alphabet
        alp = ['A', 'B', 'C', 'D']
        probs = equally_probable(alp, {'F': 0.4})
        self.assertEqual(len(alp), len(probs))
        self.assertAlmostEqual(1.0, sum(probs))
        for prob in probs:
            self.assertEqual(0.25, prob)
