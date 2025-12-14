"""Tests for Uniform Language Model"""

import unittest

from bcipy.core.symbols import BACKSPACE_CHAR, DEFAULT_SYMBOL_SET
from bcipy.exceptions import InvalidSymbolSetException
from bcipy.language.main import CharacterLanguageModel
from bcipy.language.model.uniform import UniformLanguageModel, equally_probable


class TestUniformLanguageModel(unittest.TestCase):
    """Tests for language model"""

    @classmethod
    def setUpClass(cls):
        cls.lm = UniformLanguageModel()
        cls.lm.set_symbol_set(DEFAULT_SYMBOL_SET)

    def test_init(self):
        """Test default parameters"""
        lmodel = UniformLanguageModel()
        lmodel.set_symbol_set(DEFAULT_SYMBOL_SET)
        self.assertEqual(
            len(lmodel.symbol_set), 28,
            "Should be the alphabet plus the backspace and space chars")
        self.assertTrue(isinstance(lmodel, CharacterLanguageModel))

    def test_invalid_symbol_set(self):
        """Should raise an exception if predict is called before setting a symbol set"""
        lm = UniformLanguageModel()
        lm.set_symbol_set([])
        with self.assertRaises(InvalidSymbolSetException):
            lm.predict_character("this_should_fail")

    def test_predict(self):
        """Test the predict method"""
        symbol_probs = self.lm.predict_character(evidence=[])

        # Backspace can be 0
        probs = [prob for sym, prob in symbol_probs if sym != BACKSPACE_CHAR]

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
