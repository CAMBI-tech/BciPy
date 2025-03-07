"""Tests for Oracle Language Model"""

import unittest

from bcipy.language.model.oracle import OracleLanguageModelAdapter
from bcipy.language.main import ResponseType
from bcipy.core.symbols import BACKSPACE_CHAR


class TestOracleLanguageModelAdapter(unittest.TestCase):
    """Tests for language model"""

    def test_init(self):
        """Test default parameters"""
        with self.assertRaises(AssertionError):
            OracleLanguageModelAdapter()

    def test_init_with_text(self):
        """Test with task_text provided"""
        lmodel = OracleLanguageModelAdapter(task_text="HELLO_WORLD")
        self.assertEqual(lmodel.response_type, ResponseType.SYMBOL)
        self.assertEqual(
            len(lmodel.symbol_set), 28,
            "Should be the alphabet plus the backspace and space chars")

    def test_name(self):
        """Test model name."""
        self.assertEqual("ORACLE", OracleLanguageModelAdapter.name())

    def test_predict(self):
        """Test the predict method"""
        lm = OracleLanguageModelAdapter(task_text="HELLO_WORLD")
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
                               places=4)

    def test_predict_with_spelled_text(self):
        """Test predictions with previously spelled symbols"""
        lm = OracleLanguageModelAdapter(task_text="HELLO_WORLD")
        symbol_probs = lm.predict(evidence=list("HELLO_"))

        probs = [prob for sym, prob in symbol_probs]
        self.assertEqual(len(set(probs)), 2,
                         "All non-target values should be the same")

        probs_dict = dict(symbol_probs)
        self.assertTrue(probs_dict['W'] > probs_dict['A'])

    def test_predict_with_incorrectly_spelled_text(self):
        """Test predictions with incorrectly spelled prior."""
        lm = OracleLanguageModelAdapter(task_text="HELLO_WORLD")
        symbol_probs = lm.predict(evidence=list("HELP"))

        probs = [prob for sym, prob in symbol_probs]
        self.assertEqual(len(set(probs)), 2)

        probs_dict = dict(symbol_probs)
        self.assertTrue(probs_dict[BACKSPACE_CHAR] > probs_dict['A'])

    def test_target_bump_parameter(self):
        """Test setting the target_bump parameter."""
        lm = OracleLanguageModelAdapter(task_text="HELLO_WORLD", target_bump=0.2)
        symbol_probs = lm.predict(evidence=[])
        probs_dict = dict(symbol_probs)
        self.assertTrue(probs_dict['H'] > probs_dict['A'],
                        "Target should have a higher value")
        self.assertAlmostEqual(0.2,
                               probs_dict['H'] - probs_dict['A'],
                               places=4)

    def test_setting_task_text_to_none(self):
        """Test that task_text is required"""
        lmodel = OracleLanguageModelAdapter(task_text="HELLO_WORLD")
        with self.assertRaises(AssertionError):
            lmodel.task_text = None

    def test_updating_task_text(self):
        """Test updating the task_text property."""
        lm = OracleLanguageModelAdapter(task_text="HELLO_WORLD", target_bump=0.2)
        probs = dict(lm.predict(evidence=list("HELLO_")))
        self.assertTrue(probs['W'] > probs['T'],
                        "Target should have a higher value")

        lm.task_text = "HELLO_THERE"
        probs = dict(lm.predict(evidence=list("HELLO_")))
        self.assertTrue(probs['T'] > probs['W'],
                        "Target should have a higher value")

    def test_target_bump_bounds(self):
        """Test the bounds of target_bump property"""
        with self.assertRaises(AssertionError):
            OracleLanguageModelAdapter(task_text="HI", target_bump=-1.0)

        with self.assertRaises(AssertionError):
            OracleLanguageModelAdapter(task_text="HI", target_bump=1.1)

        lm = OracleLanguageModelAdapter(task_text="HI", target_bump=0.0)
        with self.assertRaises(AssertionError):
            lm.target_bump = -1.0

        lm.target_bump = 0.5
        self.assertEqual(0.5, lm.target_bump)

    def test_evidence_exceeds_task(self):
        """Test probs when evidence exceeds task_text."""
        lm = OracleLanguageModelAdapter(task_text="HELLO")

        probs = dict(lm.predict(evidence="HELL"))
        self.assertEqual(2, len(set(probs.values())))
        self.assertEqual(max(probs.values()), probs['O'])

        probs = dict(lm.predict(evidence="HELLO"))
        self.assertEqual(1, len(set(probs.values())))

        probs = dict(lm.predict(evidence="HELLP"))
        self.assertEqual(2, len(set(probs.values())))
        self.assertEqual(max(probs.values()), probs[BACKSPACE_CHAR])

        probs = dict(lm.predict(evidence="HELLO_"))
        self.assertEqual(1, len(set(probs.values())))

        probs = dict(lm.predict(evidence="HELPED"))
        self.assertEqual(2, len(set(probs.values())))
        self.assertEqual(max(probs.values()), probs[BACKSPACE_CHAR])


if __name__ == '__main__':
    unittest.main()
