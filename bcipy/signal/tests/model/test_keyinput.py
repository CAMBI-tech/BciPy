import unittest
import numpy as np
from bcipy.signal.model.keyinput import compute_keyinput_probs


class TestKeyinput(unittest.TestCase):
    def setUp(self):
        self.inquiry = ["A", "E", "I", "O", "U"]
        self.symbol_set = self.inquiry + ["Y"]
        self.error_prob = 0.05

    def test_user_likes_inquiry(self):
        proceed = True
        results = compute_keyinput_probs(self.inquiry, self.symbol_set, self.error_prob, proceed)["KEYINPUT"]

        p_shown = (1 - self.error_prob) / len(self.inquiry)
        p_not_shown = (self.error_prob) / (len(self.symbol_set) - len(self.inquiry))
        expected = [p_shown] * 5 + [p_not_shown]

        self.assertTrue(np.allclose(results, expected))

    def test_user_dislikes_inquiry(self):
        proceed = False
        results = compute_keyinput_probs(self.inquiry, self.symbol_set, self.error_prob, proceed)["KEYINPUT"]

        p_shown = (self.error_prob) / len(self.inquiry)
        p_not_shown = (1 - self.error_prob) / (len(self.symbol_set) - len(self.inquiry))
        expected = [p_shown] * 5 + [p_not_shown]

        self.assertTrue(np.allclose(results, expected))

    def test_invalid_error_prob(self):
        with self.assertRaises(ValueError):
            compute_keyinput_probs(self.inquiry, self.symbol_set, -0.01, True)

        with self.assertRaises(ValueError):
            compute_keyinput_probs(self.inquiry, self.symbol_set, 1.01, True)


if __name__ == "__main__":
    unittest.main()