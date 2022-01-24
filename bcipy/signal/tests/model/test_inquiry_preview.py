from abc import ABC
import unittest
import numpy as np
from bcipy.signal.model.inquiry_preview import compute_probs_after_preview
from string import ascii_uppercase


class InqPreviewBaseTest(ABC):
    def _compare(self, results, p_shown, p_not_shown):
        expected = np.array(self.show_first * [p_shown] + (len(self.symbol_set) - self.show_first) * [p_not_shown])
        expected /= expected.sum()
        self.assertTrue(np.allclose(results, expected))

    def test_user_likes_inquiry(self):
        # Upvote the shown letters
        results = compute_probs_after_preview(self.inquiry, self.symbol_set, self.error_prob, proceed=True)
        p_shown = self.uniform_prior * (1 - self.error_prob)
        p_not_shown = self.uniform_prior * self.error_prob
        self._compare(results, p_shown, p_not_shown)

    def test_user_dislikes_inquiry(self):
        # Downvote the shown letters
        results = compute_probs_after_preview(self.inquiry, self.symbol_set, self.error_prob, proceed=False)
        p_shown = self.uniform_prior * self.error_prob
        p_not_shown = self.uniform_prior * (1 - self.error_prob)
        self._compare(results, p_shown, p_not_shown)

    def test_invalid_error_prob(self):
        with self.assertRaises(ValueError):
            compute_probs_after_preview(self.inquiry, self.symbol_set, -0.01, True)

        with self.assertRaises(ValueError):
            compute_probs_after_preview(self.inquiry, self.symbol_set, 1.01, True)

    def test_user_likes_with_LM(self):
        results = compute_probs_after_preview(self.inquiry, self.symbol_set, self.error_prob, proceed=True)
        p_shown = self.uniform_prior * (1 - self.error_prob)
        p_not_shown = self.uniform_prior * self.error_prob

        # Dummy LM only uses 2 values (V and 2*V, such that the alphabet is normalized
        # Its large value is used for some shown, and some unshown letters. Its small value also used for both.
        lm_prior = np.ones(len(self.symbol_set))
        lm_prior[0 : self.show_first - 1] = 2
        lm_prior[-1] = 2
        lm_prior /= lm_prior.sum()

        update = np.array(self.show_first * [p_shown] + (len(self.symbol_set) - self.show_first) * [p_not_shown])
        update /= update.sum()

        expected = update * lm_prior
        expected /= expected.sum()
        breakpoint()

    def test_user_dislikes_with_LM(self):
        results = compute_probs_after_preview(self.inquiry, self.symbol_set, self.error_prob, proceed=False)


class TestShortAlphabet(unittest.TestCase, InqPreviewBaseTest):
    def setUp(self):
        self.symbol_set = list("ABCDEFG")
        self.inquiry, self.show_first = list("ABC"), 3
        self.error_prob = 0.05
        self.uniform_prior = 1 / len(self.symbol_set)


class TestLongAlphabet(unittest.TestCase, InqPreviewBaseTest):
    def setUp(self):
        self.symbol_set = list(ascii_uppercase)
        self.inquiry, self.show_first = list("ABCDE"), 5
        self.error_prob = 0.05
        self.uniform_prior = 1 / len(self.symbol_set)


if __name__ == "__main__":
    unittest.main()
