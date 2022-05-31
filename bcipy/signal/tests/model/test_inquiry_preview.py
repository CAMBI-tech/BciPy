import unittest
import numpy as np
from bcipy.signal.model.inquiry_preview import compute_probs_after_preview
from string import ascii_uppercase


class TestInquiryPreview(unittest.TestCase):
    def setUp(self):
        self.symbol_set = list(ascii_uppercase)

    def _test(self, inquiry_len, error_prob, user_likes: bool):
        if user_likes:
            p_shown = 1 - error_prob
            p_not_shown = error_prob
        else:
            p_shown = error_prob
            p_not_shown = 1 - error_prob

        inquiry = self.symbol_set[:inquiry_len]
        results = compute_probs_after_preview(inquiry, self.symbol_set, error_prob, user_likes)

        expected = np.array(inquiry_len * [p_shown] + (len(self.symbol_set) - inquiry_len) * [p_not_shown])
        self.assertTrue(np.allclose(results, expected))

    def test_user_likes_short_inquiry(self):
        self._test(5, 0.95, True)

    def test_user_likes_long_inquiry(self):
        self._test(15, 0.95, True)

    def test_user_dislikes_short_inquiry(self):
        self._test(5, 0.95, False)

    def test_user_dislikes_long_inquiry(self):
        self._test(15, 0.95, False)

    def test_invalid_error_prob(self):
        inquiry = self.symbol_set[:10]
        with self.assertRaises(ValueError):
            compute_probs_after_preview(inquiry, self.symbol_set, -0.01, True)

        with self.assertRaises(ValueError):
            compute_probs_after_preview(inquiry, self.symbol_set, 1.01, True)


if __name__ == "__main__":
    unittest.main()
