import unittest
import numpy as np
from bcipy.signal.model.inquiry_preview import compute_probs_after_preview


class TestInquiryPreview(unittest.TestCase):
    def setUp(self):
        self.inquiry = ["A", "E", "I", "O", "U"]
        self.symbol_set = self.inquiry + ["Y"]
        self.error_prob = 0.05

    def test_user_likes_inquiry(self):
        proceed = True
        results = compute_probs_after_preview(self.inquiry, self.symbol_set, self.error_prob, proceed)

        p_shown = 0.19  # (1 - 0.05) / 5
        p_not_shown = 0.05  # 0.05 / (6 - 1)
        expected = [p_shown, p_shown, p_shown, p_shown, p_shown, p_not_shown]

        self.assertTrue(np.allclose(results, expected))

    def test_user_dislikes_inquiry(self):
        proceed = False
        results = compute_probs_after_preview(self.inquiry, self.symbol_set, self.error_prob, proceed)

        p_shown = 0.01  # 0.05 / 5
        p_not_shown = 0.95  # (1 - 0.05) / 1
        expected = [p_shown, p_shown, p_shown, p_shown, p_shown, p_not_shown]

        self.assertTrue(np.allclose(results, expected))

    def test_invalid_error_prob(self):
        with self.assertRaises(ValueError):
            compute_probs_after_preview(self.inquiry, self.symbol_set, -0.01, True)

        with self.assertRaises(ValueError):
            compute_probs_after_preview(self.inquiry, self.symbol_set, 1.01, True)


if __name__ == "__main__":
    unittest.main()
