import unittest
from string import ascii_uppercase

import numpy as np

from bcipy.signal.model.inquiry_preview import compute_probs_after_preview


class TestInquiryPreview(unittest.TestCase):
    """Test button press probabilities under various conditions."""

    def setUp(self):
        self.symbol_set = list(ascii_uppercase)
        self.error_prob = 0.05

    def test_user_likes_short_inquiry(self):
        """Test short inquiry where user wants to proceed with the inquiry.
        Should provide support for symbols in inquiry and downvote others."""

        inquiry = ['A', 'B', 'C', 'D', 'E']
        user_likes = True
        expected = [
            0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05
        ]
        results = compute_probs_after_preview(inquiry, self.symbol_set,
                                              self.error_prob, user_likes)
        self.assertTrue(np.allclose(results, expected))

    def test_user_likes_long_inquiry(self):
        """Test long inquiry where user wants to proceed with the inquiry."""
        inquiry = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O'
        ]
        expected = [
            0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
            0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05
        ]
        results = compute_probs_after_preview(inquiry,
                                              self.symbol_set,
                                              self.error_prob,
                                              proceed=True)

        self.assertTrue(np.allclose(results, expected))

    def test_user_dislikes_short_inquiry(self):
        """Test probabilities for short inquiry where user does not want to proceed.
        Should downvote letters in the inquiry and upvote everything else."""

        inquiry = ['A', 'B', 'C', 'D', 'E']
        expected = [
            0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
            0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
            0.95, 0.95, 0.95, 0.95
        ]
        results = compute_probs_after_preview(inquiry,
                                              self.symbol_set,
                                              self.error_prob,
                                              proceed=False)
        self.assertTrue(np.allclose(results, expected))

    def test_user_dislikes_long_inquiry(self):
        """Test probabilities for long inquiry where user does not want to proceed."""
        inquiry = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O'
        ]
        expected = [
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
            0.95, 0.95, 0.95, 0.95
        ]
        results = compute_probs_after_preview(inquiry,
                                              self.symbol_set,
                                              self.error_prob,
                                              proceed=False)
        self.assertTrue(np.allclose(results, expected))

    def test_invalid_error_prob(self):
        """Test error probability out of range"""
        inquiry = self.symbol_set[:10]
        with self.assertRaises(ValueError):
            compute_probs_after_preview(inquiry, self.symbol_set, -0.01, True)

        with self.assertRaises(ValueError):
            compute_probs_after_preview(inquiry, self.symbol_set, 1.01, True)


if __name__ == "__main__":
    unittest.main()
