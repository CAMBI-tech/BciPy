"""Tests for Button Press Model"""
import unittest
from string import ascii_uppercase

import numpy as np

from bcipy.signal.model.switch_model import SwitchModel


class SwitchModelTest(unittest.TestCase):
    """Tests for Switch Model."""

    def setUp(self):
        """Override; set up model."""
        self.model = SwitchModel()
        self.symbol_set = list(ascii_uppercase)

    def test_all_positive_inquiry(self):
        """Test model results when the data indicates that the inquiry should
        be supported. All inquiry symbols have value of 1.0"""
        inquiry = ['A', 'B', 'C', 'D', 'E']

        expected = [
            0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05
        ]
        results = self.model.compute_likelihood_ratio(
            data=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            inquiry=inquiry,
            symbol_set=self.symbol_set)

        self.assertTrue(np.allclose(results, expected))

    def test_any_positive_inquiry(self):
        """Test model results when the data indicates that the inquiry should
        be supported. Some inquiry symbols have value of 1.0"""
        inquiry = ['A', 'B', 'C', 'D', 'E']

        expected = [
            0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
            0.05, 0.05, 0.05, 0.05
        ]
        results = self.model.compute_likelihood_ratio(
            data=np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            inquiry=inquiry,
            symbol_set=self.symbol_set)

        self.assertTrue(np.allclose(results, expected))

    def test_all_negative_inquiry(self):
        """Test model results when the data indicates that the inquiry symbols
        should be downgraded. All inquiry symbols have value of 0.0"""
        inquiry = ['A', 'B', 'C', 'D', 'E']

        expected = [
            0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
            0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
            0.95, 0.95, 0.95, 0.95
        ]
        results = self.model.compute_likelihood_ratio(
            data=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            inquiry=inquiry,
            symbol_set=self.symbol_set)

        self.assertTrue(np.allclose(results, expected))


if __name__ == '__main__':
    unittest.main()
