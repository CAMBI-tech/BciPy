"""Tests for Copy Phrase DecisionMaker"""

import unittest
import numpy as np
from mockito import any, mock, unstub, when

import bcipy.tasks.rsvp.main_frame as mf
from bcipy.helpers.load import load_json_parameters


class TestDecisionMaker(unittest.TestCase):
    """Tests for DecisionMaker"""

    def test_decision_maker(self):
        """Test default behavior"""
        decision_maker = mf.DecisionMaker(
            min_num_seq=1,
            max_num_seq=10,
            state='',
            alphabet=['a', 'b', 'c', 'd'],
            decision_threshold=0.8)

        likelihood = np.array([0.2, 0.2, 0.2, 0.4])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertFalse(decision)
        self.assertEqual('.', decision_maker.state)
        self.assertEqual('', decision_maker.displayed_state)

        likelihood = np.array([0.05, 0.05, 0.1, 0.8])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertFalse(decision, "Must exceed threshold")
        self.assertEqual('..', decision_maker.state)

        likelihood = np.array([0.05, 0.05, 0.09, 0.81])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertTrue(decision)
        self.assertEqual('..d', decision_maker.state)
        self.assertEqual('d', decision_maker.displayed_state)

    def test_min_sequences(self):
        """Test min sequences criteria."""

        decision_maker = mf.DecisionMaker(
            min_num_seq=2,
            max_num_seq=10,
            state='',
            alphabet=['a', 'b', 'c', 'd'],
            decision_threshold=0.8)

        likelihood = np.array([0.05, 0.05, 0.09, 0.81])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertFalse(decision, "Not enough sequences presented")

        likelihood = np.array([0.05, 0.05, 0.09, 0.81])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertTrue(decision)
        self.assertEqual('.d', decision_maker.state)

    def test_max_sequences(self):
        """Test max sequences criteria."""

        decision_maker = mf.DecisionMaker(
            min_num_seq=1,
            max_num_seq=3,
            state='',
            alphabet=['a', 'b', 'c', 'd'],
            decision_threshold=0.8)

        likelihood = np.array([0.2, 0.2, 0.4, 0.2])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertFalse(decision)

        likelihood = np.array([0.2, 0.2, 0.4, 0.2])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertFalse(decision)

        likelihood = np.array([0.2, 0.2, 0.4, 0.2])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertTrue(
            decision,
            "Should make a decision after max sequences are reached even if max value is below threshold"
        )
        self.assertEqual('c', decision_maker.displayed_state)

    def test_displayed_state(self):
        """Test displayed state"""

        decision_maker = mf.DecisionMaker(
            min_num_seq=1,
            max_num_seq=10,
            state='ab',
            alphabet=['a', 'b', 'c', 'd'],
            decision_threshold=0.8)

        likelihood = np.array([0.025, 0.025, 0.9, 0.05])
        _decision, _arg = decision_maker.decide(likelihood)
        self.assertEqual('abc', decision_maker.displayed_state)

    def test_decision_maker_threshold(self):
        """Threshold should be configurable"""
        decision_maker = mf.DecisionMaker(
            min_num_seq=1,
            max_num_seq=10,
            state='',
            alphabet=['a', 'b', 'c', 'd'],
            decision_threshold=0.5)

        likelihood = np.array([0.2, 0.6, 0.1, 0.1])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertTrue(decision, "Value should be above configured threshold")
        self.assertEqual('b', decision_maker.displayed_state)