"""Tests for Copy Phrase DecisionMaker"""

import unittest
import numpy as np

import bcipy.tasks.rsvp.main_frame as mf
from bcipy.tasks.rsvp.stopping_criteria import CriteriaEvaluator, \
    MaxIterationsCriteria, MinIterationsCriteria, ProbThresholdCriteria
from bcipy.tasks.rsvp.query_mechanisms import NBestStimuliAgent


class TestDecisionMaker(unittest.TestCase):
    """Tests for DecisionMaker"""

    def test_decision_maker(self):
        """Test default behavior"""
        alphabet = ['a', 'b', 'c', 'd']
        stopping_criteria = CriteriaEvaluator(
            continue_criteria=[MinIterationsCriteria(min_num_inq=1)],
            commit_criteria=[MaxIterationsCriteria(max_num_inq=10),
                             ProbThresholdCriteria(threshold=0.8)])

        stimuli_agent = NBestStimuliAgent(alphabet=alphabet,
                                          len_query=2)

        decision_maker = mf.DecisionMaker(
            stimuli_agent=stimuli_agent,
            stopping_evaluator=stopping_criteria,
            state='',
            alphabet=['a', 'b', 'c', 'd'])

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

    def test_min_inquiries(self):
        """Test min inquiries criteria."""

        alphabet = ['a', 'b', 'c', 'd']
        stopping_criteria = CriteriaEvaluator(
            continue_criteria=[MinIterationsCriteria(min_num_inq=2)],
            commit_criteria=[MaxIterationsCriteria(max_num_inq=10),
                             ProbThresholdCriteria(threshold=0.8)])

        stimuli_agent = NBestStimuliAgent(alphabet=alphabet,
                                          len_query=2)

        decision_maker = mf.DecisionMaker(
            stimuli_agent=stimuli_agent,
            stopping_evaluator=stopping_criteria,
            state='',
            alphabet=['a', 'b', 'c', 'd'])

        # Initialize with initial (language model) probabilities for each letter.
        lm_prior = np.array([0.1, 0.1, 0.1, 0.1])
        decision_maker.decide(lm_prior)

        likelihood = np.array([0.05, 0.05, 0.09, 0.81])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertFalse(decision, "Not enough inquiries presented")

        likelihood = np.array([0.05, 0.05, 0.09, 0.81])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertTrue(decision)
        self.assertEqual('d', decision_maker.displayed_state)

    def test_max_inquiries(self):
        """Test max inquiries criteria."""

        alphabet = ['a', 'b', 'c', 'd']
        stopping_criteria = CriteriaEvaluator(
            continue_criteria=[MinIterationsCriteria(min_num_inq=1)],
            commit_criteria=[MaxIterationsCriteria(max_num_inq=3),
                             ProbThresholdCriteria(threshold=0.8)])

        stimuli_agent = NBestStimuliAgent(alphabet=alphabet,
                                          len_query=2)

        decision_maker = mf.DecisionMaker(
            stimuli_agent=stimuli_agent,
            stopping_evaluator=stopping_criteria,
            state='',
            alphabet=['a', 'b', 'c', 'd'])

        # Initialize with initial (language model) probabilities for each letter.
        lm_prior = np.array([0.1, 0.1, 0.1, 0.1])
        decision_maker.decide(lm_prior)

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
            "Should make a decision after max inquiries are reached even if max value is below threshold"
        )
        self.assertEqual('c', decision_maker.displayed_state)

    def test_displayed_state(self):
        """Test displayed state"""

        alphabet = ['a', 'b', 'c', 'd']
        stopping_criteria = CriteriaEvaluator(
            continue_criteria=[MinIterationsCriteria(min_num_inq=1)],
            commit_criteria=[MaxIterationsCriteria(max_num_inq=10),
                             ProbThresholdCriteria(threshold=0.8)])

        stimuli_agent = NBestStimuliAgent(alphabet=alphabet,
                                          len_query=2)

        decision_maker = mf.DecisionMaker(
            stimuli_agent=stimuli_agent,
            stopping_evaluator=stopping_criteria,
            state='ab',
            alphabet=['a', 'b', 'c', 'd'])

        # Initialize with initial (language model) probabilities for each letter.
        lm_prior = np.array([0.1, 0.1, 0.1, 0.1])
        decision_maker.decide(lm_prior)

        likelihood = np.array([0.025, 0.025, 0.9, 0.05])
        _decision, _arg = decision_maker.decide(likelihood)
        self.assertEqual('abc', decision_maker.displayed_state)

    def test_decision_maker_threshold(self):
        """Threshold should be configurable"""
        alphabet = ['a', 'b', 'c', 'd']
        stopping_criteria = CriteriaEvaluator(
            continue_criteria=[MinIterationsCriteria(min_num_inq=1)],
            commit_criteria=[MaxIterationsCriteria(max_num_inq=10),
                             ProbThresholdCriteria(threshold=0.5)])

        stimuli_agent = NBestStimuliAgent(alphabet=alphabet,
                                          len_query=2)

        decision_maker = mf.DecisionMaker(
            stimuli_agent=stimuli_agent,
            stopping_evaluator=stopping_criteria,
            state='',
            alphabet=['a', 'b', 'c', 'd'])
        # Initialize with initial (language model) probabilities for each letter.
        lm_prior = np.array([0.1, 0.1, 0.1, 0.1])
        decision_maker.decide(lm_prior)

        likelihood = np.array([0.2, 0.6, 0.1, 0.1])
        decision, _arg = decision_maker.decide(likelihood)
        self.assertTrue(decision, "Value should be above configured threshold")
        self.assertEqual('b', decision_maker.displayed_state)
