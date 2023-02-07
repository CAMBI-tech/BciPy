import unittest
import numpy as np
import math
from bcipy.task.control.handler import DecisionMaker, EvidenceFusion
from bcipy.task.control.criteria import CriteriaEvaluator, \
    MaxIterationsCriteria, MinIterationsCriteria, ProbThresholdCriteria
from bcipy.task.control.query import NBestStimuliAgent
from bcipy.language.main import alphabet


class TestDecisionMaker(unittest.TestCase):
    """Test for decision maker class """

    def setUp(self):
        """Set up decision maker object for testing """
        alp = alphabet()
        stopping_criteria = CriteriaEvaluator(
            continue_criteria=[MinIterationsCriteria(min_num_inq=1)],
            commit_criteria=[MaxIterationsCriteria(max_num_inq=10),
                             ProbThresholdCriteria(threshold=0.8)])

        stimuli_agent = NBestStimuliAgent(alphabet=alp,
                                          len_query=10)
        self.decision_maker = DecisionMaker(
            stimuli_agent=stimuli_agent,
            stopping_evaluator=stopping_criteria,
            state='HE',
            alphabet=alp,
            is_txt_stim=True,
            stimuli_timing=[1, .2],
            inq_constants=None)

        self.evidence_fusion = EvidenceFusion(list_name_evidence=['A', 'B'],
                                              len_dist=2)

    def tearDown(self):
        """Reset decision maker and evidence fusion at the
        end of each test. """
        self.decision_maker.reset()
        self.evidence_fusion.reset_history()

    def test_evidence_fusion_init(self):
        self.assertEqual(self.evidence_fusion.evidence_history,
                         {'A': [], 'B': []})
        self.assertEqual(self.evidence_fusion.likelihood[0], [0.5])
        self.assertEqual(self.evidence_fusion.likelihood[1], [0.5])

    def test_reset_history(self):
        self.evidence_fusion.reset_history()
        self.assertEqual(self.evidence_fusion.evidence_history,
                         {'A': [], 'B': []})
        self.assertEqual(self.evidence_fusion.likelihood[0], [0.5])
        self.assertEqual(self.evidence_fusion.likelihood[1], [0.5])

    def test_update_and_fuse_with_float_evidence(self):
        dict_evidence = {'A': [0.5], 'B': [0.5]}
        self.evidence_fusion.update_and_fuse(dict_evidence)
        self.assertEqual(self.evidence_fusion.evidence_history['A'][0],
                         dict_evidence['A'])
        self.assertEqual(self.evidence_fusion.evidence_history['B'][0],
                         dict_evidence['B'])
        self.assertEqual(self.evidence_fusion.likelihood[0], [[0.5]])
        self.assertEqual(self.evidence_fusion.likelihood[1], [[0.5]])

    def test_update_and_fuse_with_inf_evidence(self):
        dict_evidence = {'A': [0.5], 'B': [math.inf]}
        self.evidence_fusion.update_and_fuse(dict_evidence)
        self.assertEqual(self.evidence_fusion.evidence_history['A'][0],
                         dict_evidence['A'])
        self.assertEqual(self.evidence_fusion.evidence_history['B'][0],
                         dict_evidence['B'])
        self.assertEqual(self.evidence_fusion.likelihood[0], [1.0])
        self.assertEqual(self.evidence_fusion.likelihood[1], [0.0])

    def test_save_history(self):
        """This method is not implemented. It returns None."""
        history = self.evidence_fusion.save_history()
        self.assertEqual(None, history)

    def test_decision_maker_init(self):
        """Test initialization"""
        # TODO: Update that test part
        # self.assertEqual(self.decision_maker.min_num_inq, 1)
        # self.assertEqual(self.decision_maker.max_num_inq, 3)
        self.assertEqual(self.decision_maker.state, 'HE')
        self.assertEqual(self.decision_maker.displayed_state, 'HE')

    def test_update_with_letter(self):
        """Test update method with letter being the new state"""
        new_state = 'E'
        self.decision_maker.update(state=new_state)
        self.assertEqual(self.decision_maker.state, new_state)
        self.assertEqual(self.decision_maker.displayed_state, new_state)

    def test_update_with_backspace(self):
        """Test update method with backspace being the new state"""
        new_state = '<'
        self.decision_maker.update(state=new_state)
        self.assertEqual(self.decision_maker.state, new_state)
        self.assertEqual(self.decision_maker.displayed_state, '')

    def test_reset(self):
        """Test reset of decision maker state"""
        self.decision_maker.reset()
        self.assertEqual(self.decision_maker.state, '')
        self.assertEqual(self.decision_maker.displayed_state, '')
        self.assertEqual(self.decision_maker.time, 0)
        self.assertEqual(self.decision_maker.inquiry_counter, 0)

    def test_form_display_state(self):
        """Test form display state method with a dummy state"""
        self.decision_maker.update(state='ABC<.E')
        self.decision_maker.form_display_state(self.decision_maker.state)
        self.assertEqual(self.decision_maker.displayed_state, 'ABE')
        self.decision_maker.reset()

    def test_do_series(self):
        """Test do_series method"""
        probability_distribution = np.ones(
            len(self.decision_maker.alphabet)) / 8
        self.decision_maker.decide(probability_distribution)
        self.decision_maker.do_series()
        self.assertEqual(self.decision_maker.inquiry_counter, 0)

    def test_decide_state_update(self):
        """Tests decide state update method"""
        probability_distribution = np.ones(len(self.decision_maker.alphabet))
        self.decision_maker.list_series[-1]['list_distribution'].append(
            probability_distribution)
        decision = self.decision_maker.decide_state_update()
        # expect to commit to first letter in inquiry, due to uniform probability
        self.assertEqual(decision, 'A')

    def test_schedule_inquiry(self):
        """Test inquiry scheduling. Should return new stimuli list, at random."""
        probability_distribution = np.ones(len(self.decision_maker.alphabet))
        old_counter = self.decision_maker.inquiry_counter
        self.decision_maker.list_series[-1]['list_distribution'].append(
            probability_distribution)
        stimuli = self.decision_maker.schedule_inquiry()
        self.assertEqual(self.decision_maker.state, 'HE.')
        self.assertEqual(stimuli[0],
                         self.decision_maker.list_series[-1]['list_sti'][-1])
        self.assertLess(old_counter, self.decision_maker.inquiry_counter)

    def test_prepare_stimuli(self):
        """Test that stimuli are prepared as expected"""
        probability_distribution = np.ones(len(self.decision_maker.alphabet))
        self.decision_maker.list_series[-1]['list_distribution'].append(
            probability_distribution)
        stimuli = self.decision_maker.prepare_stimuli()
        self.assertEqual(self.decision_maker.stimuli_agent.len_query + 1,
                         len(stimuli[0][0]))
        for i in range(1, len(stimuli[0][0])):
            self.assertIn(stimuli[0][0][i], self.decision_maker.alphabet)
        self.assertEqual(stimuli[1][0][0:2], self.decision_maker.stimuli_timing)


class TestDecisionMakerOld(unittest.TestCase):
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

        decision_maker = DecisionMaker(
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

        decision_maker = DecisionMaker(
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

        decision_maker = DecisionMaker(
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

        decision_maker = DecisionMaker(
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

        decision_maker = DecisionMaker(
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


class TestEvidenceFusion(unittest.TestCase):
    """Tests for EvidenceFusion"""

    def test_fusion(self):
        len_alp = 4
        evidence_names = ['LM', 'ERP', 'FRP']
        num_inquiries = 10

        conjugator = EvidenceFusion(evidence_names, len_dist=len_alp)

        # generated random inquiries
        erp_evidence = [
            np.array([1.46762823, 0.40728661, 9.48721375, 11.47963171]),
            np.array([6.18696887, 8.66360132, 4.10906719, 10.40379058]),
            np.array([14.842917, 7.95145995, 17.68779128, 10.94195076]),
            np.array([2.1032388, 14.56945629, 9.9442332, 8.59807331]),
            np.array([17.2194632, 7.49341414, 10.7793301, 8.42595077]),
            np.array([12.78049307, 3.22652363, 0.55427707, 14.64100114]),
            np.array([7.23923751, 26.61319581, 7.72974217, 1.82107856]),
            np.array([6.35755791, 1.40185201, 5.94498732, 2.88180149]),
            np.array([13.72248648, 8.91011389, 13.76731832, 15.66952482]),
            np.array([16.64688501, 8.2827469, 1.75102074, 9.12089907])
        ]
        frp_evidence = [
            np.array([2.59452579, 18.37016033, 0.99017855, 7.75622695]),
            np.array([14.43263341, 10.97966831, 0.08324027, 7.39021412]),
            np.array([13.17879992, 6.62931003, 6.27445546, 7.72110125]),
            np.array([7.78663088, 17.96760446, 2.51192754, 6.26679419]),
            np.array([8.63589394, 10.97667028, 8.02164873, 26.98926272]),
            np.array([7.49884707, 12.56370517, 7.77508668, 1.45465084]),
            np.array([5.94854662, 5.13292461, 7.74513707, 3.78547598]),
            np.array([7.19402239, 0.32145887, 9.15348207, 1.86521141]),
            np.array([6.61159441, 0.14524912, 8.85456089, 2.65464896]),
            np.array([5.44631832, 3.94499887, 25.16119557, 9.28041577])
        ]
        lm_evidence = np.array([0.3198295, 0.24571754, 0.06543074, 0.36902222])

        expected_posterior = [
            9.16884720e-01, 2.38754536e-04, 4.33163904e-05, 8.28332090e-02
        ]

        # Single series

        # initialize with language model priors
        conjugator.update_and_fuse({'LM': lm_evidence})

        for idx in range(num_inquiries):
            conjugator.update_and_fuse({
                'ERP': erp_evidence[idx],
                'FRP': frp_evidence[idx]
            })

        self.assertEqual(1, len(conjugator.evidence_history['LM']))
        for i, val in enumerate(conjugator.evidence_history['LM'][0]):
            self.assertEqual(lm_evidence[i], val)

        for i, val in enumerate(conjugator.evidence_history['ERP']):
            self.assertSequenceEqual(list(erp_evidence[i]), list(val))

        for i, val in enumerate(conjugator.evidence_history['FRP']):
            self.assertSequenceEqual(list(frp_evidence[i]), list(val))

        for i, val in enumerate(conjugator.likelihood):
            self.assertAlmostEqual(expected_posterior[i], val)


if __name__ == '__main__':
    unittest.main()
