import unittest
from bcipy.helpers.task import SPACE_CHAR
import numpy as np
import math
import string
from bcipy.tasks.rsvp.main_frame import DecisionMaker, EvidenceFusion


class TestDecisionMaker(unittest.TestCase):
    """Test for decision maker class """

    def setUp(self):
        """Set up decision maker object for testing """
        self.decision_maker = DecisionMaker(
            1, 3, state='',
            alphabet=list(string.ascii_uppercase)
            + ['<'] + [SPACE_CHAR],
            is_txt_stim=True,
            stimuli_timing=[1, .2],
            seq_constants=None)

        self.evidence_fusion = EvidenceFusion(list_name_evidence = ['A','B'], len_dist = 2)

    def tearDown(self):
        """Reset decision maker and evidence fusion at the
        end of each test. """
        self.decision_maker.reset()
        self.evidence_fusion.reset_history()

    def test_evidence_fusion_init(self):
        self.assertEqual(self.evidence_fusion.evidence_history,{'A': [], 'B': []})
        self.assertEqual(self.evidence_fusion.likelihood[0],[0.5])
        self.assertEqual(self.evidence_fusion.likelihood[1],[0.5])

    def test_reset_history(self):
        self.evidence_fusion.reset_history()
        self.assertEqual(self.evidence_fusion.evidence_history,{'A': [], 'B': []})
        self.assertEqual(self.evidence_fusion.likelihood[0],[0.5])
        self.assertEqual(self.evidence_fusion.likelihood[1],[0.5])

    def test_update_and_fuse_with_float_evidence(self):
        dict_evidence = {'A': [0.5], 'B': [0.5]}
        self.evidence_fusion.update_and_fuse(dict_evidence)
        self.assertEqual(self.evidence_fusion.evidence_history['A'][0],dict_evidence['A'])
        self.assertEqual(self.evidence_fusion.evidence_history['B'][0],dict_evidence['B'])        
        self.assertEqual(self.evidence_fusion.likelihood[0],[[0.5]])
        self.assertEqual(self.evidence_fusion.likelihood[1],[[0.5]])
    
    def test_update_and_fuse_with_inf_evidence(self):
        dict_evidence = {'A': [0.5], 'B': [math.inf]}
        self.evidence_fusion.update_and_fuse(dict_evidence)
        self.assertEqual(self.evidence_fusion.evidence_history['A'][0],dict_evidence['A'])
        self.assertEqual(self.evidence_fusion.evidence_history['B'][0],dict_evidence['B']) 
        self.assertEqual(self.evidence_fusion.likelihood[0],[1.0])
        self.assertEqual(self.evidence_fusion.likelihood[1],[0.0])
        
    def test_save_history(self):
        history = self.evidence_fusion.save_history()
        self.assertEqual(0,history)

    def test_decision_maker_init(self):
        """Test initialization"""
        self.assertEqual(self.decision_maker.min_num_seq, 1)
        self.assertEqual(self.decision_maker.max_num_seq, 3)
        self.assertEqual(self.decision_maker.state, '')
        self.assertEqual(self.decision_maker.displayed_state, '')

    def test_decide_without_commit(self):
        """
        Test decide method with case of no commit
        using a fake probability distribution
        """
        probability_distribution = np.ones(len(
            self.decision_maker.alphabet)) / 8
        decision, chosen_stimuli = self.decision_maker.decide(
            probability_distribution)
        self.assertTrue(np.all(
                        self.decision_maker.list_epoch[-1]['list_distribution'][-1]
                        == probability_distribution))
        self.assertFalse(decision)
        self.decision_maker.do_epoch()
        self.assertEqual(self.decision_maker.sequence_counter,0)

    def test_decide_with_commit(self):
        """Test decide method with case of commit"""
        probability_distribution = np.ones(len(
            self.decision_maker.alphabet))
        self.decision_maker.sequence_counter = self.decision_maker.min_num_seq
        decision, chosen_stimuli = self.decision_maker.decide(
            probability_distribution)
        self.assertTrue(decision)
        self.assertEqual(chosen_stimuli, None)

    def test_update_with_letter(self):
        """Test update method with letter being the new state"""
        old_displayed_state = self.decision_maker.displayed_state
        old_state = self.decision_maker.state
        new_state = 'E'
        self.decision_maker.update(state=new_state)
        self.assertEqual(self.decision_maker.state, old_state + new_state)
        self.assertEqual(self.decision_maker.displayed_state,
                         old_displayed_state + 'E')

    def test_update_with_backspace(self):
        """Test update method with backspace being the new state"""
        old_displayed_state = self.decision_maker.displayed_state
        old_state = self.decision_maker.state
        new_state = '<'
        self.decision_maker.update(state=new_state)
        self.assertEqual(self.decision_maker.state, old_state + new_state)
        self.assertEqual(self.decision_maker.displayed_state,
                         old_displayed_state[0:-1])
        self.assertLess(len(self.decision_maker.displayed_state),
                        len(self.decision_maker.state))

    def test_reset(self):
        """Test reset of decision maker state"""
        self.decision_maker.reset()
        self.assertEqual(self.decision_maker.state, '')
        self.assertEqual(self.decision_maker.displayed_state, '')
        self.assertEqual(self.decision_maker.time, 0)
        self.assertEqual(self.decision_maker.sequence_counter, 0)

    def test_form_display_state(self):
        """Test form display state method with a dummy state"""
        self.decision_maker.update(state='ABC<.E')
        self.decision_maker.form_display_state(self.decision_maker.state)
        self.assertEqual(self.decision_maker.displayed_state, 'ABE')
        self.decision_maker.reset()

    def test_do_epoch(self):
        """Test do_epoch method"""
        probability_distribution = np.ones(
            len(self.decision_maker.alphabet)) / 8
        decision, chosen_stimuli = self.decision_maker.decide(
            probability_distribution)
        self.decision_maker.do_epoch()
        self.assertEqual(self.decision_maker.sequence_counter, 0)


    def test_decide_state_update(self):
        """Tests decide state update method"""
        probability_distribution = np.ones(len(self.decision_maker.alphabet))
        self.decision_maker.list_epoch[-1]['list_distribution'].append(probability_distribution)
        decision = self.decision_maker.decide_state_update()
        expected = 'A' # expect to commit to first letter in sequence, due to uniform probability
        self.assertEqual(decision,'A')

    def test_schedule_sequence(self):
        """Test sequence scheduling. Should return new stimuli list, at random."""
        probability_distribution = np.ones(len(self.decision_maker.alphabet))
        old_counter = self.decision_maker.sequence_counter
        self.decision_maker.list_epoch[-1]['list_distribution'].append(probability_distribution)
        stimuli = self.decision_maker.schedule_sequence()
        self.assertEqual(self.decision_maker.state,'.')
        self.assertEqual(stimuli[0],self.decision_maker.list_epoch[-1]['list_sti'][-1])
        self.assertLess(old_counter,self.decision_maker.sequence_counter)

    def test_prepare_stimuli(self):
        """Test that stimuli are prepared as expected"""
        probability_distribution = np.ones(len(self.decision_maker.alphabet))
        self.decision_maker.list_epoch[-1]['list_distribution'].append(probability_distribution)
        stimuli = self.decision_maker.prepare_stimuli()
        self.assertEqual(11,len(stimuli[0][0]))
        for i in range(1,len(stimuli[0][0])):
            self.assertIn(stimuli[0][0][i],self.decision_maker.alphabet)
        self.assertEqual(stimuli[1][0][0:2],self.decision_maker.stimuli_timing)
