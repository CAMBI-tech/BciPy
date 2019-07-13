import unittest
from bcipy.helpers.task import SPACE_CHAR
import numpy as np
import string
from bcipy.tasks.rsvp.main_frame import DecisionMaker


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

    def test_init(self):
        """Test initialisation"""
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

    def test_decide_with_commit(self):
        """Test decide method with case of commit"""
        probability_distribution = np.ones(len(
            self.decision_maker.alphabet))
        self.decision_maker.sequence_counter = self.decision_maker.min_num_seq
        decision, chosen_stimuli = self.decision_maker.decide(
            probability_distribution)
        self.assertTrue(decision)
        self.assertEqual(chosen_stimuli, [])

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
