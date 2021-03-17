"""Tests for session-related functionality."""

import unittest
from bcipy.tasks.session_data import Session, StimSequence


def sample_stim_seq():
    return StimSequence(
        stimuli=["+", "I", "D", "H", "G", "F", "<", "E", "B", "C", "A"],
        eeg_len=17,
        timing_sti=[0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        triggers=[["+", 0.0], ["I", 0.4688645350106526],
                  ["D", 0.737725474027684], ["H", 1.0069859480136074],
                  ["G", 1.2752754990069661], ["F", 1.5437563360028435],
                  ["<", 1.8121200330206193], ["E", 2.080774770001881],
                  ["B", 2.3487972170114517], ["C", 2.6170289120054804],
                  ["A", 2.8860043640015647]],
        target_info=[
            "nontarget", "nontarget", "nontarget", "nontarget", "nontarget",
            "nontarget", "nontarget", "nontarget", "nontarget", "nontarget",
            "nontarget"
        ],
        target_letter="H",
        current_text="",
        copy_phrase="HELLO_WORLD",
        next_display_state="H")


class TestSession(unittest.TestCase):
    """Tests for session."""

    def setUp(self):
        """Override; set up the needed path for load functions."""
        pass

    def tearDown(self):
        """Override"""
        pass

    def test_stim_sequence(self):
        """Test stim sequence can be created and serialized to dict."""
        stim_seq = sample_stim_seq()
        self.assertEqual([], stim_seq.lm_evidence)
        self.assertEqual([], stim_seq.eeg_evidence)

        serialized = stim_seq.as_dict()
        self.assertEqual(dict, type(serialized))

        expected_keys = [
            'stimuli', 'eeg_len', 'timing_sti', 'triggers', 'target_info',
            'target_letter', 'current_text', 'copy_phrase',
            'next_display_state'
        ]

        for key in expected_keys:
            self.assertTrue(key in serialized)
            self.assertEqual(serialized[key], stim_seq.__getattribute__(key))

        for key in ['lm_evidence', 'eeg_evidence', 'likelihood']:
            self.assertFalse(key in serialized)

    def test_empty_session(self):
        """Test initial session creation"""
        session = Session(save_location=".")
        serialized = session.as_dict()

        self.assertEqual(0, serialized['total_time_spent'])
        self.assertEqual({}, serialized['series'])
        self.assertEqual(0, serialized['total_number_series'])

    def test_session(self):
        """Test session functionality"""
        session = Session(save_location=".")
        session.add_sequence(sample_stim_seq())
        session.add_sequence(sample_stim_seq())

        stim_seq = sample_stim_seq()
        stim_seq.target_letter = "E",
        stim_seq.current_text = "H",
        stim_seq.next_display_state = "HE"

        session.add_sequence(stim_seq, new_series=True)

        session.total_time_spent = 0.01

        serialized = session.as_dict()

        self.assertEqual(dict, type(serialized))
        self.assertEqual(2, serialized['total_number_series'])
        self.assertTrue('1' in serialized['series'].keys())
        self.assertTrue('2' in serialized['series'].keys())

        self.assertEqual(2, len(serialized['series']['1'].keys()))
        self.assertEqual(1, len(serialized['series']['2'].keys()))

        self.assertTrue('0' in serialized['series']['1'].keys())
        self.assertTrue('1' in serialized['series']['1'].keys())

    def test_session_add_series(self):
        """Test session functionality for adding a series"""
        session = Session(save_location=".")

        session.add_series()
        self.assertEqual(0, session.total_number_series)

        session.add_sequence(sample_stim_seq())
        self.assertEqual(1, session.total_number_series)

        session.add_series()
        session.add_sequence(sample_stim_seq())
        self.assertEqual(2, session.total_number_series)
