"""Tests for session-related functionality."""

import unittest
from bcipy.tasks.session_data import Session, Inquiry, EvidenceType


def sample_stim_seq(include_evidence: bool = False):
    stim_seq = Inquiry(
        stimuli=["+", "I", "D", "H", "G", "F", "<", "E", "B", "C", "A"],
        timing=[0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
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
        target_text="HELLO_WORLD",
        next_display_state="H")

    if include_evidence:
        stim_seq.evidences = {
            EvidenceType.LM: [
                0.03518518518518518, 0.03518518518518518, 0.03518518518518518,
                0.03518518518518518, 0.03518518518518518, 0.03518518518518518,
                0.03518518518518518, 0.03518518518518518, 0.03518518518518518,
                0.03518518518518518, 0.03518518518518518, 0.03518518518518518,
                0.03518518518518518, 0.03518518518518518, 0.03518518518518518,
                0.03518518518518518, 0.03518518518518518, 0.03518518518518518,
                0.03518518518518518, 0.03518518518518518, 0.03518518518518518,
                0.03518518518518518, 0.03518518518518518, 0.03518518518518518,
                0.03518518518518518, 0.03518518518518518, 0.05,
                0.03518518518518518
            ],
            EvidenceType.ERP: [
                1.0771572587661082, 0.9567667980052755, 0.9447790096182402,
                0.9557979187496592, 0.9639921426239895, 1.0149791038166587,
                0.9332784168303235, 1.0020770058735426, 1.0143856794734767,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0075605462487651, 1.0
            ]
        }
        stim_seq.likelihood = [
            0.03806880657023323, 0.03381397643627917, 0.03339030496807247,
            0.03377973438232484, 0.03406933399382669, 0.03587131114010814,
            0.03298385192816283, 0.035415326215948305, 0.035850338383947605,
            0.03534192083878387, 0.03534192083878387, 0.03534192083878387,
            0.03534192083878387, 0.03534192083878387, 0.03534192083878387,
            0.03534192083878387, 0.03534192083878387, 0.03534192083878387,
            0.03534192083878387, 0.03534192083878387, 0.03534192083878387,
            0.03534192083878387, 0.03534192083878387, 0.03534192083878387,
            0.03534192083878387, 0.03534192083878387, 0.05060244088298704,
            0.03534192083878387
        ]
    return stim_seq


class TestSessionData(unittest.TestCase):
    """Tests for session data."""

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
            'stimuli', 'timing', 'triggers', 'target_info', 'target_letter',
            'current_text', 'target_text', 'next_display_state'
        ]

        for key in expected_keys:
            self.assertTrue(key in serialized)
            self.assertEqual(serialized[key], stim_seq.__getattribute__(key))

        for key in ['lm_evidence', 'erp_evidence', 'likelihood']:
            self.assertFalse(key in serialized)

    def test_stim_sequence_deserialization(self):
        """Test that a stim sequence can be deserialized from a dict."""
        stim_seq = sample_stim_seq(include_evidence=True)
        serialized = stim_seq.as_dict()
        self.assertTrue('lm_evidence' in serialized.keys())
        self.assertEquals(serialized['lm_evidence'], stim_seq.evidences[EvidenceType.LM])

        deserialized = Inquiry.from_dict(serialized)

        self.assertEquals(stim_seq.stimuli, deserialized.stimuli)
        self.assertEquals(stim_seq.timing, deserialized.timing)
        self.assertEquals(stim_seq.triggers, deserialized.triggers)
        self.assertEquals(stim_seq.target_info, deserialized.target_info)
        self.assertEquals(stim_seq.target_letter, deserialized.target_letter)
        self.assertEquals(stim_seq.current_text, deserialized.current_text)
        self.assertEquals(stim_seq.target_text, deserialized.target_text)
        self.assertEquals(stim_seq.next_display_state,
                          deserialized.next_display_state)
        self.assertEquals(stim_seq.evidences[EvidenceType.LM], deserialized.evidences[EvidenceType.LM])

    def test_stim_sequence_evidence(self):
        """Test simplified evidence view"""
        alp = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '<', '_'
        ]

        stim_seq = sample_stim_seq(include_evidence=True)
        evidence = stim_seq.stim_evidence(alp, n_most_likely=5)
        self.assertEqual(len(evidence['most_likely']), 5)
        self.assertAlmostEqual(evidence['most_likely']['<'], 0.05, places=2)

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

    def test_session_deserialization(self):
        """Test that a Session can be deserialized"""
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

        deserialized = Session.from_dict(serialized)
        self.assertEqual(deserialized.total_time_spent,
                         session.total_time_spent)
        self.assertEqual(deserialized.total_number_series,
                         session.total_number_series)
        self.assertEqual(deserialized.last_series()[-1].next_display_state,
                         'HE')

        first_stim_seq = deserialized.series[0][0]
        self.assertEqual(
            first_stim_seq.stimuli,
            ["+", "I", "D", "H", "G", "F", "<", "E", "B", "C", "A"])
