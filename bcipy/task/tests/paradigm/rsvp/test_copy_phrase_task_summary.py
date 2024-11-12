"""Tests for Copy Phrase metrics."""
import unittest
from typing import List

from mock import patch

from bcipy.core.triggers import Trigger
from bcipy.task.data import EvidenceType, Inquiry, Session
from bcipy.task.paradigm.rsvp.copy_phrase import TaskSummary

SYMBOL_SET = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '<', '_'
]


def mock_inquiry(target_letter="",
                 current_text="",
                 selection="",
                 next_display_state="",
                 eeg_evidence: bool = False,
                 btn_evidence: bool = False) -> Inquiry:
    """Generate a mock Inquiry object"""
    alp_size = 28
    inquiry = Inquiry(stimuli=[],
                      timing=[],
                      triggers=[],
                      target_info=[],
                      target_letter=target_letter,
                      current_text=current_text,
                      selection=selection,
                      next_display_state=next_display_state)
    if eeg_evidence:
        inquiry.evidences[EvidenceType.ERP] = [1.0] * alp_size
    if btn_evidence:
        inquiry.evidences[EvidenceType.BTN] = [0.01] * alp_size
    return inquiry


def session_no_preview() -> Session:
    """Mock session"""
    session = Session(".", symbol_set=SYMBOL_SET)

    seq1 = [{
        'target_letter': "E",
        'current_text': "H",
        'selection': "",
        'next_display_state': "H"
    }, {
        'target_letter': "E",
        'current_text': "H",
        'selection': "E",
        'next_display_state': "HE"
    }]
    seq2 = [{
        'target_letter': "L",
        'current_text': "HE",
        'selection': "",
        'next_display_state': "HE"
    }, {
        'target_letter': "L",
        'current_text': "HE",
        'selection': "M",
        'next_display_state': "HEM"
    }]
    seq3 = [{
        'target_letter': "<",
        'current_text': "HEM",
        'selection': "<",
        'next_display_state': "HE"
    }]
    for seq in [seq1, seq2, seq3]:
        for i, fields in enumerate(seq):
            session.add_sequence(mock_inquiry(**fields), new_series=(i == 0))
    session.total_time_spent = 68.21
    return session


def session_btn() -> Session:
    """Mock session with button press evidence."""
    session = Session(".", symbol_set=SYMBOL_SET)
    seq1 = [{
        'selection': "",
        'btn_evidence': True,
        'eeg_evidence': False,
    }, {
        'selection': "E",
        'btn_evidence': True,
        'eeg_evidence': True,
    }]
    seq2 = [{
        'selection': "",
        'btn_evidence': True,
        'eeg_evidence': True,
    }, {
        'selection': "M",
        'btn_evidence': True,
        'eeg_evidence': True,
    }]
    seq3 = [{
        'selection': "<",
        'btn_evidence': True,
        'eeg_evidence': True,
    }]
    for seq in [seq1, seq2, seq3]:
        for i, fields in enumerate(seq):
            session.add_sequence(mock_inquiry(**fields), new_series=(i == 0))

    session.total_time_spent = 53.82
    return session


def sample_triggers() -> List[Trigger]:
    """Sample trigger data for computing switch response"""
    data = [['starting_offset', 'offset', '771032.89'],
            ['inquiry_preview', 'preview', '771033.96'],
            ['bcipy_key_press_space', 'event', '771034.69'],
            ['+', 'fixation', '771035.73'], ['<', 'nontarget', '771036.23'],
            ['A', 'nontarget', '771036.48'], ['C', 'nontarget', '771036.73'],
            ['E', 'nontarget', '771036.98'],
            ['inquiry_preview', 'preview', '771040.29'],
            ['bcipy_key_press_space', 'event', '771041.00'],
            ['+', 'fixation', '771042.03'], ['H', 'nontarget', '771042.53'],
            ['G', 'nontarget', '771042.78'], ['A', 'nontarget', '771043.03'],
            ['I', 'nontarget', '771043.29'],
            ['inquiry_preview', 'preview', '771053.17'],
            ['inquiry_preview', 'preview', '771060.69'],
            ['bcipy_key_press_space', 'event', '771061.59'],
            ['+', 'fixation', '771062.62'], ['A', 'nontarget', '771063.12'],
            ['D', 'target', '771063.37'], ['G', 'nontarget', '771063.62'],
            ['C', 'nontarget', '771063.87']]
    return list(map(Trigger.from_list, data))


class TestCopyPhraseTaskSummary(unittest.TestCase):
    """Tests for summary metrics."""

    def test_session_stats(self):
        """Test summary statistics"""
        session = session_no_preview()
        params = {
            'preview_inquiry_progress_method': 0,
            'show_preview_inquiry': False
        }
        summary = TaskSummary(session).as_dict()

        self.assertEqual(summary['selections_correct'], 2)
        self.assertEqual(summary['selections_incorrect'], 1)
        self.assertEqual(summary['selections_correct_symbols'], 1)
        self.assertEqual(summary['switch_total'], 0)
        self.assertEqual(summary['switch_per_selection'], 0)
        self.assertEqual(summary['typing_accuracy'], 2 / 3)
        self.assertEqual(summary['correct_rate'],
                         2 / (session.total_time_spent / 60))
        self.assertEqual(summary['copy_rate'],
                         1 / (session.total_time_spent / 60))

    def test_switch_with_preview_only(self):
        """Test switch metrics when preview inquiry mode is set to preview
        only."""
        session = session_btn()
        summary = TaskSummary(session, show_preview=True,
                              preview_mode=0).as_dict()
        self.assertEqual(summary['switch_total'], 0)
        self.assertEqual(summary['switch_per_selection'], 0)

    def test_switch_with_btn_confirm(self):
        """Test calculations for switch metrics when preview inquiry mode is
        set with press switch to continue."""
        session = session_btn()
        summary = TaskSummary(session, show_preview=True,
                              preview_mode=1).as_dict()
        self.assertEqual(summary['switch_total'], 4)
        self.assertEqual(summary['switch_per_selection'], 4 / 3)

    def test_switch_with_btn_skip(self):
        """Test calculations for switch metrics when preview inquiry mode is
        set with press switch to skip."""
        session = session_btn()
        summary = TaskSummary(session, show_preview=True,
                              preview_mode=2).as_dict()
        self.assertEqual(summary['switch_total'], 1)
        self.assertEqual(summary['switch_per_selection'], 1 / 3)

    @patch('bcipy.task.paradigm.rsvp.copy_phrase.TriggerHandler.read_text_file')
    def test_switch_response_time(self, read_triggers):
        """Test that switch response is correctly calculated from triggers."""
        read_triggers.return_value = (sample_triggers(), 0.0)
        summary = TaskSummary(session=None, trigger_path='triggers.txt')
        self.assertEqual(summary.switch_response_time(), 0.7799999999891346)
