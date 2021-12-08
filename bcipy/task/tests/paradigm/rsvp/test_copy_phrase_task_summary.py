"""Tests for Copy Phrase metrics."""

import unittest
from bcipy.task.data import Session, Inquiry, EvidenceType
from bcipy.task.paradigm.rsvp.copy_phrase import TaskSummary


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
    session = Session(".")

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
    session = Session(".")
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


class TestCopyPhraseTaskSummary(unittest.TestCase):
    """Tests for summary metrics."""

    def test_session_stats(self):
        """Test summary statistics"""
        session = session_no_preview()
        params = {
            'preview_inquiry_progress_method': 0,
            'show_preview_inquiry': False
        }
        summary = TaskSummary(session, params).as_dict()

        self.assertEqual(summary['selections_correct'], 2)
        self.assertEqual(summary['selections_incorrect'], 1)
        self.assertEqual(summary['selections_correct_letters'], 1)
        self.assertEqual(summary['selections_correct_backspace'], 1)
        self.assertEqual(summary['switch_total'], 0)
        self.assertEqual(summary['switch_per_selection'], 0)
        self.assertEqual(summary['typing_accuracy'], 2 / 3)
        self.assertEqual(summary['correct_rate'], 2 / (session.total_time_spent / 60))
        self.assertEqual(summary['copy_rate'], 1 / (session.total_time_spent / 60))

    def test_switch_with_preview_only(self):
        """Test switch metrics when preview inquiry mode is set to preview
        only."""
        session = session_btn()
        params = {
            'preview_inquiry_progress_method': 0,
            'show_preview_inquiry': True
        }
        summary = TaskSummary(session, params).as_dict()
        self.assertEqual(summary['switch_total'], 0)
        self.assertEqual(summary['switch_per_selection'], 0)

    def test_switch_with_btn_confirm(self):
        """Test calculations for switch metrics when preview inquiry mode is
        set with press switch to continue."""
        session = session_btn()
        params = {
            'preview_inquiry_progress_method': 1,
            'show_preview_inquiry': True
        }
        summary = TaskSummary(session, params).as_dict()
        self.assertEqual(summary['switch_total'], 4)
        self.assertEqual(summary['switch_per_selection'], 4 / 3)

    def test_switch_with_btn_skip(self):
        """Test calculations for switch metrics when preview inquiry mode is
        set with press switch to skip."""
        session = session_btn()
        params = {
            'preview_inquiry_progress_method': 2,
            'show_preview_inquiry': True
        }
        summary = TaskSummary(session, params).as_dict()
        self.assertEqual(summary['switch_total'], 1)
        self.assertEqual(summary['switch_per_selection'], 1 / 3)
