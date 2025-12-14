"""Tests for functions in the evidence module."""

import unittest
from unittest.mock import Mock

from bcipy.acquisition.multimodal import ContentType
from bcipy.task.control.evidence import (EEGEvaluator, SwitchEvaluator,
                                         find_matching_evaluator,
                                         get_evaluator)
from bcipy.task.data import EvidenceType
from bcipy.task.exceptions import MissingEvidenceEvaluator


class TestEvidence(unittest.TestCase):
    """Test evidence functions"""

    def test_get_evaluator(self):
        """Test getting an evaluator"""
        self.assertEqual(EEGEvaluator, get_evaluator(ContentType.EEG))
        self.assertEqual(EEGEvaluator,
                         get_evaluator(ContentType.EEG, EvidenceType.ERP))
        self.assertEqual(SwitchEvaluator, get_evaluator(ContentType.MARKERS))

    def test_missing_evaluator(self):
        """Test error condition when evaluator is missing"""

        with self.assertRaises(MissingEvidenceEvaluator):
            get_evaluator(ContentType.EEG, EvidenceType.EYE)

    def test_find_matching_evaluator(self):
        """Test finding an evaluator for a given SignalModel"""
        device_mock = Mock()
        device_mock.content_type = 'EEG'
        signal_model_mock = Mock()
        meta_mock = Mock()
        meta_mock.device_spec = device_mock
        signal_model_mock.metadata = meta_mock

        self.assertEqual(EEGEvaluator,
                         find_matching_evaluator(signal_model_mock))

    def test_match_with_bad_evidence_type(self):
        """Test matching an evaulator when EvidenceType is not found"""
        device_mock = Mock()
        device_mock.content_type = 'EEG'
        signal_model_mock = Mock()
        meta_mock = Mock()
        meta_mock.device_spec = device_mock
        meta_mock.evidence_type = 'NOT_AN_EVIDENCE_TYPE'
        signal_model_mock.metadata = meta_mock

        self.assertEqual(EEGEvaluator,
                         find_matching_evaluator(signal_model_mock),
                         "should ignore the evidence type")

    def test_match_should_use_evidence(self):
        """Test matching an evaluator should use the evidence if provided."""
        device_mock = Mock()
        device_mock.content_type = 'EEG'
        signal_model_mock = Mock()
        meta_mock = Mock()
        meta_mock.device_spec = device_mock
        meta_mock.evidence_type = 'EYE'
        signal_model_mock.metadata = meta_mock

        with self.assertRaises(MissingEvidenceEvaluator):
            find_matching_evaluator(signal_model_mock)
