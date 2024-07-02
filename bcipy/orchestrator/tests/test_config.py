import unittest
from bcipy.orchestrator.config import parse_sequence, serialize_sequence, validate_sequence_string
from bcipy.task import TaskType
from bcipy.orchestrator.actions import OfflineAnalysisAction
from bcipy.task.paradigm.rsvp.calibration.calibration import RSVPCalibrationTask

from bcipy.orchestrator.actions import OfflineAnalysisAction
from bcipy.task.paradigm.rsvp.copy_phrase import RSVPCopyPhraseTask


class TestTaskProtocolProcessing(unittest.TestCase):

    def test_parses_one_task(self) -> None:
        sequence = 'RSVP Calibration'
        parsed = parse_sequence(sequence)
        assert len(parsed) == 1
        assert parsed[0] is RSVPCalibrationTask

    def test_parses_with_task_name(self) -> None:
        actions = OfflineAnalysisAction.name
        parsed = parse_sequence(actions)
        assert len(parsed) == 1
        assert parsed[0] is OfflineAnalysisAction

    def test_parses_multiple_tasks(self) -> None:
        actions = 'RSVP Calibration -> RSVP Copy Phrase Task'
        parsed = parse_sequence(actions)
        assert len(parsed) == 2
        assert parsed[0] is RSVPCalibrationTask
        assert parsed[1] is RSVPCopyPhraseTask

    def test_parses_actions_and_tasks(self) -> None:
        sequence = 'RSVP Calibration -> Offline Analysis Action -> RSVP Copy Phrase Task'
        parsed = parse_sequence(sequence)
        assert len(parsed) == 3
        assert parsed[0] is RSVPCalibrationTask
        assert parsed[1] is OfflineAnalysisAction
        assert parsed[2] is RSVPCopyPhraseTask

    def test_throws_exception_on_invalid_task(self) -> None:
        actions = 'RSVP Calibration -> does not exist'
        with self.assertRaises(ValueError):
            parse_sequence(actions)

    def test_throws_exception_on_invalid_string(self) -> None:
        actions = 'thisstringisbad'
        with self.assertRaises(ValueError):
            parse_sequence(actions)

    def test_validates_valid_action_string(self) -> None:
        actions = 'RSVP Calibration -> RSVP Copy Phrase Task'
        validate_sequence_string(actions)

    def test_throws_exception_on_invalid_action_string(self) -> None:
        actions = 'RSVP Calibration -> RSVP Copy Phrase -> does not exist'
        with self.assertRaises(ValueError):
            validate_sequence_string(actions)

    def test_serializes_one_task(self) -> None:
        actions = [RSVPCalibrationTask]
        serialized = serialize_sequence(actions)
        assert serialized == RSVPCalibrationTask.name

    def test_serializes_multiple_tasks(self) -> None:
        sequence = [RSVPCalibrationTask, OfflineAnalysisAction, RSVPCopyPhraseTask]
        serialized = serialize_sequence(sequence)
        assert serialized == f'{RSVPCalibrationTask.name} -> {OfflineAnalysisAction.name} -> {RSVPCopyPhraseTask.name}'