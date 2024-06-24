import unittest
from bcipy.orchestrator.config import parse_sequence, serialize_sequence, validate_sequence_string
from bcipy.task import TaskType
from bcipy.orchestrator.actions import OfflineAnalysisAction


class TestTaskProtocolProcessing(unittest.TestCase):

    def test_parses_one_task(self) -> None:
        sequence = 'RSVP Calibration'
        parsed = parse_sequence(sequence)
        assert len(parsed) == 1
        assert parsed[0] == TaskType.RSVP_CALIBRATION

    def test_parses_one_action(self) -> None:
        actions = 'Offline Analysis'
        parsed = parse_sequence(actions)
        assert len(parsed) == 1
        assert parsed[0] is OfflineAnalysisAction

    def test_parses_multiple_tasks(self) -> None:
        actions = 'RSVP Calibration -> RSVP Copy Phrase'
        parsed = parse_sequence(actions)
        assert len(parsed) == 2
        assert parsed[0] == TaskType.RSVP_CALIBRATION
        assert parsed[1] == TaskType.RSVP_COPY_PHRASE

    def test_parses_actions_and_tasks(self) -> None:
        sequence = 'RSVP Calibration -> Offline Analysis -> RSVP Copy Phrase'
        parsed = parse_sequence(sequence)
        assert len(parsed) == 3
        assert parsed[0] == TaskType.RSVP_CALIBRATION
        assert parsed[1] is OfflineAnalysisAction
        assert parsed[2] == TaskType.RSVP_COPY_PHRASE

    def test_throws_exception_on_invalid_action(self) -> None:
        actions = 'RSVP Calibration -> does not exist'
        with self.assertRaises(ValueError):
            parse_sequence(actions)

    def test_throws_exception_on_invalid_task(self) -> None:
        actions = 'RSVP Calibration -> RSVP Copy Phrase -> does not exist'
        with self.assertRaises(ValueError):
            parse_sequence(actions)

    def test_validates_valid_action_string(self) -> None:
        actions = 'RSVP Calibration -> RSVP Copy Phrase'
        validate_sequence_string(actions)

    def test_throws_exception_on_invalid_action_string(self) -> None:
        actions = 'RSVP Calibration -> RSVP Copy Phrase -> does not exist'
        with self.assertRaises(ValueError):
            validate_sequence_string(actions)

    def test_serializes_one_task(self) -> None:
        actions = [TaskType.RSVP_CALIBRATION]
        serialized = serialize_sequence(actions)
        assert serialized == 'RSVP Calibration'

    def test_serializes_one_action(self) -> None:
        sequence = [OfflineAnalysisAction]
        serialized = serialize_sequence(sequence)
        assert serialized == 'Offline Analysis'

    def test_serializes_actions_and_tasks(self) -> None:
        sequence = [TaskType.RSVP_CALIBRATION, OfflineAnalysisAction, TaskType.RSVP_COPY_PHRASE]
        serialized = serialize_sequence(sequence)
        assert serialized == 'RSVP Calibration -> Offline Analysis -> RSVP Copy Phrase'

    def test_serializes_multiple_tasks(self) -> None:
        actions = [TaskType.RSVP_CALIBRATION, TaskType.RSVP_COPY_PHRASE]
        serialized = serialize_sequence(actions)
        assert serialized == 'RSVP Calibration -> RSVP Copy Phrase'
