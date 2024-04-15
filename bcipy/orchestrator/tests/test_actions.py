import unittest
from bcipy.orchestrator.parameters.actions import parse_actions, serialize_actions, validate_action_string
from bcipy.task import TaskType


class TestActions(unittest.TestCase):

    def test_parses_one_task(self) -> None:
        actions = 'RSVP Calibration'
        parsed = parse_actions(actions)
        assert len(parsed) == 1
        assert parsed[0] == TaskType.RSVP_CALIBRATION

    def test_parses_multiple_tasks(self) -> None:
        actions = 'RSVP Calibration -> RSVP Copy Phrase'
        parsed = parse_actions(actions)
        assert len(parsed) == 2
        assert parsed[0] == TaskType.RSVP_CALIBRATION
        assert parsed[1] == TaskType.RSVP_COPY_PHRASE

    def test_throws_exception_on_invalid_task(self) -> None:
        actions = 'RSVP Calibration -> RSVP Copy Phrase -> does not exist'
        with self.assertRaises(ValueError):
            parse_actions(actions)

    def test_validates_valid_action_string(self) -> None:
        actions = 'RSVP Calibration -> RSVP Copy Phrase'
        validate_action_string(actions)

    def test_throws_exception_on_invalid_action_string(self) -> None:
        actions = 'RSVP Calibration -> RSVP Copy Phrase -> does not exist'
        with self.assertRaises(ValueError):
            validate_action_string(actions)

    def test_serializes_one_task(self) -> None:
        actions = [TaskType.RSVP_CALIBRATION]
        serialized = serialize_actions(actions)
        assert serialized == 'RSVP Calibration'

    def test_serializes_multiple_tasks(self) -> None:
        actions = [TaskType.RSVP_CALIBRATION, TaskType.RSVP_COPY_PHRASE]
        serialized = serialize_actions(actions)
        assert serialized == 'RSVP Calibration -> RSVP Copy Phrase'
