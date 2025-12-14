"""Test functionality of UI component"""

import sys
import unittest
from typing import Optional

# pylint: disable=E0611
from PyQt6.QtWidgets import QApplication

from bcipy.simulator.ui.obj_args_widget import ObjectArgInputs

app = QApplication(sys.argv)


class NoArgs:
    """Object without args"""


class ArgsObj:
    """Object with args used for testing"""

    def __init__(self, a: int, b: str, c: int = 10, d: Optional[str] = None):
        self.a = a
        self.b = b
        self.c = c
        self.d = d


class ObjectArgsWidgetTest(unittest.TestCase):
    """Tests for component."""

    def test_no_args(self):
        """Test object with no args"""
        widget = ObjectArgInputs(object_type=NoArgs)
        self.assertEqual(0, len(widget.controls))

    def test_with_args(self):
        """Test object with args"""
        widget = ObjectArgInputs(object_type=ArgsObj)
        self.assertEqual(4, len(widget.controls))

        labels = [control[0] for control in widget.controls]
        self.assertTrue('a' in labels)
        self.assertTrue('b' in labels)
        self.assertTrue('c' in labels)
        self.assertTrue('d' in labels)

    def test_set_obj(self):
        """Test that the widget updates"""
        widget = ObjectArgInputs(object_type=NoArgs)
        widget.set_object_type(ArgsObj)
        self.assertEqual(4, len(widget.controls))

    def test_value_no_args(self):
        """Test getting value with no args"""
        widget = ObjectArgInputs(object_type=NoArgs)
        self.assertEqual('{}', widget.value())

    def test_value_with_args(self):
        """Test get value with default args"""
        widget = ObjectArgInputs(object_type=ArgsObj)
        self.assertEqual('{"a": null, "b": null, "c": 10, "d": null}',
                         widget.value())

    def test_value_set_args(self):
        """Test get value after updating inputs"""
        widget = ObjectArgInputs(object_type=ArgsObj)
        b_control = None
        for label, control in widget.controls:
            if label == "b":
                b_control = control
                break
        self.assertIsNotNone(b_control)
        b_control.setText("hello")

        self.assertEqual('{"a": null, "b": "hello", "c": 10, "d": null}',
                         widget.value())

    def test_required_inputs(self):
        """Test check for required inputs"""
        widget = ObjectArgInputs(object_type=ArgsObj)
        [a, b, _c, _d] = [control for _name, control in widget.controls]

        self.assertFalse(widget.required_inputs_provided())

        a.setText("10")
        b.setText("hi")
        self.assertTrue(widget.required_inputs_provided())

    def test_required_inputs_no_args(self):
        """Test check for required inputs with no args"""
        widget = ObjectArgInputs(object_type=NoArgs)
        self.assertTrue(widget.required_inputs_provided())


if __name__ == '__main__':
    unittest.main()
