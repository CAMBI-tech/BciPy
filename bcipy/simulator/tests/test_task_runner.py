import unittest

from bcipy.simulator.task.task_runner import (TargetNontargetSampler, classify,
                                              parse_args)


class TaskRunnerTest(unittest.TestCase):
    """Unit tests for task_runner module."""

    def test_classify(self):
        """Test classifying a string"""
        self.assertEqual(TargetNontargetSampler,
                         classify("TargetNontargetSampler"))

    def test_parse_args(self):
        """Test parsing sampler args"""
        expected = dict(a=1, b=2, c="hello")
        self.assertEqual(expected,
                         parse_args('{"a": 1, "b": 2.0, "c": "hello"}'))
