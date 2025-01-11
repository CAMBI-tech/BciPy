"""Tests for GUI utils"""
import unittest
from typing import Optional

from bcipy.simulator.data.sampler.base_sampler import Sampler
from bcipy.simulator.ui.gui_utils import InputField, get_inputs


class TestSampler(Sampler):
    """Sampler used for testing"""

    def __init__(self,
                 data_engine,
                 a: int,
                 b: str,
                 c: int = 10,
                 d: Optional[str] = None):
        super().__init__(data_engine)
        self.a = a
        self.b = b
        self.c = c
        self.d = d


class GuiUtilsTest(unittest.TestCase):
    """Tests for GUI utility functions."""

    def test_sampler_params(self):
        """Test that introspecting a Sampler object produces the correct input
        parameters."""

        inputs = get_inputs(TestSampler)

        self.assertTrue(
            InputField(name="a", input_type="int", value=None) in inputs,
            "param a should be an int")
        self.assertTrue(
            InputField(name="b", input_type="str", value=None) in inputs,
            "param b should be a str")
        self.assertTrue(
            InputField(name="c", input_type="int", value=10) in inputs,
            "param c should be an int with value")
        self.assertTrue(
            InputField(name="d", input_type="str", value=None, required=False)
            in inputs, "param d should be an optional str")


if __name__ == '__main__':
    unittest.main()
