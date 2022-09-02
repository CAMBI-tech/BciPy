import unittest
from bcipy.gui.main import float_input_properties, FloatInputProperties


class TestFloatInputProperties(unittest.TestCase):
    """Test calculation of Float input properties."""

    def test_negative(self):
        """Test negative floats"""

        self.assertEqual(float_input_properties('-0.075'),
                         FloatInputProperties(decimals=3, step=0.001))
        self.assertEqual(float_input_properties('-100'),
                         FloatInputProperties(decimals=0, step=1))

    def test_positive(self):
        """Test positive floats"""
        self.assertEqual(float_input_properties('0.2'),
                         FloatInputProperties(decimals=1, step=0.1))
        self.assertEqual(float_input_properties('1023'),
                         FloatInputProperties(decimals=0, step=1))
        self.assertEqual(float_input_properties('1023.0'),
                         FloatInputProperties(decimals=1, step=0.1))
        self.assertEqual(float_input_properties('1023.52'),
                         FloatInputProperties(decimals=2, step=0.01))

    def test_negative_exponents(self):
        """Test floats with negative exponents"""
        # -0.000753
        self.assertEqual(float_input_properties('-753E-6'),
                         FloatInputProperties(decimals=6, step=0.000001))
        # -0.000075
        self.assertEqual(float_input_properties('-75E-6'),
                         FloatInputProperties(decimals=6, step=0.000001))
        # -0.000075
        self.assertEqual(float_input_properties('-7.5e-05'),
                         FloatInputProperties(decimals=6, step=0.000001))
        self.assertEqual(float_input_properties('3e2'),
                         FloatInputProperties(decimals=1, step=0.1))

    def test_positive_exponents(self):
        """Test floats with positive exponents"""
        self.assertEqual(float_input_properties('75E+6'),
                         FloatInputProperties(decimals=1, step=0.1))


if __name__ == '__main__':
    unittest.main()
