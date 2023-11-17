import unittest

from bcipy.display.paradigm.vep.codes import ssvep_to_code
from bcipy.helpers.exceptions import BciPyCoreException


class SSVEPStimuli(unittest.TestCase):
    """Tests for ssvep_to_code"""

    def test_default_flicker_and_refresh_rate_return_codes(self):
        """Test defaults"""
        response = ssvep_to_code()
        self.assertIsInstance(response, list)
        self.assertEqual(response[0], 0)

    def test_ssvep_to_codes_returns_the_length_of_refresh_rate(self):
        """Test with custom refresh rate"""
        refresh_rate = 40
        flicker_rate = 2
        response = ssvep_to_code(flicker_rate=flicker_rate,
                                 refresh_rate=refresh_rate)
        self.assertTrue(len(response) == refresh_rate)
        self.assertEqual(response[0], 0)
        self.assertEqual(response[-1], 1)

    def test_exception_when_refresh_rate_less_than_flicker_rate(self):
        """Test refresh rate less than flicker rate"""
        flicker_rate = 300
        refresh_rate = 1

        with self.assertRaises(BciPyCoreException):
            ssvep_to_code(refresh_rate, flicker_rate)

    def test_refresh_rate_not_evenly_divisible(self):
        """Test when refresh to flicker rates is not evenly divisible."""
        flicker_rate = 11
        refresh_rate = 60

        with self.assertRaises(BciPyCoreException):
            ssvep_to_code(refresh_rate, flicker_rate)

    def test_expected_codes(self):
        """Test that the expected codes are returned"""
        flicker_rate = 2
        refresh_rate = 4
        response = ssvep_to_code(flicker_rate=flicker_rate,
                                 refresh_rate=refresh_rate)
        expected_output = [0, 0, 1, 1]
        self.assertEqual(response, expected_output)

    def test_insufficient_flicker_rate(self):
        """Test that an exception is thrown when the flicker rate is 1 or less
        """
        flicker_rate = 1
        refresh_rate = 2
        with self.assertRaises(BciPyCoreException):
            ssvep_to_code(refresh_rate, flicker_rate)

        flicker_rate = 0
        refresh_rate = 2
        with self.assertRaises(BciPyCoreException):
            ssvep_to_code(refresh_rate, flicker_rate)


if __name__ == '__main__':
    unittest.main()
