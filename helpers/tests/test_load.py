import errno
import os
import sys
import unittest

# add previous dirs to python path for importing of modules
sys.path.append('helpers/')

from load import load_json_parameters


class TestLoad(unittest.TestCase):
    """This is Test Case for Loading BCI data."""

    def setUp(self):
        """set up the needed path for load functions."""

        self.parameters_used = './parameters/parameters.json'

    def test_load_json_parameters_returns_dict(self):
        """Test load parameters returns a Python dict."""

        # call the load parameters function
        parameters = load_json_parameters(self.parameters_used)

        # assert that load function turned json parameters into a dict
        self.assertTrue(type(parameters), 'dict')

    def test_load_json_parameters_throws_useful_error_on_wrong_path(self):
        """Test load parameters returns error on entering wrong path."""

        # call the load parameters function with incorrect path
        try:
            load_json_parameters('/garbage/dir/wont/work')

        # catch the exception and make sure it's as expected
        except Exception as error:
            self.assertEqual(
                error.message,
                "Incorrect path to parameters given! Please try again.")
