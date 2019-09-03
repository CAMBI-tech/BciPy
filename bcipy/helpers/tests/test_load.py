import unittest

import tempfile
import shutil
import pickle
from os import remove

from bcipy.helpers.load import (
    load_json_parameters,
    load_signal_model,
    get_missing_parameter_keys,
    PARAM_LOCATION_DEFAULT)


class TestLoad(unittest.TestCase):
    """This is Test Case for Loading BCI data."""

    def setUp(self):
        """set up the needed path for load functions."""

        self.parameters = PARAM_LOCATION_DEFAULT
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_json_parameters_returns_dict(self):
        """Test load parameters returns a Python dict."""

        # call the load parameters function
        parameters = load_json_parameters(self.parameters)

        # assert that load function turned json parameters into a dict
        self.assertTrue(type(parameters), 'dict')

    def test_load_json_parameters_throws_error_on_wrong_path(self):
        """Test load parameters returns error on entering wrong path."""

        # call the load parameters function with incorrect path
        with self.assertRaises(Exception):
            load_json_parameters('/garbage/dir/wont/work')

    def test_load_classifier(self):
        """Test load classifier can load pickled file when given path."""

        # create a pickle file to save a pickled json
        pickle_file = self.temp_dir + "save.p"
        pickle.dump(self.parameters, open(pickle_file, "wb"))

        # Load classifier
        unpickled_parameters = load_signal_model(pickle_file)

        # assert the same data was returned
        self.assertEqual(unpickled_parameters, (self.parameters, pickle_file))

    def test_get_parameter_keys(self):
        """Test the function that adds default values from parameters.json to
        the parameters if they are not present in the loaded file"""
        parameters = load_json_parameters(self.parameters, True)
        parameters.pop('fake_data', None)
        with open('temp_parameter_file.json', 'w') as outfile:
            outfile.write('test')
        missing_key_list = get_missing_parameter_keys(parameters, 'temp_parameter_file.json')
        assert list(missing_key_list.keys()) == ['fake_data']
        remove('temp_parameter_file.json')


if __name__ == '__main__':
    unittest.main()
