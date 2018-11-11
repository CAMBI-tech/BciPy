import unittest

import tempfile
import shutil
import pickle

from bcipy.helpers.load import (
    load_json_parameters,
    load_experimental_data,
    load_signal_model,
    load_txt_data)


class TestLoad(unittest.TestCase):
    """This is Test Case for Loading BCI data."""

    def setUp(self):
        """set up the needed path for load functions."""

        self.parameters = 'bcipy/parameters/parameters.json'
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

if __name__ == '__main__':
    unittest.main()