import shutil
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from bcipy.helpers import load
from bcipy.simulator.helpers.data_engine import (RawDataEngine,
                                                 RawDataEngineWrapper)

# Import the module or class you want to test

FILE_PATH_PREFIX = "bcipy/simulator/tests"


class TestRawDataEngine(unittest.TestCase):

    @classmethod
    def setUp(cls) -> None:
        with zipfile.ZipFile(f'{FILE_PATH_PREFIX}/resource/rsvpTest1.zip', 'r') as zip_ref:
            zip_ref.extractall(f'{FILE_PATH_PREFIX}/resource/rsvpTest1')

    @classmethod
    def tearDown(cls) -> None:
        shutil.rmtree(f'{FILE_PATH_PREFIX}/resource/rsvpTest1')

    @classmethod
    def check_eeg_shape(cls, schema: pd.DataFrame):
        # checking all eeg response data has same shape

        eeg_responses = schema['eeg']
        ret = True
        ret = ret and not (None in eeg_responses or np.NaN in eeg_responses)
        shape_set = set(eeg_responses.apply(lambda x: x.shape))
        ret = ret and (len(shape_set) == 1)

        return ret

    def test_one_source(self):
        # Test the function with a specific input and assert the expected output

        data_paths = ["/Users/srikarananthoju/cambi/tab_test_dynamic/16sec_-0700"]
        param_path = "/Users/srikarananthoju/cambi/tab_test_dynamic/calibr_37sec_-0700/parameters.json"
        params = load.load_json_parameters(str(Path(param_path)), value_cast=True)

        data_engine = RawDataEngine(data_paths, params)
        self.assertFalse(data_engine.get_data())

        data_engine.transform()
        self.assertTrue(len(data_engine.schema))

        self.assertTrue(self.check_eeg_shape(data_engine.get_data()))

    def test_two_sources(self):
        # Test the function with a specific input and assert the expected output

        data_paths = ["/Users/srikarananthoju/cambi/tab_test_dynamic/16sec_-0700",
                      "/Users/srikarananthoju/cambi/tab_test_dynamic/50sec_-0700"]
        param_path = "/Users/srikarananthoju/cambi/tab_test_dynamic/calibr_37sec_-0700/parameters.json"
        params = load.load_json_parameters(str(Path(param_path)), value_cast=True)

        data_engine = RawDataEngine(data_paths, params)
        self.assertFalse(data_engine.get_data())

        data_engine.transform()
        self.assertTrue(len(data_engine.get_data()))

        # asserting all eeg response data has same shape
        self.assertTrue(self.check_eeg_shape(data_engine.get_data()))

    def test_RawDataEngineWrapper(self):
        # Testing loading and transformaing data from a singular wrapper folder

        # source_dir = "/Users/srikarananthoju/cambi/tab_test_dynamic/wrapper"
        source_dir = FILE_PATH_PREFIX + "/resource/rsvpTest1/wrapper"
        param_path = "/Users/srikarananthoju/cambi/tab_test_dynamic/calibr_37sec_-0700/parameters.json"
        params = load.load_json_parameters(str(Path(param_path)), value_cast=True)

        data_engine = RawDataEngineWrapper(source_dir, params)
        self.assertEqual(len(data_engine.source_dirs), 2)

        data_engine.transform()
        self.assertTrue(len(data_engine.get_data()))

        self.assertTrue(self.check_eeg_shape(data_engine.get_data()))

    def test_matrix_single(self):
        source_dir = "/Users/srikarananthoju/cambi/data/matrixTestWrapper"
        param_path = f"{source_dir}/S007_Matrix_Calibration_Thu_20_Jul_2023_09hr48min56sec_-0400/parameters.json"
        params = load.load_json_parameters(str(Path(param_path)), value_cast=True)

        data_engine = RawDataEngineWrapper(source_dir, params)
        data_engine.transform()

        self.assertTrue(len(data_engine.get_data()))

        self.assertTrue(self.check_eeg_shape(data_engine.get_data()))


if __name__ == '__main__':
    unittest.main()
