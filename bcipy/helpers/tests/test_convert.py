"""Tests for data conversion related functionality."""
import os
import shutil
import tempfile
import unittest
import warnings
from itertools import chain
from pathlib import Path
from typing import List

from mne.io import read_raw_edf

from bcipy.helpers.convert import convert_to_edf
from bcipy.helpers.parameters import Parameters
from bcipy.signal.generator.generator import gen_random_data


MOCK_TRIGGER_DATA = '''calibration_trigger calib 0.4748408449813724
J first_pres_target 6.151848723005969
+ fixation 8.118640798988054
F nontarget 8.586895030981395
D nontarget 8.887798132986063
J target 9.18974666899885
T nontarget 9.496583286992973
K nontarget 9.798354075988755
Q nontarget 10.099591801001225
O nontarget 10.401458177977474
Z nontarget 10.70310750597855
R nontarget 11.00485198898241
_ nontarget 11.306160968990298
offset offset_correction 1.23828125'''


def sample_data(rows: int = 1000, ch_names: List[str] = None) -> str:
    """Creates sample data to be written as a raw_data.csv file

    rows - number of sample rows to generate
    ch_names - channel names
    """
    if not ch_names:
        ch_names = ['ch1', 'ch2', 'ch3']
    # Mock the raw_data file
    sep = '\n'
    meta = sep.join([f'daq_type,LSL', 'sample_rate,256.0'])
    header = 'timestamp,' + ','.join(ch_names) + ',TRG'

    data = []
    for i in range(rows):
        channel_data = gen_random_data(low=-1000,
                                       high=1000,
                                       channel_count=len(ch_names))
        columns = chain([str(i)], map(str, channel_data), ['0.0'])
        data.append(','.join(columns))

    return sep.join([meta, header, *data])


class TestConvert(unittest.TestCase):
    """Tests for data format conversions."""

    @classmethod
    def setUpClass(cls):
        """Initialize data once"""
        cls.trg_data = MOCK_TRIGGER_DATA
        cls.channels = ['ch1', 'ch2', 'ch3']
        cls.sample_data = sample_data(rows=3000, ch_names=cls.channels)

    def setUp(self):
        """Override; set up the needed path for load functions."""

        self.temp_dir = tempfile.mkdtemp()

        self.default_mode = 'calibration'

        with open(Path(self.temp_dir, 'triggers.txt'), 'w') as trg_file:
            trg_file.write(self.__class__.trg_data)

        with open(Path(self.temp_dir, 'raw_data.csv'), 'w') as data_file:
            data_file.write(self.__class__.sample_data)

        params = Parameters.from_cast_values(raw_data_name='raw_data.csv',
                                             trigger_file_name='triggers.txt')
        params.save(self.temp_dir, 'parameters.json')

    def tearDown(self):
        """Override"""
        shutil.rmtree(self.temp_dir)

    def test_convert_defaults(self):
        """Test default behavior"""
        path = convert_to_edf(self.temp_dir, mode=self.default_mode)
        self.assertTrue(os.path.exists(path))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            edf = read_raw_edf(path, preload=True)

        self.assertTrue(len(edf.get_data()) > 0)

        for ch_name in self.channels:
            self.assertTrue(ch_name in edf.ch_names)

    def test_overwrite_false(self):
        """Test overwriting fails"""

        convert_to_edf(self.temp_dir, mode=self.default_mode)
        with self.assertRaises(OSError):
            convert_to_edf(self.temp_dir, mode=self.default_mode, overwrite=False)

    def test_overwrite_true(self):
        """Test that overwriting can be configured"""

        convert_to_edf(self.temp_dir, mode=self.default_mode)
        convert_to_edf(self.temp_dir, overwrite=True, mode=self.default_mode)

    def test_with_custom_path(self):
        """Test creating the EDF without event annotations"""
        path = convert_to_edf(self.temp_dir,
                              mode=self.default_mode,
                              edf_path=Path(self.temp_dir, 'mydata.edf'))

        self.assertEqual(Path(path).name, 'mydata.edf')

    def test_with_write_targetness(self):
        """Test creating the EDF without event annotations"""
        path = convert_to_edf(self.temp_dir,
                              mode=self.default_mode,
                              write_targetness=True,
                              edf_path=Path(self.temp_dir, 'mydata.edf'))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            edf = read_raw_edf(path, preload=True)

        self.assertIn('target', edf.annotations.description)
        self.assertIn('nontarget', edf.annotations.description)

    def test_with_annotation_channels(self):
        """Test creating the EDF without event annotations"""
        # Increase the number of annotation channels by 1
        annotation_channels = 2
        # The expected channels should be equal to the number of annotation channels +
        #   channels defined in the class + a TRG channel
        expected_channel_number = annotation_channels + len(self.channels) + 1
        path = convert_to_edf(self.temp_dir,
                              mode=self.default_mode,
                              write_targetness=True,
                              annotation_channels=annotation_channels,
                              edf_path=Path(self.temp_dir, 'mydata.edf'))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            edf = read_raw_edf(path, preload=True)

        self.assertEqual(len(edf.ch_names), expected_channel_number)


if __name__ == '__main__':
    unittest.main()
