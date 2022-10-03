"""Tests for data conversion related functionality."""
import os
import shutil
import tempfile
from typing import Tuple
import unittest
import warnings

from pathlib import Path

from mockito import mock, when, unstub, verify, verifyNoMoreInteractions, any as any_value

from bcipy.config import DEFAULT_ENCODING, RAW_DATA_FILENAME, TRIGGER_FILENAME, DEFAULT_PARAMETER_FILENAME
from bcipy.helpers import convert
from bcipy.helpers.convert import (
    archive_list,
    compress,
    convert_to_bdf,
    convert_to_edf,
    decompress,
    pyedf_convert,
)
from bcipy.helpers.parameters import Parameters
from bcipy.helpers.raw_data import sample_data, write, RawData
from bcipy.helpers.triggers import MOCK_TRIGGER_DATA

from mne.io import read_raw_edf, read_raw_bdf


def create_bcipy_session_artifacts(
        write_dir: str,
        channels: int = 3,
        sample_rate: int = 300,
        samples: int = 5000,
        filter_settings: dict = {
            'filter_low': 0.5,
            'filter_high': 30,
            'filter_order': 5,
            'notch_filter_frequency': 60,
            'down_sampling_rate': 3,
            'static_trigger_offset': 0.0
        },
) -> Tuple[str, RawData, Parameters]:
    """Write BciPy session artifacts to a temporary directory.

    This includes a raw data file, trigger file, and a parameters file.
    """
    trg_data = MOCK_TRIGGER_DATA
    channels = [f'ch{i}' for i in range(channels)]
    data = sample_data(ch_names=channels, sample_rate=sample_rate, rows=samples)
    with open(Path(write_dir, TRIGGER_FILENAME), 'w', encoding=DEFAULT_ENCODING) as trg_file:
        trg_file.write(trg_data)

    write(data, Path(write_dir, f'{RAW_DATA_FILENAME}.csv'))

    params = Parameters.from_cast_values(raw_data_name=f'{RAW_DATA_FILENAME}.csv',
                                         trigger_file_name=TRIGGER_FILENAME,
                                         preview_inquiry_length=0.5,
                                         time_fixation=0.5,
                                         time_prompt=0.5,
                                         time_flash=0.5,
                                         # define filter settings
                                         static_trigger_offset=filter_settings['static_trigger_offset'],
                                         down_sampling_rate=filter_settings['down_sampling_rate'],
                                         notch_filter_frequency=filter_settings['notch_filter_frequency'],
                                         filter_high=filter_settings['filter_high'],
                                         filter_low=filter_settings['filter_low'],
                                         filter_order=filter_settings['filter_order'])
    params.save(write_dir, DEFAULT_PARAMETER_FILENAME)
    return trg_data, data, params


class TestEDFConvert(unittest.TestCase):
    """Tests for EDF data format conversions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        _, data, _ = create_bcipy_session_artifacts(self.temp_dir)
        self.channels = data.channels

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        verifyNoMoreInteractions()
        unstub()

    def test_convert_to_edf_defaults(self):
        """Test default behavior"""
        path = convert_to_edf(self.temp_dir)
        self.assertTrue(os.path.exists(path))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            edf = read_raw_edf(path, preload=True)

        self.assertTrue(len(edf.get_data()) > 0)

        for ch_name in self.channels:
            self.assertTrue(ch_name in edf.ch_names)

    def test_convert_to_edf_overwrite_false(self):
        """Test overwriting fails"""

        convert_to_edf(self.temp_dir)
        with self.assertRaises(OSError):
            convert_to_edf(self.temp_dir, overwrite=False)

    def test_convert_to_edf_overwrite_true(self):
        """Test that overwriting can be configured"""

        convert_to_edf(self.temp_dir)
        convert_to_edf(self.temp_dir, overwrite=True)

    def test_convert_to_edf_with_custom_path(self):
        """Test creating the EDF using a custom edf path"""
        path = convert_to_edf(self.temp_dir,
                              edf_path=str(Path(self.temp_dir, 'mydata.edf').resolve()))

        self.assertEqual(Path(path).name, 'mydata.edf')

    def test_convert_to_edf_raises_value_error_with_invalid_edfpath(self):
        """Test that an value error is raised when the edf path is invalid"""
        edf_path = 'invalid.pdf'
        with self.assertRaises(ValueError):
            convert_to_edf(self.temp_dir, edf_path=edf_path)

    def test_convert_to_edf_with_write_targetness(self):
        """Test creating the EDF using targetness for event annotations"""
        path = convert_to_edf(self.temp_dir,
                              write_targetness=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            edf = read_raw_edf(path, preload=True)

        self.assertIn('target', edf.annotations.description)
        self.assertIn('nontarget', edf.annotations.description)

    def test_convert_to_edf_without_write_targetness(self):
        """Test creating the EDF with labels as event annotations"""
        path = convert_to_edf(self.temp_dir,
                              write_targetness=False)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            edf = read_raw_edf(path, preload=True)

        self.assertNotIn('target', edf.annotations.description)
        self.assertNotIn('nontarget', edf.annotations.description)
        self.assertIn('+', edf.annotations.description)

    def test_convert_to_edf_with_pre_filter(self):
        """Test creating the EDF with pre-filtering"""

        data = mock()
        channels = mock()
        fs = 100
        events = mock()
        annotation_channels = 1
        return_path = mock()
        when(convert).pyedf_convert(
            any_value(str),
            write_targetness=any_value(bool),
            use_event_durations=any_value(bool),
            remove_pre_fixation=any_value(bool),
            pre_filter=True,
        ).thenReturn((
            data,
            channels,
            fs,
            events,
            annotation_channels,
            'filter_settings'
        ))
        when(convert).write_pyedf(
            any_value(str),
            data,
            channels,
            fs,
            events,
            any_value(int),
            any_value(bool),
            annotation_channels,
            any_value(str)).thenReturn(return_path)
        path = convert_to_edf(self.temp_dir,
                              pre_filter=True)

        verify(convert, times=1).pyedf_convert(
            any_value(str),
            write_targetness=any_value(bool),
            use_event_durations=any_value(bool),
            remove_pre_fixation=any_value(bool),
            pre_filter=True,
        )
        verify(convert, times=1).write_pyedf(
            any_value(str),
            data,
            channels,
            fs,
            events,
            any_value(int),
            any_value(bool),
            annotation_channels,
            any_value(str))
        self.assertEqual(path, return_path)


class TestBDFConvert(unittest.TestCase):
    """Tests for BDF data format conversions."""

    def setUp(self):
        """Override; set up the needed path for load functions."""

        self.temp_dir = tempfile.mkdtemp()
        _, data, _ = create_bcipy_session_artifacts(self.temp_dir)
        self.channels = data.channels

    def tearDown(self):
        """Override"""
        shutil.rmtree(self.temp_dir)
        verifyNoMoreInteractions()
        unstub()

    def test_convert_to_bdf_defaults(self):
        """Test default convert to bdf behavior"""
        path = convert_to_bdf(self.temp_dir)
        self.assertTrue(os.path.exists(path))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bdf = read_raw_bdf(path, preload=True)

        self.assertTrue(len(bdf.get_data()) > 0)

        for ch_name in self.channels:
            self.assertTrue(ch_name in bdf.ch_names)

    def test_convert_to_bdf_overwrite_false(self):
        """Test overwriting fails if not configured"""

        convert_to_bdf(self.temp_dir)
        with self.assertRaises(OSError):
            convert_to_bdf(self.temp_dir, overwrite=False)

    def test_convert_to_bdf_overwrite_true(self):
        """Test that overwriting can be configured"""

        convert_to_bdf(self.temp_dir)
        convert_to_bdf(self.temp_dir, overwrite=True)

    def test_convert_to_bdf_with_custom_path(self):
        """Test creating the EDF using a custom edf path"""
        path = convert_to_bdf(self.temp_dir,
                              bdf_path=str(Path(self.temp_dir, 'mydata.bdf').resolve()))

        self.assertEqual(Path(path).name, 'mydata.bdf')

    def test_convert_to_bdf_raises_value_error_with_invalid_edfpath(self):
        """Test that a value error is raised with the bdf path is invalid"""
        bdf_path = 'invalid.pdf'
        with self.assertRaises(ValueError):
            convert_to_bdf(self.temp_dir, bdf_path=bdf_path)

    def test_convert_to_bdf_with_write_targetness(self):
        """Test creating the EDF using targetness for event annotations"""
        path = convert_to_bdf(self.temp_dir,
                              write_targetness=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            edf = read_raw_bdf(path, preload=True)

        self.assertIn('target', edf.annotations.description)
        self.assertIn('nontarget', edf.annotations.description)

    def test_convert_to_bdf_without_write_targetness(self):
        """Test creating the EDF with labels as event annotations"""
        path = convert_to_bdf(self.temp_dir,
                              write_targetness=False)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            edf = read_raw_bdf(path, preload=True)

        self.assertNotIn('target', edf.annotations.description)
        self.assertNotIn('nontarget', edf.annotations.description)
        self.assertIn('+', edf.annotations.description)

    def test_convert_to_bdf_with_pre_filter(self):
        """Test creating the BDF with pre-filtering"""

        data = mock()
        channels = mock()
        fs = 100
        events = mock()
        annotation_channels = 1
        return_path = mock()
        when(convert).pyedf_convert(
            any_value(str),
            write_targetness=any_value(bool),
            use_event_durations=any_value(bool),
            remove_pre_fixation=any_value(bool),
            pre_filter=True,
        ).thenReturn((
            data,
            channels,
            fs,
            events,
            annotation_channels,
            'filter_settings'
        ))
        when(convert).write_pyedf(
            any_value(str),
            data,
            channels,
            fs,
            events,
            any_value(int),
            any_value(bool),
            annotation_channels,
            any_value(str)).thenReturn(return_path)
        path = convert_to_bdf(self.temp_dir,
                              pre_filter=True)

        verify(convert, times=1).pyedf_convert(
            any_value(str),
            write_targetness=any_value(bool),
            use_event_durations=any_value(bool),
            remove_pre_fixation=any_value(bool),
            pre_filter=True,
        )
        verify(convert, times=1).write_pyedf(
            any_value(str),
            data,
            channels,
            fs,
            events,
            any_value(int),
            any_value(bool),
            annotation_channels,
            any_value(str))
        self.assertEqual(path, return_path)


class TestPyedfconvert(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.sample_rate = 300
        self.filter_settings = {
            'filter_low': 0.5,
            'filter_high': 30,
            'filter_order': 5,
            'notch_filter_frequency': 60,
            'down_sampling_rate': 3,
            'static_trigger_offset': 0.0
        }
        create_bcipy_session_artifacts(
            self.temp_dir,
            sample_rate=self.sample_rate,
            filter_settings=self.filter_settings,
            samples=10000)  # sample number determines the annotation count, set it high here!

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)
        verifyNoMoreInteractions()
        unstub()

    def test_pyedf_convert_defaults(self):
        """Test the pyedf_convert function"""
        data, channels, fs, events, annotation_channels, pre_filter = pyedf_convert(
            self.temp_dir)

        self.assertTrue(len(data) > 0)
        self.assertTrue(len(channels) > 0)
        self.assertEqual(fs, self.sample_rate)
        self.assertTrue(len(events) > 2)
        # see the validate_annotations function for details on how this is calculated
        self.assertEqual(annotation_channels, 1)
        self.assertFalse(pre_filter)

    def test_pyedf_convert_with_pre_filter(self):
        """Test the pyedf_convert function with pre-filtering"""
        data, channels, fs, events, annotation_channels, pre_filter = pyedf_convert(
            self.temp_dir, pre_filter=True)

        self.assertTrue(len(data) > 0)
        self.assertTrue(len(channels) > 0)
        self.assertEqual(fs, self.sample_rate / self.filter_settings['down_sampling_rate'])
        self.assertTrue(len(events) > 2)
        # see the validate_annotations function for details on how this is calculated
        #  with less samples, the number of annotation channels increases
        self.assertEqual(annotation_channels, 2)
        self.assertIsInstance(pre_filter, str)

    def test_pyedf_convert_with_write_targetness(self):
        """Test the pyedf_convert function with targetness for event annotations"""
        data, channels, fs, events, annotation_channels, _ = pyedf_convert(
            self.temp_dir, write_targetness=True)

        self.assertTrue(len(data) > 0)
        self.assertTrue(len(channels) > 0)
        self.assertEqual(fs, self.sample_rate)
        self.assertTrue(len(events) > 2)
        # see the validate_annotations function for details on how this is calculated
        self.assertEqual(annotation_channels, 1)
        for event in events:
            _, _, label = event
            # in this casae label and targetness should not be the same
            self.assertTrue(label in ['target', 'nontarget'])

    def test_pyedf_convert_without_write_targetness(self):
        """Test the pyedf_convert function with labels as event annotations"""
        data, channels, fs, events, annotation_channels, _ = pyedf_convert(
            self.temp_dir, write_targetness=False)

        self.assertTrue(len(data) > 0)
        self.assertTrue(len(channels) > 0)
        self.assertEqual(fs, self.sample_rate)
        self.assertTrue(len(events) > 2)
        # see the validate_annotations function for details on how this is calculated
        self.assertEqual(annotation_channels, 1)
        for event in events:
            _, _, label = event
            self.assertTrue(label not in ['target', 'nontarget'])

    def test_pyedf_convert_with_use_event_durations(self):
        """Test the pyedf_convert function with event durations"""
        data, channels, fs, events, annotation_channels, _ = pyedf_convert(
            self.temp_dir, use_event_durations=True)

        self.assertTrue(len(data) > 0)
        self.assertTrue(len(channels) > 0)
        self.assertEqual(fs, self.sample_rate)
        self.assertTrue(len(events) > 2)
        # see the validate_annotations function for details on how this is calculated
        self.assertEqual(annotation_channels, 1)
        for _, duration, _ in events:
            self.assertTrue(duration > 0)

    def test_pyedf_convert_without_use_event_durations(self):
        """Test the pyedf_convert function without event durations"""
        data, channels, fs, events, annotation_channels, _ = pyedf_convert(
            self.temp_dir, use_event_durations=False)

        self.assertTrue(len(data) > 0)
        self.assertTrue(len(channels) > 0)
        self.assertEqual(fs, self.sample_rate)
        self.assertTrue(len(events) > 2)
        # see the validate_annotations function for details on how this is calculated
        self.assertEqual(annotation_channels, 1)
        for _, duration, _ in events:
            self.assertEqual(duration, 0)

    def test_pyedf_convert_with_remove_pre_fixation(self):
        """Test the pyedf_convert function with pre-fixation removal"""
        data, channels, fs, events, annotation_channels, _ = pyedf_convert(
            self.temp_dir, remove_pre_fixation=True)

        self.assertTrue(len(data) > 0)
        self.assertTrue(len(channels) > 0)
        self.assertEqual(fs, self.sample_rate)
        self.assertTrue(len(events) > 2)
        # see the validate_annotations function for details on how this is calculated
        self.assertEqual(annotation_channels, 1)
        for event in events:
            self.assertNotIn('fixation', event)
            self.assertNotIn('prompt', event)


class TestCompressionSupport(unittest.TestCase):

    def setUp(self):
        self.dir_name = 'test/'
        self.tar_file_name = 'test_file'
        self.tar_file_full_name = f'{self.tar_file_name}.tar.gz'
        self.test_file_name = 'test.text'
        with open(self.test_file_name, 'w', encoding=DEFAULT_ENCODING) as fp:
            pass

    def tearDown(self):
        os.remove(self.test_file_name)

        if os.path.exists(self.tar_file_full_name):
            os.remove(self.tar_file_full_name)

    def test_compression_writes_tar_gz_no_extension(self):
        # Test with no extension on tar name
        compress(self.tar_file_name, [self.test_file_name])
        # Assert correct file was written
        self.assertTrue(os.path.exists(self.tar_file_full_name))

    def test_compression_writes_tar_gz_with_extension(self):
        # Test with extension on tar name
        compress(self.tar_file_full_name, [self.test_file_name])
        # Assert correct file was written
        self.assertTrue(os.path.exists(self.tar_file_full_name))

    def test_decompression_extracts_file_no_extension(self):
        # Test with no extension on tar name
        compress(self.tar_file_name, [self.test_file_name])
        decompress(self.tar_file_name, ".")
        self.assertTrue(os.path.exists(self.test_file_name))

    def test_decompression_extracts_file_with_extension(self):
        # Test with extension on tar name
        compress(self.tar_file_name, [self.test_file_name])
        decompress(self.tar_file_full_name, ".")
        self.assertTrue(os.path.exists(self.test_file_name))

    def test_file_not_found_error_thrown_on_compression(self):
        garbage_name = 'not_possible_to_exist.biz'
        with self.assertRaises(FileNotFoundError):
            compress(self.tar_file_name, [garbage_name])

    def test_file_list_returns_compressed_file_name_no_extension(self):
        # Test with no extension on tar name
        compress(self.tar_file_name, [self.test_file_name])
        tar_list = archive_list(self.tar_file_name)
        self.assertTrue(tar_list[0] == self.test_file_name)

    def test_file_list_returns_compressed_file_name_with_extension(self):
        # Test with extension on tar name
        compress(self.tar_file_name, [self.test_file_name])
        tar_list = archive_list(self.tar_file_full_name)
        self.assertTrue(tar_list[0] == self.test_file_name)


if __name__ == '__main__':
    unittest.main()
