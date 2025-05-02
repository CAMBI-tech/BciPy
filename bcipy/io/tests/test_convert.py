"""Tests for data conversion related functionality."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Tuple, Union
from unittest.mock import patch, MagicMock

import bcipy.acquisition.devices as devices
import mne
from bcipy.config import (DEFAULT_ENCODING, DEFAULT_PARAMETERS_FILENAME,
                          RAW_DATA_FILENAME, TRIGGER_FILENAME)
from bcipy.io.convert import (
    archive_list,
    compress,
    ConvertFormat,
    convert_to_bids,
    convert_to_mne,
    decompress,
    norm_to_tobii,
    tobii_to_norm,
    convert_eyetracking_to_bids,
    BIDS_to_MNE
)
from bcipy.core.parameters import Parameters
from bcipy.core.raw_data import RawData, sample_data, write
from bcipy.core.triggers import MOCK_TRIGGER_DATA
from bcipy.signal.generator.generator import gen_random_data


CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']


def create_bcipy_session_artifacts(
        write_dir: str,
        channels: Union[int, list] = 3,
        sample_rate: int = 300,
        samples: int = 5000,
        filter_settings: dict = {
            'filter_low': 0.5,
            'filter_high': 30,
            'filter_order': 5,
            'notch_filter_frequency': 60,
            'down_sampling_rate': 3
        },
) -> Tuple[str, RawData, Parameters]:
    """Write BciPy session artifacts to a temporary directory.

    This includes a raw data file, trigger file, and a parameters file.
    """
    trg_data = MOCK_TRIGGER_DATA
    if isinstance(channels, int):

        channels = [CHANNEL_NAMES[i] for i in range(channels)]
    data = sample_data(ch_names=channels, daq_type='SampleDevice', sample_rate=sample_rate, rows=samples)
    devices.register(devices.DeviceSpec('SampleDevice', channels=channels, sample_rate=sample_rate))

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
                                         down_sampling_rate=filter_settings['down_sampling_rate'],
                                         notch_filter_frequency=filter_settings['notch_filter_frequency'],
                                         filter_high=filter_settings['filter_high'],
                                         filter_low=filter_settings['filter_low'],
                                         filter_order=filter_settings['filter_order'])
    params.save(write_dir, DEFAULT_PARAMETERS_FILENAME)
    return trg_data, data, params


class TestBIDSConversion(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.trg_data, self.data, self.params = create_bcipy_session_artifacts(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_convert_to_bids_generates_bids_strucutre(self):
        """Test the convert_to_bids function"""
        response = convert_to_bids(
            f"{self.temp_dir}",
            participant_id='01',
            session_id='01',
            run_id='01',
            task_name='TestTask',
            output_dir=self.temp_dir,
        )
        self.assertTrue(os.path.exists(response))
        # Assert that the BIDS structure was created
        self.assertTrue(os.path.exists(f"{self.temp_dir}"))
        # Assert the session directory was created with eeg
        self.assertTrue(os.path.exists(f"{self.temp_dir}/sub-01/ses-01/eeg/"))
        # Assert the eeg file was created (default of BV format)
        self.assertTrue(os.path.exists(
            f"{self.temp_dir}/sub-01/ses-01/eeg/sub-01_ses-01_task-TestTask_run-01_eeg.vhdr"))
        # Assert the events file was created
        self.assertTrue(os.path.exists(
            f"{self.temp_dir}/sub-01/ses-01/eeg/sub-01_ses-01_task-TestTask_run-01_events.tsv"))
        # Assert the channels file was created
        self.assertTrue(os.path.exists(
            f"{self.temp_dir}/sub-01/ses-01/eeg/sub-01_ses-01_task-TestTask_run-01_channels.tsv"))

    def test_convert_to_bids_reflects_participant_id(self):
        """Test the convert_to_bids function with a participant id"""
        response = convert_to_bids(
            f"{self.temp_dir}",
            participant_id='100',
            session_id='01',
            run_id='01',
            task_name='TestTask',
            output_dir=self.temp_dir,
        )
        self.assertTrue(os.path.exists(response))
        self.assertTrue(os.path.exists(f"{self.temp_dir}/sub-100/"))

    def test_convert_to_bids_reflects_session_id(self):
        """Test the convert_to_bids function with a session id"""
        response = convert_to_bids(
            f"{self.temp_dir}",
            participant_id='01',
            session_id='100',
            run_id='01',
            task_name='TestTask',
            output_dir=self.temp_dir,
        )
        self.assertTrue(os.path.exists(response))
        self.assertTrue(os.path.exists(f"{self.temp_dir}/sub-01/ses-100/"))

    def test_convert_to_bids_reflects_run_id(self):
        """Test the convert_to_bids function with a run id"""
        response = convert_to_bids(
            f"{self.temp_dir}",
            participant_id='01',
            session_id='01',
            run_id='100',
            task_name='TestTask',
            output_dir=self.temp_dir,
        )
        self.assertTrue(os.path.exists(response))
        self.assertTrue(os.path.exists(
            f"{self.temp_dir}/sub-01/ses-01/eeg/sub-01_ses-01_task-TestTask_run-100_eeg.vhdr"))

    def test_convert_to_bids_reflects_task_name(self):
        """Test the convert_to_bids function with a task name"""
        response = convert_to_bids(
            f"{self.temp_dir}",
            participant_id='01',
            session_id='01',
            run_id='01',
            task_name='TestTaskEtc',
            output_dir=self.temp_dir,
        )
        self.assertTrue(os.path.exists(response))
        self.assertTrue(os.path.exists(
            f"{self.temp_dir}/sub-01/ses-01/eeg/sub-01_ses-01_task-TestTaskEtc_run-01_eeg.vhdr"))

    def test_convert_to_bids_edf(self):
        """Test the convert_to_bids function with edf format"""
        response = convert_to_bids(
            f"{self.temp_dir}",
            participant_id='01',
            session_id='01',
            run_id='01',
            task_name='TestTask',
            output_dir=self.temp_dir,
            format=ConvertFormat.EDF
        )

        self.assertTrue(os.path.exists(response))
        # Assert that the BIDS structure was created
        self.assertTrue(os.path.exists(f"{self.temp_dir}"))
        # Assert the session directory was created with eeg
        self.assertTrue(os.path.exists(f"{self.temp_dir}/sub-01/ses-01/eeg/"))
        # Assert the eeg file was created (edf format)
        self.assertTrue(os.path.exists(
            f"{self.temp_dir}/sub-01/ses-01/eeg/sub-01_ses-01_task-TestTask_run-01_eeg.edf"))
        # Assert the events file was created
        self.assertTrue(os.path.exists(
            f"{self.temp_dir}/sub-01/ses-01/eeg/sub-01_ses-01_task-TestTask_run-01_events.tsv"))
        # Assert the channels file was created
        self.assertTrue(os.path.exists(
            f"{self.temp_dir}/sub-01/ses-01/eeg/sub-01_ses-01_task-TestTask_run-01_channels.tsv"))

    def test_convert_to_bids_raises_error_with_invalid_format(self):
        """Test the convert_to_bids function raises an error with invalid format"""
        with self.assertRaises(ValueError):
            convert_to_bids(
                f"{self.temp_dir}",
                participant_id='01',
                session_id='01',
                run_id='01',
                task_name='TestTask',
                output_dir=self.temp_dir,
                format='invalid_format'
            )

    def test_convert_to_bids_raises_error_with_invalid_data_dir(self):
        """Test the convert_to_bids function raises an error with invalid output directory"""
        with self.assertRaises(FileNotFoundError):
            convert_to_bids(
                'invalid_data_dir',
                participant_id='01',
                session_id='01',
                run_id='01',
                task_name='TestTask',
                output_dir=self.temp_dir
            )

    def test_convert_to_bids_raises_error_with_invalid_line_freq(self):
        """Test the convert_to_bids function raises an error with invalid line frequency"""
        with self.assertRaises(ValueError):
            convert_to_bids(
                f"{self.temp_dir}",
                participant_id='01',
                session_id='01',
                run_id='01',
                task_name='TestTask',
                output_dir=self.temp_dir,
                line_frequency=0
            )


class TestMNEConvert(unittest.TestCase):
    """Test the convert_to_mne function, which converts bcipy RawData into an mne format"""

    def setUp(self):
        """Set up the test case with a temporary directory and sample data"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_rate = 300
        self.filter_settings = {
            'filter_low': 0.5,
            'filter_high': 30,
            'filter_order': 5,
            'notch_filter_frequency': 60,
            'down_sampling_rate': 3
        }
        self.channels = ['timestamp', 'O1', 'O2', 'Pz', 'TRG', 'lsl_timestamp']
        self.raw_data = RawData('SampleDevice', self.sample_rate, self.channels)
        devices.register(devices.DeviceSpec('SampleDevice', channels=self.channels, sample_rate=self.sample_rate))

        # generate 100 random samples of data
        for _ in range(0, 100):
            channel_data = gen_random_data(low=-1000,
                                           high=1000,
                                           channel_count=len(self.channels))
            self.raw_data.append(channel_data)

    def tearDown(self):
        """Remove the temporary directory and its contents after each test"""
        shutil.rmtree(self.temp_dir)

    def test_convert_to_mne_defaults(self):
        """Test the convert_to_mne function with default parameters"""
        data = convert_to_mne(self.raw_data)

        self.assertTrue(len(data) > 0)
        self.assertEqual(data.ch_names, self.channels[1:-2])
        self.assertEqual(data.info['sfreq'], self.sample_rate)

    def test_convert_to_mne_with_channel_map(self):
        """Test the convert_to_mne function with channel mapping"""
        # here we know only three channels are generated, using the channel map let's only use the last one
        channel_map = [0, 0, 1, 0, 0]
        data = convert_to_mne(self.raw_data, channel_map=channel_map)

        self.assertTrue(len(data) > 0)
        self.assertTrue(len(data.ch_names) == 1)  # this is the main assertion!
        self.assertEqual(data.info['sfreq'], self.sample_rate)

    def test_convert_to_mne_with_remove_system_channels(self):
        """Test the convert_to_mne function with system channels removed"""
        data = convert_to_mne(self.raw_data, remove_system_channels=True)

        self.assertTrue(len(data) > 0)
        self.assertEqual(data.ch_names, self.channels[1:-2])
        self.assertEqual(data.info['sfreq'], self.sample_rate)

    def test_convert_to_mne_without_remove_system_channels_throws_error(self):
        """Test the convert_to_mne function with system channels removed raises an error.

        This is due to MNE requiring the channels be EEG. The system channels are not EEG.
        """
        with self.assertRaises(ValueError):
            data = convert_to_mne(self.raw_data, remove_system_channels=False)

            self.assertTrue(len(data) > 0)
            self.assertEqual(data.ch_names, self.channels[1:])
            self.assertEqual(data.info['sfreq'], self.sample_rate)

    def test_convert_to_mne_with_channel_types(self):
        """Test the convert_to_mne function with channel types"""
        channel_types = ['eeg', 'eeg', 'seeg']
        data = convert_to_mne(self.raw_data, channel_types=channel_types)

        self.assertTrue(len(data) > 0)
        self.assertEqual(data.ch_names, self.channels[1:-2])
        self.assertEqual(data.info['sfreq'], self.sample_rate)
        self.assertTrue(data.get_channel_types()[2] == 'seeg')

    def test_convert_to_mne_with_transform(self):
        """Test the convert_to_mne function with a transform"""
        multiplier = 2

        def transform(x, fs):
            return x * multiplier, fs

        data = convert_to_mne(self.raw_data, transform=transform, volts=True)

        self.assertTrue(len(data) > 0)
        self.assertEqual(data.ch_names, self.channels[1:-2])
        self.assertEqual(data.info['sfreq'], self.sample_rate)

        # apply the transform to the first data point and compare to data returned
        expected_first_data_point = self.raw_data.channel_data[0][0] * multiplier
        self.assertTrue(data.get_data()[0][0] == expected_first_data_point)

    def test_convert_to_mne_with_mv_conversion(self):
        """Test the convert_to_mne function with a mv conversion"""
        data = convert_to_mne(self.raw_data, volts=False)

        self.assertTrue(len(data) > 0)
        self.assertEqual(data.ch_names, self.channels[1:-2])
        self.assertEqual(data.info['sfreq'], self.sample_rate)

        # apply the transform to the first data point and compare to data returned
        expected_first_data_point = self.raw_data.channel_data[0][0] * 1e-6
        self.assertTrue(data.get_data()[0][0] == expected_first_data_point)

    def test_convert_to_mne_with_custom_montage(self):
        """Test the convert_to_mne function with a custom montage"""

        # see https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
        # for more information on montages and available defaults
        montage_type = 'biosemi64'
        data = convert_to_mne(self.raw_data, montage=montage_type)

        self.assertTrue(len(data) > 0)
        self.assertEqual(data.ch_names, self.channels[1:-2])
        self.assertEqual(data.info['sfreq'], self.sample_rate)

    def test_convert_to_mne_with_custom_channel_types_length_mismatch(self):
        """Test the convert_to_mne function raises an error when channel types length doesn't match channels"""
        # Not enough channel types
        channel_types = ['eeg', 'eeg']  # Only 2 types for 3 channels

        with self.assertRaises(AssertionError):
            convert_to_mne(self.raw_data, channel_types=channel_types)


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


class TestConvertTobii(unittest.TestCase):

    def test_tobii_to_norm(self):
        """Test the tobii_to_norm function"""
        tobii_data = (0.5, 0.5)  # center of screen in tobii coordinates
        excepted_norm_data = (0, 0)  # center of screen in norm coordinates
        norm_data = tobii_to_norm(tobii_data)
        self.assertEqual(norm_data, excepted_norm_data)

        tobii_data = (0, 0)  # top left of screen in tobii coordinates
        excepted_norm_data = (-1, 1)  # top left of screen in norm coordinates
        norm_data = tobii_to_norm(tobii_data)
        self.assertEqual(norm_data, excepted_norm_data)

        tobii_data = (1, 1)  # bottom right of screen in tobii coordinates
        excepted_norm_data = (1, -1)  # bottom right of screen in norm coordinates
        norm_data = tobii_to_norm(tobii_data)
        self.assertEqual(norm_data, excepted_norm_data)

    def test_tobii_to_norm_raises_error_with_invalid_units(self):
        """Test the tobii_to_norm function raises an error with invalid units"""
        tobii_data = (-1, 1)  # invalid tobii coordinates
        with self.assertRaises(AssertionError):
            tobii_to_norm(tobii_data)

        tobii_data = (1, 11)  # invalid tobii coordinates

        with self.assertRaises(AssertionError):
            tobii_to_norm(tobii_data)

    def test_norm_to_tobii(self):
        """Test the norm_to_tobii function"""
        norm_data = (0, 0)  # center of screen in norm coordinates
        excepted_tobii_data = (0.5, 0.5)  # center of screen in tobii coordinates
        tobii_data = norm_to_tobii(norm_data)
        self.assertEqual(tobii_data, excepted_tobii_data)

        norm_data = (-1, 1)  # top left of screen in norm coordinates
        excepted_tobii_data = (0, 0)  # top left of screen in tobii coordinates
        tobii_data = norm_to_tobii(norm_data)
        self.assertEqual(tobii_data, excepted_tobii_data)

        norm_data = (1, -1)  # bottom right of screen in norm coordinates
        excepted_tobii_data = (1, 1)  # bottom right of screen in tobii coordinates
        tobii_data = norm_to_tobii(norm_data)
        self.assertEqual(tobii_data, excepted_tobii_data)

    def test_norm_to_tobii_raises_error_with_invalid_units(self):
        """Test the norm_to_tobii function raises an error with invalid units"""
        norm_data = (-1.1, 1)
        with self.assertRaises(AssertionError):
            norm_to_tobii(norm_data)

        norm_data = (1, 1.1)
        with self.assertRaises(AssertionError):
            norm_to_tobii(norm_data)


class TestConvertETBIDS(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.trg_data, self.data, self.params = create_bcipy_session_artifacts(self.temp_dir, channels=3)
        self.eyetracking_data = sample_data(
            ch_names=[
                'timestamp',
                'x',
                'y',
                'pupil'],
            daq_type='Gaze',
            sample_rate=60,
            rows=5000)
        devices.register(devices.DeviceSpec('Gaze', channels=['timestamp', 'x', 'y', 'pupil'], sample_rate=60))

        write(self.eyetracking_data, Path(self.temp_dir, 'eyetracker.csv'))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_convert_eyetracking_to_bids_generates_bids_strucutre(self):
        """Test the convert_eyetracking_to_bids function"""
        response = convert_eyetracking_to_bids(
            f"{self.temp_dir}/",
            participant_id='01',
            session_id='01',
            run_id='01',
            task_name='TestTask',
            output_dir=self.temp_dir,
        )
        self.assertTrue(os.path.exists(response))
        # Assert the session directory was created with et
        self.assertTrue(os.path.exists(f"{self.temp_dir}/et/"))
        # Assert the et tsv file was created with the correct name
        self.assertTrue(os.path.exists(f"{self.temp_dir}/et/sub-01_ses-01_task-TestTask_run-01_eyetracking.tsv"))

    def test_convert_eyetracking_to_bids_reflects_participant_id(self):
        """Test the convert_eyetracking_to_bids function with a participant id"""
        response = convert_eyetracking_to_bids(
            f"{self.temp_dir}/",
            participant_id='100',
            session_id='01',
            run_id='01',
            task_name='TestTask',
            output_dir=self.temp_dir,
        )
        self.assertTrue(os.path.exists(response))
        # Assert the et tsv file was created with the correct name
        self.assertTrue(os.path.exists(f"{self.temp_dir}/et/sub-100_ses-01_task-TestTask_run-01_eyetracking.tsv"))

    def test_convert_eyetracking_to_bids_reflects_session_id(self):
        """Test the convert_eyetracking_to_bids function with a session id"""
        response = convert_eyetracking_to_bids(
            f"{self.temp_dir}/",
            participant_id='01',
            session_id='100',
            run_id='01',
            task_name='TestTask',
            output_dir=self.temp_dir,
        )
        self.assertTrue(os.path.exists(response))
        # Assert the et tsv file was created with the correct name
        self.assertTrue(os.path.exists(f"{self.temp_dir}/et/sub-01_ses-100_task-TestTask_run-01_eyetracking.tsv"))

    def test_convert_eyetracking_to_bids_reflects_run_id(self):
        """Test the convert_eyetracking_to_bids function with a run id"""
        response = convert_eyetracking_to_bids(
            f"{self.temp_dir}/",
            participant_id='01',
            session_id='01',
            run_id='100',
            task_name='TestTask',
            output_dir=self.temp_dir,
        )
        self.assertTrue(os.path.exists(response))
        # Assert the et tsv file was created with the correct name
        self.assertTrue(os.path.exists(f"{self.temp_dir}/et/sub-01_ses-01_task-TestTask_run-100_eyetracking.tsv"))

    def test_convert_eyetracking_to_bids_reflects_task_name(self):
        """Test the convert_eyetracking_to_bids function with a task name"""
        response = convert_eyetracking_to_bids(
            f"{self.temp_dir}/",
            participant_id='01',
            session_id='01',
            run_id='01',
            task_name='TestTaskEtc',
            output_dir=self.temp_dir,
        )
        self.assertTrue(os.path.exists(response))
        # Assert the et tsv file was created with the correct name
        self.assertTrue(os.path.exists(f"{self.temp_dir}/et/sub-01_ses-01_task-TestTaskEtc_run-01_eyetracking.tsv"))

    def test_convert_et_raises_error_with_invalid_data_dir(self):
        """Test the convert_eyetracking_to_bids function raises an error with invalid output directory"""
        with self.assertRaises(FileNotFoundError):
            convert_eyetracking_to_bids(
                'invalid_data_dir',
                participant_id='01',
                session_id='01',
                run_id='01',
                task_name='TestTask',
                output_dir=self.temp_dir
            )

    def test_convert_et_raises_error_with_output_dir_not_exist(self):
        """Test the convert_eyetracking_to_bids function raises an error with invalid output directory"""
        with self.assertRaises(FileNotFoundError):
            convert_eyetracking_to_bids(
                f"{self.temp_dir}/",
                participant_id='01',
                session_id='01',
                run_id='01',
                task_name='TestTask',
                output_dir='invalid_output_dir'
            )

    def test_convert_et_raises_error_with_no_data_file(self):
        """Test the convert_eyetracking_to_bids function raises an error with no data file"""
        # remove the csv file
        os.remove(f"{self.temp_dir}/eyetracker.csv")
        with self.assertRaises(FileNotFoundError):
            convert_eyetracking_to_bids(
                f"{self.temp_dir}/",
                participant_id='01',
                session_id='01',
                run_id='01',
                task_name='TestTask',
                output_dir=self.temp_dir,
            )

    def test_convert_et_raises_error_with_multiple_data_files(self):
        """Test the convert_eyetracking_to_bids function raises an error with multiple data files"""
        # create a second data file
        write(self.eyetracking_data, Path(self.temp_dir, 'eyetracker_2.csv'))
        with self.assertRaises(ValueError):
            convert_eyetracking_to_bids(
                f"{self.temp_dir}/",
                participant_id='01',
                session_id='01',
                run_id='01',
                task_name='TestTask',
                output_dir=self.temp_dir,
            )


class TestBIDSToMNE(unittest.TestCase):
    """Tests for the BIDS_to_MNE function."""

    @patch('bcipy.io.convert.os.path.exists')
    @patch('bcipy.io.convert.get_entity_vals')
    @patch('bcipy.io.convert.find_matching_paths')
    @patch('bcipy.io.convert.read_raw_bids')
    def test_successful_conversion(self, mock_read_raw_bids, mock_find_matching_paths,
                                   mock_get_entity_vals, mock_path_exists):
        """Test successful conversion from BIDS to MNE."""
        # Setup mocks
        mock_path_exists.return_value = True
        mock_get_entity_vals.return_value = ['01']

        # Create mock BIDSPath objects
        mock_bids_path1 = MagicMock()
        mock_bids_path1.task = 'RSVPCalibration'
        mock_bids_path2 = MagicMock()
        mock_bids_path2.task = 'RSVPCalibration'
        mock_find_matching_paths.return_value = [mock_bids_path1, mock_bids_path2]

        # Create mock Raw objects that read_raw_bids will return
        mock_raw1 = MagicMock(spec=mne.io.Raw)
        mock_raw2 = MagicMock(spec=mne.io.Raw)
        mock_read_raw_bids.side_effect = [mock_raw1, mock_raw2]

        result = BIDS_to_MNE('/fake/bids/path', task_name='RSVPCalibration')

        self.assertEqual(len(result), 2)
        self.assertIs(result[0], mock_raw1)
        self.assertIs(result[1], mock_raw2)
        mock_path_exists.assert_called_once_with('/fake/bids/path')
        mock_get_entity_vals.assert_called_once_with('/fake/bids/path', 'session')
        mock_find_matching_paths.assert_called_once()
        self.assertEqual(mock_read_raw_bids.call_count, 2)

    @patch('bcipy.io.convert.os.path.exists')
    def test_nonexistent_path(self, mock_path_exists):
        """Test handling of non-existent BIDS path."""
        mock_path_exists.return_value = False

        with self.assertRaises(FileNotFoundError):
            BIDS_to_MNE('/nonexistent/path')

        mock_path_exists.assert_called_once_with('/nonexistent/path')

    @patch('bcipy.io.convert.os.path.exists')
    @patch('bcipy.io.convert.get_entity_vals')
    @patch('bcipy.io.convert.find_matching_paths')
    def test_no_matching_files(self, mock_find_matching_paths, mock_get_entity_vals, mock_path_exists):
        """Test handling of no matching BIDS files."""
        mock_path_exists.return_value = True
        mock_get_entity_vals.return_value = ['01']
        mock_find_matching_paths.return_value = []

        with self.assertRaises(FileNotFoundError):
            BIDS_to_MNE('/fake/bids/path')

        mock_path_exists.assert_called_once_with('/fake/bids/path')
        mock_get_entity_vals.assert_called_once_with('/fake/bids/path', 'session')
        mock_find_matching_paths.assert_called_once()

    @patch('bcipy.io.convert.os.path.exists')
    @patch('bcipy.io.convert.get_entity_vals')
    @patch('bcipy.io.convert.find_matching_paths')
    @patch('bcipy.io.convert.read_raw_bids')
    @patch('bcipy.io.convert.logger')  # Mock the logger to prevent actual logging during tests
    def test_task_filtering(self, mock_logger, mock_read_raw_bids, mock_find_matching_paths,
                            mock_get_entity_vals, mock_path_exists):
        """Test task name filtering."""
        # Setup mocks
        mock_path_exists.return_value = True
        mock_get_entity_vals.return_value = ['01']

        # Create mock BIDSPath objects with different tasks
        mock_bids_path1 = MagicMock()
        mock_bids_path1.task = 'RSVPCalibration'
        mock_bids_path2 = MagicMock()
        mock_bids_path2.task = 'OtherTask'
        mock_find_matching_paths.return_value = [mock_bids_path1, mock_bids_path2]

        # Create mock Raw object that read_raw_bids will return
        mock_raw = MagicMock(spec=mne.io.Raw)
        mock_read_raw_bids.return_value = mock_raw

        # Call function with specific task_name
        result = BIDS_to_MNE('/fake/bids/path', task_name='RSVPCalibration')

        # Assertions
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], mock_raw)
        mock_read_raw_bids.assert_called_once_with(mock_bids_path1)

    @patch('bcipy.io.convert.os.path.exists')
    @patch('bcipy.io.convert.get_entity_vals')
    @patch('bcipy.io.convert.find_matching_paths')
    @patch('bcipy.io.convert.read_raw_bids')
    def test_multiple_sessions(self, mock_read_raw_bids, mock_find_matching_paths,
                               mock_get_entity_vals, mock_path_exists):
        """Test handling multiple sessions."""
        # Setup mocks
        mock_path_exists.return_value = True
        mock_get_entity_vals.return_value = ['01', '02']  # Multiple sessions

        # Create mock BIDSPath objects for different sessions
        mock_bids_path1 = MagicMock()
        mock_bids_path1.task = 'RSVPCalibration'
        mock_bids_path1.session = '01'
        mock_bids_path2 = MagicMock()
        mock_bids_path2.task = 'RSVPCalibration'
        mock_bids_path2.session = '02'
        mock_find_matching_paths.return_value = [mock_bids_path1, mock_bids_path2]

        # Create mock Raw objects
        mock_raw1 = MagicMock(spec=mne.io.Raw)
        mock_raw2 = MagicMock(spec=mne.io.Raw)
        mock_read_raw_bids.side_effect = [mock_raw1, mock_raw2]

        result = BIDS_to_MNE('/fake/bids/path')

        self.assertEqual(len(result), 2)
        mock_get_entity_vals.assert_called_once_with('/fake/bids/path', 'session')
        self.assertEqual(mock_read_raw_bids.call_count, 2)

    @patch('bcipy.io.convert.os.path.exists')
    @patch('bcipy.io.convert.get_entity_vals')
    @patch('bcipy.io.convert.find_matching_paths')
    @patch('bcipy.io.convert.read_raw_bids')
    def test_extension_filtering(self, mock_read_raw_bids, mock_find_matching_paths,
                                 mock_get_entity_vals, mock_path_exists):
        """Test file extension filtering."""
        mock_path_exists.return_value = True
        mock_get_entity_vals.return_value = ['01']

        # Check that the function properly searches for supported extensions
        mock_find_matching_paths.return_value = []

        with self.assertRaises(FileNotFoundError):
            BIDS_to_MNE('/fake/bids/path')

        # Verify extensions were specified in the find_matching_paths call
        call_args = mock_find_matching_paths.call_args[1]
        self.assertIn('extensions', call_args)
        self.assertTrue(len(call_args['extensions']) > 0)
        self.assertIn('.vhdr', call_args['extensions'])  # BrainVision format
        self.assertIn('.edf', call_args['extensions'])   # EDF format

    @patch('bcipy.io.convert.os.path.exists')
    @patch('bcipy.io.convert.get_entity_vals')
    @patch('bcipy.io.convert.find_matching_paths')
    @patch('bcipy.io.convert.read_raw_bids')
    @patch('bcipy.io.convert.logger')
    def test_debug_logging(self, mock_logger, mock_read_raw_bids, mock_find_matching_paths,
                           mock_get_entity_vals, mock_path_exists):
        """Test debug logging when skipping files due to task name mismatch."""
        mock_path_exists.return_value = True
        mock_get_entity_vals.return_value = ['01']

        # Create mock BIDSPath objects with different tasks
        mock_bids_path1 = MagicMock()
        mock_bids_path1.task = 'RSVPCalibration'
        mock_bids_path2 = MagicMock()
        mock_bids_path2.task = 'OtherTask'
        mock_find_matching_paths.return_value = [mock_bids_path1, mock_bids_path2]

        # Mock the debug logging method specifically
        mock_logger.debug = MagicMock()

        # Create mock Raw object
        mock_raw = MagicMock(spec=mne.io.Raw)
        mock_read_raw_bids.return_value = mock_raw

        BIDS_to_MNE('/fake/bids/path', task_name='RSVPCalibration')

        # Check if debug was called for skipping a file
        self.assertTrue(any('Skipping' in str(args) for args, _ in mock_logger.debug.call_args_list))


if __name__ == '__main__':
    unittest.main()
