import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from bcipy.simulator.helpers.data_engine import (ExtractedExperimentData,
                                                 RawDataEngine)


class TestRawDataEngine(unittest.TestCase):
    """Tests for RawDataEngine"""

    def setUp(self) -> None:
        """Setup common state"""

        self.data_processor = Mock()
        self.data_processor.process = Mock()
        self.parameters = Mock()

    def test_init(self):
        """"Test initialization"""
        data_engine = RawDataEngine(source_dirs=[],
                                    parameters=self.parameters,
                                    data_processor=self.data_processor)

        self.assertEqual(self.parameters, data_engine.parameters)

    def test_single_data_source(self):
        """Test loading data from a single directory."""
        RawDataEngine(source_dirs=['data-dir1'],
                      parameters=self.parameters,
                      data_processor=self.data_processor)
        self.data_processor.process.assert_called_once_with(
            'data-dir1', self.parameters)

    def test_multiple_sources(self):
        """Test loading data from multiple directories."""
        RawDataEngine(source_dirs=['data-dir1', 'data-dir2', 'data-dir3'],
                      parameters=self.parameters,
                      data_processor=self.data_processor)
        self.assertEqual(self.data_processor.process.call_count, 3)
        # last call
        self.data_processor.process.assert_called_with('data-dir3',
                                                       self.parameters)

    def test_transform(self):
        """Test the data transformation method"""
        # Setup context
        # mock the experiment data; single inquiry
        mock_data = ExtractedExperimentData(
            source_dir=Path('data-dir1'),
            inquiries=np.zeros((7, 1, 692)),
            trials=np.zeros((7, 10, 74)),
            labels=np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]),
            inquiry_timing=[[150, 185, 220, 255, 291, 326, 361, 397, 432,
                             467]],
            decoded_triggers=([
                'nontarget', 'nontarget', 'nontarget', 'nontarget',
                'nontarget', 'target', 'nontarget', 'nontarget', 'nontarget',
                'nontarget'
            ], [
                9.730993399999988, 9.966709099999662, 10.201713299999938,
                10.436247999999978, 10.672083899999961, 10.90684519999968,
                11.141479799999615, 11.377909999999702, 11.614850199999637,
                11.849502299999585
            ], ['G', 'C', 'D', 'B', 'I', 'A', 'H', '<', 'E', 'F']),
            trials_per_inquiry=10)
        self.data_processor.process = Mock(return_value=mock_data)
        engine = RawDataEngine(source_dirs=['data-dir1'],
                               parameters=self.parameters,
                               data_processor=self.data_processor)
        data = engine.trials_df

        self.assertEqual(10, len(data))

        self.assertEqual(data.iloc[0].symbol, 'G')
        self.assertEqual(data.iloc[0].inquiry_n, 0)
        self.assertEqual(data.iloc[0].inquiry_pos, 1)
        self.assertEqual(data.iloc[0].target, 0)

        self.assertEqual(data.iloc[len(data) - 1].symbol, 'F')
        self.assertEqual(data.iloc[len(data) - 1].inquiry_n, 0)
        self.assertEqual(data.iloc[len(data) - 1].inquiry_pos, 10)
        self.assertEqual(data.iloc[len(data) - 1].target, 0)


if __name__ == '__main__':
    unittest.main()
