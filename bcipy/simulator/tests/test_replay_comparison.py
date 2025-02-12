import os
import shutil
import tempfile
import unittest
from pathlib import Path

from bcipy.simulator.util.replay_comparison import (load_eeg_records,
                                                    prepare_data)


class ReplayComparisonTest(unittest.TestCase):
    """Tests comparing evidence from identical sessions."""

    def setUp(self):
        """Override; set up the needed path for load functions."""
        self.data_dir = f"{os.path.dirname(__file__)}/resources/"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override"""
        shutil.rmtree(self.temp_dir)

    def test_load_eeg_evidence(self):
        """Test loading evidence"""

        path = Path(self.data_dir, "session.json")
        records = load_eeg_records(str(path), is_original_model=False)

        target_eeg = [
            record.response_value for record in records
            if record.which_model == "new_target"
        ]
        nontarget_eeg = [
            record.response_value for record in records
            if record.which_model == "new_nontarget"
        ]
        msg = "Target was not shown in first two inquiries but had a high response in the third."
        self.assertEqual([1.0, 1.0, 100.0], target_eeg, msg)
        num_symbols = 28
        num_inquiries = 3
        self.assertEqual((num_symbols - 1) * num_inquiries, len(nontarget_eeg))

    def test_prepare_data(self):
        """Test data prep"""
        records = prepare_data(sim_dir=self.data_dir, data_folders=[])
        self.assertTrue(
            all(record.which_model in ["new_target", "new_nontarget"]
                for record in records))


if __name__ == '__main__':
    unittest.main()
