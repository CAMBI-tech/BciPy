"""Tests for session helpers."""

import filecmp
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from bcipy.config import SESSION_DATA_FILENAME
from bcipy.helpers.session import read_session, session_csv, session_data


class TestSessionHelper(unittest.TestCase):
    """Tests for session helper."""

    def setUp(self):
        """Override; set up the needed path for load functions."""
        self.data_dir = f"{os.path.dirname(__file__)}/resources/mock_session/"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override"""
        shutil.rmtree(self.temp_dir)

    def test_session_data(self):
        """Test session_data method outputs letters with evidence."""
        data = session_data(data_dir=self.data_dir)
        self.assertEqual(data["total_number_series"], 2)
        self.assertTrue("A" in data["series"]["1"]["0"]["likelihood"])
        self.assertTrue("B" in data["series"]["1"]["0"]["likelihood"])

    def test_read_session(self):
        """Test reading data from json."""
        session = read_session(Path(self.data_dir, SESSION_DATA_FILENAME))
        self.assertEqual(session.total_number_series, 2)

    def test_session_csv(self):
        """Test functionality to transform session.json file to a csv
        summarizing the evidence."""

        session = read_session(Path(self.data_dir, SESSION_DATA_FILENAME))
        csv_name = str(Path(self.temp_dir, 'mock_session.csv'))

        session_csv(session, csv_file=csv_name)

        generated = Path(self.temp_dir, 'mock_session.csv')
        expected = Path(self.data_dir, "session.csv")

        self.assertTrue(generated.is_file())
        self.assertTrue(filecmp.cmp(expected, generated))
