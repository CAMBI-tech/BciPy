"""Tests for session helpers."""

import filecmp
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from bcipy.helpers.session import session_csv, session_data, session_db


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
        self.assertEquals(data["total_number_series"], 2)
        self.assertTrue("A" in data["series"]["1"]["0"]["likelihood"])
        self.assertTrue("B" in data["series"]["1"]["0"]["likelihood"])

    def test_session_evidence(self):
        """Test functionality to transform session.json file to a csv
        summarizing the evidence."""

        db_name = str(Path(self.temp_dir, 'mock_session.db'))
        csv_name = str(Path(self.temp_dir, 'mock_session.csv'))

        session_db(data_dir=self.data_dir, db_name=db_name)
        session_csv(db_name=db_name, csv_name=csv_name)

        generated = Path(self.temp_dir, 'mock_session.csv')
        expected = Path(self.data_dir, "session.csv")

        self.assertTrue(generated.is_file())
        self.assertTrue(filecmp.cmp(expected, generated))
