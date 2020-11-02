"""Tests for data conversion related functionality."""
import shutil
import tempfile
import unittest

class TestConvert(unittest.TestCase):
    """Tests for data format conversions."""

    def setUp(self):
        """Override; set up the needed path for load functions."""

        self.parameters_location = 'bcipy/parameters/parameters.json'
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Override"""
        shutil.rmtree(self.temp_dir)