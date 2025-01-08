"""Tests for simulator logging and directory functions."""
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

from bcipy.simulator.util.artifact import configure_logger, remove_handlers


class TestSimArtifact(unittest.TestCase):
    """Test for simulator artifact functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_configure_named_logger(self):
        """Test configure named logger"""

        configure_logger(self.temp_dir,
                         file_name='test_1a.log',
                         logger_name='TEST_LOGGER',
                         use_stdout=True)

        log = logging.getLogger('TEST_LOGGER')
        self.assertEqual(2, len(log.handlers))
        self.assertEqual('TEST_LOGGER', log.name)
        log.info("testing 1 2 3")
        self.assertTrue(Path(self.temp_dir, "test_1a.log").exists())

        self.assertFalse(Path(self.temp_dir, "test_1b.log").exists())
        configure_logger(self.temp_dir,
                         file_name='test_1b.log',
                         logger_name='TEST_LOGGER',
                         use_stdout=True)

        log.info("testing 4 5 6")
        self.assertTrue(Path(self.temp_dir, "test_1b.log").exists())
        self.assertEqual(2, len(log.handlers))
        remove_handlers(log)

    def test_configure_named_logger_without_stdout(self):
        """Test the named logger without logging to stdout."""

        configure_logger(self.temp_dir,
                         file_name='test_2.log',
                         logger_name='TEST_LOGGER',
                         use_stdout=False)

        log = logging.getLogger('TEST_LOGGER')
        self.assertEqual(1, len(log.handlers))
        self.assertTrue(isinstance(log.handlers[0], logging.FileHandler))
        remove_handlers(log)

    def test_configure_root_logger(self):
        """Test configure root logger"""
        configure_logger(self.temp_dir,
                         file_name='test_root_logger.log',
                         logger_name=None,
                         use_stdout=False)
        log = logging.getLogger()
        self.assertEqual(1, len(log.handlers))
        self.assertTrue(isinstance(log.handlers[0], logging.FileHandler))
        remove_handlers(log)


if __name__ == '__main__':
    unittest.main()
