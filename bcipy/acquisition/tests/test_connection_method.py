"""Tests for the Connection Methods"""
import unittest

from bcipy.acquisition.connection_method import ConnectionMethod


class TestConnectionMethod(unittest.TestCase):
    """Tests for ConnectionMethods."""

    def test_list(self):
        """Should be able to list available options."""
        self.assertEqual(2, len(ConnectionMethod.list()))
        self.assertTrue('TCP' in ConnectionMethod.list())
        self.assertTrue('LSL' in ConnectionMethod.list())

    def test_search_by_name(self):
        """Should be able to find a ConnectionMethod by name."""
        self.assertEqual(ConnectionMethod.LSL, ConnectionMethod.by_name('LSL'))
        self.assertEqual(ConnectionMethod.TCP, ConnectionMethod.by_name('TCP'))
