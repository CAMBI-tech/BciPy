"""Tests for the Preferences"""
import unittest
import shutil
import tempfile
from pathlib import Path
from bcipy.preferences import Preferences


class TestPreferences(unittest.TestCase):
    """Tests for the Preferences class."""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init_without_existing_file(self):
        """Test initialization"""
        path = Path(self.temp_dir, 'test_prefs.json')
        prefs = Preferences(filename=path)
        self.assertFalse(path.is_file())
        self.assertEqual(prefs.entries, {})
        self.assertTrue(
            len(prefs.last_directory) > 0,
            "Default properties should be available.")

    def test_save(self):
        """Test saving to disk"""
        path = Path(self.temp_dir, 'test_prefs')
        prefs = Preferences(filename=path)
        self.assertFalse(path.is_file())
        prefs.save()
        self.assertTrue(path.is_file())

    def test_get(self):
        """Test getting a property."""
        prefs = Preferences(filename=Path(self.temp_dir, 'test_prefs'))
        prefs.entries = {'my_pref': 'my_value'}
        self.assertEqual(prefs.get('my_pref'), 'my_value')

    def test_set(self):
        """Setting a preference should save it to disk by default."""
        path = Path(self.temp_dir, 'test_prefs')
        prefs = Preferences(filename=path)
        self.assertFalse(path.is_file())

        value = 'my_pref_value'
        prefs.set('my_pref', value)
        self.assertTrue(path.is_file())

    def test_set_without_persisting(self):
        """Should be able to set a property without persisting."""
        path = Path(self.temp_dir, 'test_prefs')
        prefs = Preferences(filename=path)
        self.assertFalse(path.is_file())

        value = 'my_pref_value'
        prefs.set('my_pref', value, persist=False)
        self.assertFalse(path.is_file())

    def test_set_attribute_preferences(self):
        """Setting preferences defined as class attributes should save."""
        path = Path(self.temp_dir, 'test_prefs')
        prefs = Preferences(filename=path)
        self.assertFalse(path.is_file())

        prefs.last_directory = self.temp_dir

        self.assertTrue(path.is_file())

    def test_load(self):
        """Test loading from a file."""
        path = Path(self.temp_dir, 'test_prefs')
        prefs = Preferences(filename=path)

        prefs.set('pref_int', 1234, persist=False)
        prefs.set('pref_str', 'my_value', persist=False)
        prefs.set('pref_bool', True, persist=False)
        prefs.save()

        loaded_prefs = Preferences(filename=path)
        self.assertEqual(loaded_prefs.get('pref_int'), 1234)
        self.assertEqual(loaded_prefs.get('pref_str'), 'my_value')
        self.assertTrue(loaded_prefs.get('pref_bool'))


if __name__ == '__main__':
    unittest.main()
