"""Unit tests for ContentType"""
import unittest

from bcipy.acquisition.exceptions import UnsupportedContentType
from bcipy.acquisition.multimodal import ContentType


class TestContentType(unittest.TestCase):
    """Tests for multimodal ContentType enum."""

    def test_value(self):
        """Test that values are autoincremented"""
        self.assertEqual(1, ContentType.EEG.value)
        self.assertEqual(2, ContentType.EYETRACKER.value)

    def test_constructor(self):
        """Test retrieving an enum by value"""
        self.assertEqual(ContentType.EEG, ContentType('EEG'))
        self.assertEqual(ContentType.EEG, ContentType('eeg'))
        self.assertEqual(ContentType.EEG, ContentType('Eeg'))

        self.assertEqual(ContentType.EYETRACKER, ContentType('Eyetracker'))

    def test_synonyms(self):
        """Test that synonyms can be used for some values"""
        self.assertEqual(ContentType.EYETRACKER, ContentType('Gaze'))
        self.assertEqual(ContentType.EYETRACKER, ContentType('Eye_tracker'))

    def test_unsupported_type(self):
        """Test that an unsupported type raises an exception"""
        with self.assertRaises(UnsupportedContentType):
            ContentType('FooBar')


if __name__ == '__main__':
    unittest.main()
