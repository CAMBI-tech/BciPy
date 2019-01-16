import unittest

from bcipy.helpers.acquisition_related import analysis_channel_names_by_pos


class TestAcquisitionRelated(unittest.TestCase):
    """Test helper methods in the acquisition-related code"""

    def testChannelNamesByPos(self):
        names = analysis_channel_names_by_pos(['a', 'b', 'c', 'd'],
                                              [0, 1, 0, 1])
        self.assertEqual('b', names[0])
        self.assertEqual('d', names[1])