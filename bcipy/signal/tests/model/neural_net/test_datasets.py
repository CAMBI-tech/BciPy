import unittest
from bcipy.signal.model.neural_net.data import EEGDataset, FolderDataset


class TestEegDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = EEGDataset()

    def test_something(self):
        raise NotImplementedError()


class TestFolderDataset(unittest.TestCase):
    def setUp(self):
        # Create folder with fake datasets
        self.dataset = FolderDataset()

    def test_something(self):
        raise NotImplementedError()
