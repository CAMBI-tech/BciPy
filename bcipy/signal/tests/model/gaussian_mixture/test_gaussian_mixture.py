import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bcipy.signal.model import GaussianProcess


class ModelSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.tmp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)


class TestGaussianMixtureInternals(ModelSetup):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass

    def setUp(self):
        np.random.seed(0)
        self.model = GaussianProcess()

    def test_predict(self):
        ...


if __name__ == "__main__":
    unittest.main()
