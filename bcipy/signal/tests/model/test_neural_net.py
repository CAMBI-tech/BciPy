
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from bcipy.helpers.task import alphabet
from bcipy.signal.exceptions import SignalException
from bcipy.signal.model.neural_net import EegClassifierModel
from bcipy.signal.model.neural_net.config import Config, seed_everything
from bcipy.signal.model.neural_net.data.data_utils import get_fake_data


class TestEegClassifierModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        cls.cfg = Config(use_early_stop=True, epochs=50, seed=0, output_dir=cls.tmp_dir)
        # Random data - model cannot generalize to test data
        cls.x, cls.y = get_fake_data(N=cls.cfg.batch_size, channels=cls.cfg.n_channels,
                                     classes=cls.cfg.n_classes, length=cls.cfg.length)
        cls.model = EegClassifierModel(cls.cfg)

    @ classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def setUp(self):
        seed_everything(0)

    def test_fit(self):
        self.model.fit(self.x, self.y)

    def test_cannot_evaluate_before_fit(self):
        with self.assertRaises(SignalException):
            self.model.evaluate(self.x, self.y)

    def test_evaluate(self):
        self.model.fit(self.x, self.y)
        # Evaluate on training data
        report = self.model.evaluate(self.x, self.y)
        self.assertTrue(report.auroc >= 0.98)

    def test_predict(self):
        self.model.fit(self.x, self.y)
        alp = alphabet()
        inquiry_len = self.cfg.n_classes - 1
        result = self.model.predict(data=self.x, inquiry=alp[:inquiry_len], symbol_set=alp)
        breakpoint()
        raise NotImplementedError()

    def test_save(self):
        raise NotImplementedError()

    def test_load(self):
        raise NotImplementedError()


if __name__ == "__main__":
    unittest.main()
