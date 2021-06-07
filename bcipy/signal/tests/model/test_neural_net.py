
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


class EegClassifierBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        cls.alp = alphabet()
        cls.cfg = Config(use_early_stop=True, epochs=50, seed=0, output_dir=cls.tmp_dir)
        # Random data - model cannot generalize to test data
        cls.x, cls.y = get_fake_data(N=cls.cfg.batch_size, channels=cls.cfg.n_channels,
                                     classes=cls.cfg.n_classes, length=cls.cfg.length)
        cls.model = EegClassifierModel(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)


class TestEegClassifierModelAfterFit(EegClassifierBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model.fit(cls.x, cls.y)

    @classmethod
    def tearDownClass(cls):
        return super().tearDownClass()

    def setUp(self):
        seed_everything(0)

    def test_evaluate(self):
        # Evaluate on training data
        report = self.model.evaluate(self.x, self.y)
        self.assertTrue(report.auroc >= 0.98)

    def test_predict_single(self):
        self.model.fit(self.x, self.y)
        inquiry_len = self.cfg.n_classes - 1
        result = self.model.predict(data=self.x[0], inquiry=self.alp[:inquiry_len], symbol_set=self.alp).squeeze()
        self.assertTrue(len(result) == len(self.alp))

    def test_save_load(self):
        new_x, new_y = get_fake_data(N=self.cfg.batch_size, channels=self.cfg.n_channels,
                                     classes=self.cfg.n_classes, length=self.cfg.length)
        inquiry_len = self.cfg.n_classes - 1
        predictions_before = self.model.predict(
            data=new_x[0], inquiry=self.alp[:inquiry_len], symbol_set=self.alp).squeeze()
        self.model.save(self.tmp_dir / "checkpoint")
        self.model.load(self.tmp_dir / "checkpoint")
        predictions_after = self.model.predict(
            data=new_x[0], inquiry=self.alp[:inquiry_len], symbol_set=self.alp).squeeze()

        breakpoint()
        self.assertTrue(np.allclose(predictions_before, predictions_after))


class TestEegClassifierModelBeforeFit(EegClassifierBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        return super().tearDownClass()

    def test_cannot_evaluate_before_fit(self):
        with self.assertRaises(SignalException):
            self.model.evaluate(self.x, self.y)


if __name__ == "__main__":
    unittest.main()
