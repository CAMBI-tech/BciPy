import unittest
from pathlib import Path
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
from bcipy.helpers.task import alphabet

"""Unit tests for PCA/RDA/KDE model"""
from bcipy.signal.model import PcaRdaKdeModel, ModelEvaluationReport

expected_output_folder = Path(__file__).absolute().parent / "unit_test_expected_output"


class TestRefactoredPcaRdaKdeModel(unittest.TestCase):
    """
    Note - When the code is in a known & working state, generate "baseline" images using:
    `pytest --mpl-generate-path=bcipy/signal/tests/model/unit_test_expected_output -k TestPcaRdaKdeModel`
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

        # Specify data dimensions
        cls.dim_x = 10
        cls.num_channel = 8
        cls.num_x_pos = 200
        cls.num_x_neg = 200

        # Generate Gaussian random data
        cls.pos_mean, cls.pos_std = 0, 0.5
        cls.neg_mean, cls.neg_std = 1, 0.5
        x_pos = cls.pos_mean + cls.pos_std * np.random.randn(cls.num_channel, cls.num_x_pos, cls.dim_x)
        x_neg = cls.neg_mean + cls.neg_std * np.random.randn(cls.num_channel, cls.num_x_neg, cls.dim_x)
        y_pos = np.ones(cls.num_x_pos)
        y_neg = np.zeros(cls.num_x_neg)

        # Stack and permute data
        x = np.concatenate([x_pos, x_neg], 1)
        y = np.concatenate([y_pos, y_neg], 0)
        permutation = np.random.permutation(cls.num_x_pos + cls.num_x_neg)
        x = x[:, permutation, :]
        y = y[permutation]

        cls.x = x
        cls.y = y

        cls.model = PcaRdaKdeModel(k_folds=10)
        cls.model.fit(cls.x, cls.y)

        cls.tmp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    @pytest.mark.mpl_image_compare(
        baseline_dir=expected_output_folder, filename="test_inference.expected.png", remove_text=True
    )
    def test_fit_predict(self):
        """Fit and then predict"""
        alp = alphabet()

        # Create test items that resemble the fake training data
        num_x_p = 1
        num_x_n = 9

        x_test_pos = self.pos_mean + self.pos_std * np.random.randn(self.num_channel, num_x_p, self.dim_x)
        x_test_neg = self.neg_mean + self.neg_std * np.random.randn(self.num_channel, num_x_n, self.dim_x)
        x_test = np.concatenate((x_test_pos, x_test_neg), 1)  # Target letter is first

        letters = alp[10 : 10 + num_x_p + num_x_n]  # Target letter is K

        lik_r = self.model.predict(data=x_test, inquiry=letters, symbol_set=alp)

        fig, ax = plt.subplots()
        ax.plot(np.arange(len(alp)), lik_r, "ro")
        ax.set_xticks(np.arange(len(alp)))
        ax.set_xticklabels(alp)
        ax.set_yticks(np.arange(0, 101, 10))
        return fig

    def test_evaluate(self):
        model_report = self.model.evaluate(self.x, self.y)
        self.assertEqual(model_report, ModelEvaluationReport(1.0))

    def test_save_load(self):
        n_trial = 15
        symbol_set = alphabet()
        inquiry = symbol_set[:n_trial]
        data = np.random.randn(self.num_channel, n_trial, self.dim_x)
        output_before = self.model.predict(data=data, inquiry=inquiry, symbol_set=symbol_set)

        checkpoint_path = self.tmp_dir / "model.pkl"
        self.model.save(checkpoint_path)
        other_model = PcaRdaKdeModel(k_folds=self.model.k_folds)
        other_model.load(checkpoint_path)
        output_after = other_model.predict(data=data, inquiry=inquiry, symbol_set=symbol_set)

        self.assertTrue(np.allclose(output_before, output_after))


if __name__ == "__main__":
    unittest.main()
