import shutil
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import norm

from bcipy.helpers.symbols import alphabet
from bcipy.signal.model import ModelEvaluationReport, PcaRdaKdeModel
from bcipy.signal.model.classifier import RegularizedDiscriminantAnalysis
from bcipy.signal.model.cross_validation import cross_validation
from bcipy.signal.model.density_estimation import KernelDensityEstimate
from bcipy.signal.model.dimensionality_reduction import ChannelWisePrincipalComponentAnalysis
from bcipy.signal.model.pipeline import Pipeline
from bcipy.helpers.exceptions import SignalException

expected_output_folder = Path(__file__).absolute().parent.parent / "unit_test_expected_output"


class ModelSetup(unittest.TestCase):
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

        cls.tmp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)


class TestPcaRdaKdeModelInternals(ModelSetup):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass

    def setUp(self):
        np.random.seed(0)
        self.model = PcaRdaKdeModel(k_folds=10)
        self.model.fit(self.x, self.y)
        np.random.seed(0)

    def test_pca(self):
        # .fit() then .transform() should match .fit_transform()
        pca = ChannelWisePrincipalComponentAnalysis(n_components=0.9, num_ch=self.num_channel)
        pca.fit(self.x)
        x_reduced = pca.transform(self.x)
        x_reduced_2 = pca.fit_transform(self.x)
        self.assertTrue(np.allclose(x_reduced, x_reduced_2))

        # Output values should be correct
        expected = np.load(expected_output_folder / "test_pca.expected.npy")
        self.assertTrue(np.allclose(x_reduced, expected))

    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(
        baseline_dir=expected_output_folder, filename="test_kde_plot.expected.png", remove_text=True
    )
    def test_kde_plot(self):
        """
        Notes:
        - TODO - can this test be re-written to use some data from self.setUp()?
          (Not vital, but would make this file shorter)
        """
        # generate some dummy data
        n = 100
        x = np.concatenate((np.random.normal(0, 1, int(0.3 * n)), np.random.normal(5, 1, int(0.7 * n))))[:, np.newaxis]

        # append 0 label to all data as we are interested in a single class case
        y = np.zeros(x.shape)

        # a subset of domain of the random variable x
        x_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

        # generate a dummy density function to sample data from
        true_dens = 0.3 * norm(0, 1).pdf(x_plot[:, 0]) + 0.7 * norm(5, 1).pdf(x_plot[:, 0])

        fig, ax = plt.subplots()
        ax.fill(x_plot[:, 0], true_dens, fc="black", alpha=0.2, label="input distribution")

        # try different kernels and show how the look like
        for kernel in ["gaussian", "tophat", "epanechnikov"]:
            kde = KernelDensityEstimate(kernel=kernel, scores=x, num_cls=1)
            kde.fit(x, y)
            log_dens = kde.list_den_est[0].score_samples(x_plot)
            ax.plot(x_plot[:, 0], np.exp(log_dens), "-", label=f"kernel = '{kernel}'")

        ax.plot(x[:, 0], -0.005 - 0.01 * np.random.random(x.shape[0]), "+k")

        ax.set_xlim(-4, 9)
        ax.set_ylim(-0.02, 0.4)
        return fig

    def test_kde_values(self):
        pca = ChannelWisePrincipalComponentAnalysis(n_components=0.9, num_ch=self.num_channel)
        rda = RegularizedDiscriminantAnalysis()
        kde = KernelDensityEstimate()

        pipeline = Pipeline([pca, rda, kde])

        # .fit() followed by .transform() should match .fit_transform()
        z = pipeline.fit_transform(self.x, self.y)
        pipeline.fit(self.x, self.y)
        z_2 = pipeline.transform(self.x)
        self.assertTrue(np.allclose(z, z_2))

        # output values should be correct
        expected = np.load(expected_output_folder / "test_kde_values.expected.npy")
        self.assertTrue(np.allclose(z, expected))

    def test_cv(self):
        """
        Notes:
        - cross validation explicitly modifies pipeline[1], so we need a PCA step.
        - The purpose of cross_validation() is to find optimal values of lambda and gamma for the RDA model
          before fitting it - it is not clear how sensitive this test is to changes in the code
          or input data, so this may be a weak test of cross_validation().
        """
        pca = ChannelWisePrincipalComponentAnalysis(n_components=0.9, num_ch=self.num_channel)
        rda = RegularizedDiscriminantAnalysis()

        pipeline = Pipeline([pca, rda])
        lam, gam = cross_validation(self.x, self.y, pipeline)

        self.assertAlmostEqual(lam, 0.9)
        self.assertAlmostEqual(gam, 0.1)

    def test_rda(self):
        pca = ChannelWisePrincipalComponentAnalysis(n_components=0.9, num_ch=self.num_channel)
        rda = RegularizedDiscriminantAnalysis()

        pipeline = Pipeline([pca, rda])

        # .fit() followed by .transform() should match .fit_transform()
        z = pipeline.fit_transform(self.x, self.y)
        pipeline.fit(self.x, self.y)
        z_2 = pipeline.transform(self.x)
        self.assertTrue(np.allclose(z, z_2))

        # output values should be correct
        expected = np.load(expected_output_folder / "test_rda.expected.npy")
        self.assertTrue(np.allclose(z, expected))


class TestPcaRdaKdeModelExternals(ModelSetup):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass

    def setUp(self):
        np.random.seed(0)
        self.model = PcaRdaKdeModel(k_folds=10)
        self.model.fit(self.x, self.y)
        np.random.seed(0)

    @pytest.mark.slow
    @pytest.mark.mpl_image_compare(
        baseline_dir=expected_output_folder,
        filename="test_inference.expected.png",
        remove_text=True,
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

        letters = alp[10: 10 + num_x_p + num_x_n]  # Target letter is K

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

    def test_predict_before_fit(self):
        model = PcaRdaKdeModel(k_folds=10)
        with self.assertRaises(SignalException):
            model.predict(self.x, inquiry=["A"], symbol_set=alphabet())

    def test_evaluate_before_fit(self):
        model = PcaRdaKdeModel(k_folds=10)
        with self.assertRaises(SignalException):
            model.evaluate(self.x, self.y)


if __name__ == "__main__":
    unittest.main()
