import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import iqr, norm

"""Unit tests for PCA/RDA/KDE model"""
from bcipy.signal.model.inference import inference
from bcipy.signal.model.mach_learning.classifier import RegularizedDiscriminantAnalysis
from bcipy.signal.model.mach_learning.cross_validation import cross_validation
from bcipy.signal.model.mach_learning.density_estimation import KernelDensityEstimate
from bcipy.signal.model.mach_learning.dimensionality_reduction import ChannelWisePrincipalComponentAnalysis
from bcipy.signal.model.mach_learning.pipeline import Pipeline
from bcipy.signal.model.mach_learning.train_model import train_pca_rda_kde_model

from bcipy.helpers.task import alphabet

expected_output_folder = Path(__file__).absolute().parent / "unit_test_expected_output"


class TestPcaRdaKdeModel(unittest.TestCase):
    """
    Note - When the code is in a known & working state, generate "baseline" images using:
    `pytest --mpl-generate-path=bcipy/signal/tests/model/unit_test_expected_output -k TestPcaRdaKdeModel`
    """

    def setUp(self):
        np.random.seed(0)

        # Specify data dimensions
        self.dim_x = 10
        self.num_channel = 8
        self.num_x_pos = 200
        self.num_x_neg = 200

        # Generate Gaussian random data
        self.pos_mean, self.pos_std = 0, 0.5
        self.neg_mean, self.neg_std = 1, 0.5
        x_pos = self.pos_mean + self.pos_std * np.random.randn(self.num_channel, self.num_x_pos, self.dim_x)
        x_neg = self.neg_mean + self.neg_std * np.random.randn(self.num_channel, self.num_x_neg, self.dim_x)
        y_pos = np.ones(self.num_x_pos)
        y_neg = np.zeros(self.num_x_neg)

        # Stack and permute data
        x = np.concatenate([x_pos, x_neg], 1)
        y = np.concatenate([y_pos, y_neg], 0)
        permutation = np.random.permutation(self.num_x_pos + self.num_x_neg)
        x = x[:, permutation, :]
        y = y[permutation]

        self.x = x
        self.y = y

    @pytest.mark.mpl_image_compare(
        baseline_dir=expected_output_folder, filename="test_inference.expected.png", remove_text=True
    )
    def test_inference(self):
        model, _ = train_pca_rda_kde_model(self.x, self.y, k_folds=10)

        alp = alphabet()

        # Create test items that resemble the fake training data, testing inference
        num_x_p = 1
        num_x_n = 9

        x_test_pos = self.pos_mean + self.pos_std * np.random.randn(self.num_channel, num_x_p, self.dim_x)
        x_test_neg = self.neg_mean + self.neg_std * np.random.randn(self.num_channel, num_x_n, self.dim_x)
        x_test = np.concatenate((x_test_neg, x_test_pos), 1)

        idx_let = np.random.permutation(len(alp))
        letters = [alp[i] for i in idx_let[0 : (num_x_p + num_x_n)]]

        lik_r = inference(x=x_test, targets=letters, model=model, alphabet=alp)

        fig, ax = plt.subplots()
        ax.plot(np.array(list(range(len(alp)))), lik_r, "ro")
        ax.set_xticks(np.arange(len(alp)))
        ax.set_xticklabels(alp)
        return fig

    def test_pca(self):
        var_tol = 0.95

        # .fit() then .transform() should match .fit_transform()
        pca = ChannelWisePrincipalComponentAnalysis(num_ch=self.num_channel)
        pca.fit(self.x, var_tol=var_tol)
        x_reduced = pca.transform(self.x)
        x_reduced_2 = pca.fit_transform(self.x, var_tol=var_tol)
        self.assertTrue(np.allclose(x_reduced, x_reduced_2))

        # Output values should be correct
        expected = np.load(expected_output_folder / "test_pca.expected.npy")
        self.assertTrue(np.allclose(x_reduced, expected))

    @pytest.mark.mpl_image_compare(
        baseline_dir=expected_output_folder, filename="test_kde_plot.expected.png", remove_text=True
    )
    def test_kde_plot(self):
        """
        Notes:
        - TODO - can this test be re-written to use some data from self.setUp()?
          (Not vital, but would make this file shorter)
        """
        n = 100

        # generate some dummy data
        x = np.concatenate((np.random.normal(0, 1, int(0.3 * n)), np.random.normal(5, 1, int(0.7 * n))))[:, np.newaxis]

        # append 0 label to all data as we are interested in a single class case
        y = np.zeros(x.shape)

        # a subset of domain of the random variable x
        x_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

        # generate a dummy density function to sample data from
        true_dens = 0.3 * norm(0, 1).pdf(x_plot[:, 0]) + 0.7 * norm(5, 1).pdf(x_plot[:, 0])

        fig, ax = plt.subplots()
        ax.fill(x_plot[:, 0], true_dens, fc="black", alpha=0.2, label="input distribution")

        # thumb up rule for bandwidth selection
        bandwidth = 1.06 * min(np.std(x), iqr(x) / 1.34) * np.power(x.shape[0], -0.2)

        # try different kernels and show how the look like
        for kernel in ["gaussian", "tophat", "epanechnikov"]:
            kde = KernelDensityEstimate(kernel=kernel, bandwidth=bandwidth, num_cls=1)
            kde.fit(x, y)
            log_dens = kde.list_den_est[0].score_samples(x_plot)
            ax.plot(x_plot[:, 0], np.exp(log_dens), "-", label=f"kernel = '{kernel}'")

        ax.text(6, 0.38, "N={0} points".format(n))

        ax.legend(loc="upper left")
        ax.plot(x[:, 0], -0.005 - 0.01 * np.random.random(x.shape[0]), "+k")

        ax.set_xlim(-4, 9)
        ax.set_ylim(-0.02, 0.4)
        return fig

    def test_kde_values(self):
        pca = ChannelWisePrincipalComponentAnalysis(num_ch=self.num_channel, var_tol=0.5)
        rda = RegularizedDiscriminantAnalysis()
        kde = KernelDensityEstimate()

        pipeline = Pipeline()
        pipeline.add(pca)
        pipeline.add(rda)
        pipeline.add(kde)

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
        pca = ChannelWisePrincipalComponentAnalysis(num_ch=self.num_channel, var_tol=0.5)
        rda = RegularizedDiscriminantAnalysis()

        pipeline = Pipeline()
        pipeline.add(pca)
        pipeline.add(rda)
        lam, gam = cross_validation(self.x, self.y, pipeline)

        self.assertAlmostEqual(lam, 0.9)
        self.assertAlmostEqual(gam, 0.1)

    def test_rda(self):
        pca = ChannelWisePrincipalComponentAnalysis(num_ch=self.num_channel, var_tol=0.5)
        rda = RegularizedDiscriminantAnalysis()

        pipeline = Pipeline()
        pipeline.add(pca)
        pipeline.add(rda)

        # .fit() followed by .transform() should match .fit_transform()
        z = pipeline.fit_transform(self.x, self.y)
        pipeline.fit(self.x, self.y)
        z_2 = pipeline.transform(self.x)
        self.assertTrue(np.allclose(z, z_2))

        # output values should be correct
        expected = np.load(expected_output_folder / "test_rda.expected.npy")
        self.assertTrue(np.allclose(z, expected))


if __name__ == "__main__":
    unittest.main()
