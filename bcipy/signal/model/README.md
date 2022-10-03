# EEG Modeling

This module provides models to use EEG evidence to update the posterior probability of stimuli viewed by a user. 

## PCA/RDA/KDE Model

This model involves the following stages:

1. Channelwise-PCA to reduce data dimension while preserving as much variation in the data as possible. See `pca_rda_kde/dimensionality_reduction.py`.

2. Regularized Discriminant Analysis (RDA), which further reduces dimension to 1D by estimating class probabilities for a positive and negative class (i.e. whether a single letter was desired or not). RDA includes two key parameters, `gamma` and `lambda` which determine how much the estimated class covariances are regularized towards the whole-data covariance matrix and towards the identity matrix. See `pca_rda_kde/classifier.py`.

4. Kernel Density Estimation (KDE), which performs generative modeling on the reduced dimension data, computing the probability that it arose from the positive class, and from the negative class. This method involves choosing a kernel (a notion of distance) and a bandwidth (a length scale for the kernel). See `pca_rda_kde/density_estimation.py`.

5. AUC/AUROC calculation: PCA/RDA part of the model is trained using k-fold cross-validation, then the AUC is computed using the optimized `gamma` and `lambda` values. See `pca_rda_kde/cross_validation.py`.

6. In order to make a Bayesian update, we need to compute the ratio of the generative likelihood terms for the presented letter (`p(eeg | +)` and `p(eeg | -)`). This ratio is obtained from the final kernel density estimation step and is used in the final decision rule. See `pca_rda_kde/pca_rda_kde.py`.

# Testing

Run tests for this module with:
```bash
pytest --mpl bcipy/signal/tests/model -k "not slow"  # unit tests only
pytest --mpl bcipy/signal/tests/model -k "slow"      # integration tests only
pytest --mpl bcipy/signal/tests/model                # all tests
```

## pytest-mpl

Some tests in `bcipy/signal/tests/model` use a pytest plugin to compare an output plot against the expected output plot.

When the code is in a known working state, generate the "expected" results by running: 

```bash
pytest -k <NAME_OF_TEST> --mpl-generate-path=<OUTPUT_PATH>
```

where `<OUTPUT_PATH>` is either `bcipy/signal/tests/model/unit_test_expected_output` or `bcipy/signal/tests/model/integration_test_expected_output` depending on the test.

To test the code, use the `--mpl` flag as in `pytest --mpl` (otherwise these figure tests will be skipped).

To sanity check that these tests are sensitive, you can generate the baseline using one random seed, then change the seed and run the test.

Note that the tolerance for pixel differences is configurable, but nonetheless figures should be stripped down to the essential details (since text can move position slightly depending on font libraries and minor version updates of libraries). Furthermore, figures should use a fixed x-axis and y-axis scale to help ensure an easy comparison.

See more about `pytest-mpl` at https://pypi.org/project/pytest-mpl/
