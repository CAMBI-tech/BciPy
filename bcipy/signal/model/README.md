# EEG Modeling

This module provides models to use EEG evidence to update the posterior probability of stimuli viewed by a user. The module includes a PCA/RDA/KDE model and an RDA/KDE model. The PCA/RDA/KDE model uses a generative model to estimate the likelihood of the EEG data given the stimuli, and the RDA/KDE model uses a discriminative model to estimate the likelihood of the stimuli given the EEG data. In addition, the module includes a gaze model that uses gaze data to update the posterior probability of stimuli viewed by a user.

## Model Training (offline analysis)

To train a signal model (such as, `PCARDAKDE`), run the following command after installing BciPy:

`bcipy-train`

Use the help flag to see other available input options: `bcipy-train --help` You can pass it attributes with flags, if desired.

Execute without a window prompting for data session folder: `bcipy-train -d path/to/data`

Execute with data visualizations (ERPs, etc.): `bcipy-train -v`

Execute with data visualizations that do not show, but save to file:  `bcipy-train -s`

Execute with balanced accuracy: `bcipy-train --balanced-acc`

Execute with alerts after each Task execution: `bcipy-train --alert`

Execute with custom parameters: `bcipy-train -p "path/to/valid/parameters.json"`

Execute with custom number of iterations for fusion analysis (by default 10): `bcipy-train -i 10`

## EEG Models

### PCA/RDA/KDE Model

This model involves the following stages:

1. Channelwise-PCA to reduce data dimension while preserving as much variation in the data as possible. See `dimensionality_reduction.py` ChannelWisePrincipalComponentAnalysis.

2. Regularized Discriminant Analysis (RDA), which further reduces dimension to 1D by estimating class probabilities for a positive and negative class (i.e. whether a single letter was desired or not). RDA includes two key parameters, `gamma` and `lambda` which determine how much the estimated class covariances are regularized towards the whole-data covariance matrix and towards the identity matrix. See `classifier.py`.

3. Kernel Density Estimation (KDE), which performs generative modeling on the reduced dimension data, computing the probability that it arose from the positive class, and from the negative class. This method involves choosing a kernel (a notion of distance) and a bandwidth (a length scale for the kernel). See `density_estimation.py`.

4. AUC/AUROC calculation: PCA/RDA part of the model is trained using k-fold cross-validation, then the AUC is computed using the optimized `gamma` and `lambda` values. See `cross_validation.py`.

5. In order to make a Bayesian update, we need to compute the ratio of the generative likelihood terms for the presented letter (`p(eeg | +)` and `p(eeg | -)`). This ratio is obtained from the final kernel density estimation step and is used in the final decision rule. See `pca_rda_kde/pca_rda_kde.py`.

### RDA/KDE Model

This model involves the following stages:

1. MockPCA to reduce data dimension while preserving as much variation in the data as possible. See `dimensionality_reduction.py MockPCA`.

2. Regularized Discriminant Analysis (RDA), which further reduces dimension to 1D by estimating class probabilities for a positive and negative class (i.e. whether a single letter was desired or not). RDA includes two key parameters, `gamma` and `lambda` which determine how much the estimated class covariances are regularized towards the whole-data covariance matrix and towards the identity matrix. See `classifier.py`.

3. Kernel Density Estimation (KDE), which performs generative modeling on the reduced dimension data, computing the probability that it arose from the positive class, and from the negative class. This method involves choosing a kernel (a notion of distance) and a bandwidth (a length scale for the kernel). See `density_estimation.py`.

4. AUC/AUROC calculation: RDA part of the model is trained using k-fold cross-validation, then the AUC is computed using the optimized `gamma` and `lambda` values. See `cross_validation.py`.

5. In order to make a Bayesian update, we need to compute the ratio of the generative likelihood terms for the presented letter (`p(eeg | +)` and `p(eeg | -)`). This ratio is obtained from the final kernel density estimation step and is used in the final decision rule. See `rda_kde/rda_kde.py`.

## Eye Tracking Models

These models may be trained and evalulated, but are still being integrated into the BciPy system for online use.

### Gaze Model

*Note*: The gaze model is currently under development and is not yet fully implemented.

These models are used to update the posterior probability of stimuli viewed by a user based on gaze data. The gaze model uses a generative model to estimate the likelihood of the gaze data given the stimuli. There are several models implemented in this module, including a Gaussian Mixture Model (GMIndividual) and a Gaussian Process Model (GaussianProcess). When training data via offline analysis, if the data folder contains gaze data, the gaze model will be trained and saved to the output directory.

## Fusion Analyis

*Note*: The fusion analysis is currently under development and is not yet fully implemented.

The `calculate_eeg_gaze_fusion_acc` function is used to evaluate the performance of the BCI system. The function takes in a list of EEG and gaze data, and returns the accuracy of the fusion of the two signals.

## Testing

Run tests for this module as follows (from the root directory):

```bash
pytest --mpl bcipy/signal/tests/model -k "not slow"  # unit tests only
pytest --mpl bcipy/signal/tests/model -k "slow"      # integration tests only
pytest --mpl bcipy/signal/tests/model                # all tests
```

### notes on pytest-mpl

Some tests in `bcipy/signal/tests/model` use a pytest plugin to compare an output plot against the expected output plot.

If debugging integration tests using pytest-mpl (e.g. `Failed: Error: Image files did not match.`), you can use the `--mpl-generate-summary=html` flag to generate a summary of the figures generated by the tests to compare to the expected output. This will generate a file `pytest-mpl-summary.html` in the current directory.

When the code is in a known working state, you can generate the "expected" results by running:

```bash
pytest -k <NAME_OF_TEST> --mpl-generate-path=<OUTPUT_PATH>
```

where `<OUTPUT_PATH>` is either `bcipy/signal/tests/model/unit_test_expected_output` or `bcipy/signal/tests/model/integration_test_expected_output` depending on the test.

To test the code, use the `--mpl` flag as in `pytest --mpl` (otherwise these figure tests will be skipped).

To sanity check that these tests are sensitive, you can generate the baseline using one random seed, then change the seed and run the test.

Note that the tolerance for pixel differences is configurable, but nonetheless figures should be stripped down to the essential details (since text can move position slightly depending on font libraries and minor version updates of libraries). Furthermore, figures should use a fixed x-axis and y-axis scale to help ensure an easy comparison.

See more about `pytest-mpl` at <https://pypi.org/project/pytest-mpl/>
