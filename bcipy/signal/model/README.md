# EEG Model Module

Aims to model likelihood for EEG evidence.

## Machine Learning (ML)

Includes machine learning algorithms for feature extraction and likelihood calculation.

For examples refer to the `.\mach_learning\demo` folder.

### Dimensionality Reduction

Reduces dimensionality of the EEG data obtained based on a comparison metric. 

Includes 

-`PCA`: a dimensionality reduction, finds and keeps meaningfull dimensions with high variance.

-`Channel-Wise-PCA`: a PCA per channel

### Classifier

Classifiers for EEG likelihood computation is used as a dimensionality reduction wrt. the decision metric.

Includes `RDA`: an extension to `QDA` with regularization and thresholding parameters. Allows to fit tight decision boundaries in high dimensional spaces.

### Generative Models

Used the generate likelihoods based on scores obtained through feature extraction.

Includes `KDE`: a non-parametric density estimate using a pre-defined kernel and its width.

### Pipeline

Appends ML elements to a list. Currently `Pipeline:= [CW-PCA, RDA, KDE]`

### Cross Validation (CV)

Generalizes the model using data, avoids overfitting. Implementation is tied to optimizing hyperparameters of the `RDA`.

# Testing
## pytest-mpl

Tests in <tests/test_inference.py> and <mach_learning/tests/test_density_estimation.py> use a pytest plugin to compare an output plot against the expected output plot.

- Generate the "baseline" result by running: `pytest bcipy/signal/model/ -k <NAME_OF_TEST> --mpl-generate-path=bcipy/signal/model/tests/baseline` 
  (the expected output will be stored in a new folder there called `baseline`)
- Run the test using `pytest --mpl` (otherwise this test will be skipped)

To sanity check that this test is sensitive, you can generate the baseline using one seed, then change the seed and run the test (it will fail).

See more about `pytest-mpl` at https://pypi.org/project/pytest-mpl/

