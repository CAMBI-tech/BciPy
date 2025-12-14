# Signal

The BciPy Signal module contains all code needed to process, evaluate, model, and generate signals for Brain Computer Interface control using EEG and/or Eye Tracking. Further documentation provided in submodule READMEs.

## Evaluate

The evaluation module contains functions for evaluating signals based on configured rules. The module contains functionailty for detecting artifacts in EEG signals, and for evaluating the quality of the signal. In addition, analysis functions are provided to evaluate the performance of the BCI system. Currently, the fusion of the signals is evaluated using the `calculate_eeg_gaze_fusion_acc` function.

## Process

The process module contains functions for decomposing signals into frequency bands (psd, cwt), filtering signals (bandpass, notch), and other signal processing functions.

## Model

The module contains functions for training and testing classifiers, and for evaluating the performance of the classifiers. Several classifiers are provided, including a PCA/RDA/KDE classifier and several Gaussian Mixture Model classifiers. See the submodule README for more information.

### Model Training (offline analysis)

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

## Generator

Generates fake signal data.
