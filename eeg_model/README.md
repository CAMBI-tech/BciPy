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




