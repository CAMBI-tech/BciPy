"""  """
from eeg_model.mach_learning_dir.generative_mods.function_density_estimation \
    import KernelDensityEstimate
from eeg_model.mach_learning_dir.classifier.function_classifier import \
    RegularizedDiscriminantAnalysis
from eeg_model.mach_learning_dir.dimensionality_reduction.function_dim_reduction \
    import ChannelWisePrincipalComponentAnalysis
from eeg_model.mach_learning_dir.cross_validation import cross_validation
from eeg_model.mach_learning_dir.wrapper import PipeLine
from sklearn import metrics
import numpy as np
from scipy.stats import iqr

dim_x = 16
num_x_p = 2000
num_x_n = 1000
var_tol = 0.80
num_ch = 16

x_p = np.random.randn(num_ch, num_x_p, dim_x)
x_n = np.random.randn(num_ch, num_x_n, dim_x)
y_p = [1] * num_x_p
y_n = [0] * num_x_n

x = np.concatenate((x_n, x_p), axis=1)
y = np.concatenate((y_n, y_p), axis=0)

permutation = np.random.permutation(x.shape[1])
x = x[:, permutation, :]
y = y[permutation]

""" Select bandwidth of the gaussian kernel assuming data is also
    comming from a gaussian distribution.
    Ref: Silverman, Bernard W. Density estimation for statistics and data
    analysis. Vol. 26. CRC press, 1986. """
bandwidth = 1.06 * min(
    np.std(x), iqr(x) / 1.34) * np.power(x.shape[0], -0.2)

pca = ChannelWisePrincipalComponentAnalysis()
rda = RegularizedDiscriminantAnalysis()
kde = KernelDensityEstimate(bandwidth=bandwidth)

model = PipeLine()
model.add(pca)
model.add(rda)

sc1 = model.fit_transform(x, y)
fpr, tpr, _ = metrics.roc_curve(y, sc1, pos_label=1)
arg = metrics.auc(fpr, tpr)
arg2 = cross_validation(x, y, model=model, K=10)
auc2 = arg2[2]

print('Pipeline Successfully Processed Data!')
print('AUC-o: {}, AUC-cv: {}'.format(arg, np.max(arg2)))
