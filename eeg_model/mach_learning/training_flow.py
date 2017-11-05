""" Training flow for the model. """

import numpy as np
from eeg_model.mach_learning.classifier.function_classifier \
    import RegularizedDiscriminantAnalysis
from eeg_model.mach_learning.dimensionality_reduction.function_dim_reduction \
    import ChannelWisePrincipalComponentAnalysis
from eeg_model.mach_learning.wrapper import PipeLine
from eeg_model.mach_learning.cross_validation import cross_validation
from eeg_model.mach_learning.generative_mods.function_density_estimation \
    import KernelDensityEstimate
from sklearn import metrics
from scipy.stats import iqr

import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pylab as plt

dim_x = 40
num_ch = 20
num_x_p = 100
num_x_n = 1300

x_p = 1 + 5 * np.random.randn(num_ch, num_x_p, dim_x)
x_n = 5 * np.random.randn(num_ch, num_x_n, dim_x)
y_p = [1] * num_x_p
y_n = [0] * num_x_n

x = np.concatenate((x_p, x_n), 1)
y = np.concatenate(np.asarray([y_p, y_n]), 0)
permutation = np.random.permutation(x.shape[1])
x = x[:, permutation, :]
y = y[permutation]

rda = RegularizedDiscriminantAnalysis()
pca = ChannelWisePrincipalComponentAnalysis()
model = PipeLine()
model.add(pca)
model.add(rda)

sc1 = model.fit_transform(x, y)
fpr, tpr, _ = metrics.roc_curve(y, sc1, pos_label=1)
auc1 = metrics.auc(fpr, tpr)
arg2 = cross_validation(x, y, model=model, k_folds=10)
auc2 = arg2[2]

idx_max_auc = np.where(arg2[2] == np.max(arg2[2]))[0][0]
lam = arg2[0][idx_max_auc]
gam = arg2[1][idx_max_auc]
model.pipeline[1].regularize([lam, gam])

print('Pipeline Successfully Processed Data!')
print('AUC-o: {}, AUC-cv: {}'.format(auc1, np.max(auc2)))

bandwidth = 1.06 * min(
    np.std(x), iqr(x) / 1.34) * np.power(x.shape[0], -0.2)
model.add(KernelDensityEstimate(bandwidth=bandwidth))
model.fit(x, y)

fig, ax = plt.subplots()
x_plot = np.linspace(-20, 20, 1000)[:, np.newaxis]
ax.plot(model.line_el[2][y == 0],
        -0.005 - 0.01 * np.random.random(model.line_el[2][y == 0].shape[0]),
        'o', label='class_(-)')
ax.plot(model.line_el[2][y == 1],
        -0.005 - 0.01 * np.random.random(model.line_el[2][y == 1].shape[0]),
        'o', label='class_(+)')

for idx in range(len(model.pipeline[2].list_den_est)):
    log_dens = model.pipeline[2].list_den_est[idx].score_samples(x_plot)
    ax.plot(x_plot[:, 0], np.exp(log_dens), '-',
            label="p(e|{})".format('-' * (idx == 0)) + '+' * (idx == 1))

ax.legend(loc='upper right')
plt.title('AUC= {}'.format(np.mean(auc2)))
plt.show()
