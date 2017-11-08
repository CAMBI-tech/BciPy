""" Offline Analysis implementation. Trains the model and shows some results.
    List of duties,
        - reads data for a calibration session
        - initializes itself (parameters)
        - accepts filtered data
        - reshapes the data for the training procedure
        - fits the model to the data
            - uses cross validation to select parameters
            - based on the parameters, trains system using all the data
        - offline analysis screen
    """

from eeg_model.mach_learning.train_model import train_pca_rda_kde_model
import numpy as np
import matplotlib as mpl
from scipy.io import loadmat

mpl.use('TkAgg')
import matplotlib.pylab as plt

# dim_x = 30
# num_ch = 20
# num_x_p = 100
# num_x_n = 1300
#
# x_p = 2 + 5 * np.random.randn(num_ch, num_x_p, dim_x)
# x_n = 5 * np.random.randn(num_ch, num_x_n, dim_x)
# y_p = [1] * num_x_p
# y_n = [0] * num_x_n
#
# x = np.concatenate((x_p, x_n), 1)
# y = np.concatenate(np.asarray([y_p, y_n]), 0)
# permutation = np.random.permutation(x.shape[1])
# x = x[:, permutation, :]
# y = y[permutation]

# Load data
fol = loadmat('dummy_dat.mat')
y = fol['trialTargetness']
y = np.squeeze(y)
x = fol['trialData']
x = np.swapaxes(x, 0, 2)
x = np.swapaxes(x, 0, 1)

# Load triggers

model = train_pca_rda_kde_model(x, y)

fig, ax = plt.subplots()
x_plot = np.linspace(np.min(model.line_el[-1]), np.max(model.line_el[-1]),
                     1000)[:, np.newaxis]
ax.plot(model.line_el[2][y == 0],
        -0.005 - 0.01 * np.random.random(model.line_el[2][y == 0].shape[0]),
        'ro', label='class(-)')
ax.plot(model.line_el[2][y == 1],
        -0.005 - 0.01 * np.random.random(model.line_el[2][y == 1].shape[0]),
        'bo', label='class(+)')

for idx in range(len(model.pipeline[2].list_den_est)):
    log_dens = model.pipeline[2].list_den_est[idx].score_samples(x_plot)
    ax.plot(x_plot[:, 0], np.exp(log_dens),
            'r-' * (idx == 0) + 'b--' * (idx == 1),
            linewidth=2.0)

ax.legend(loc='upper right')
plt.title('Likelihoods Given the Labels')
plt.ylabel('p(e|l)')
plt.xlabel('scores')
plt.show()
