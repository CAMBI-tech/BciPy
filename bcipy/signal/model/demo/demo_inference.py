""" Inference demo """

import matplotlib.pylab as plt
import numpy as np
from bcipy.signal.model.inference import inference
from bcipy.signal.model.pca_rda_kde.train_model import train_pca_rda_kde_model

import matplotlib as mpl

mpl.use('TkAgg')

dim_x = 5
num_ch = 1
num_x_p = 100
num_x_n = 900

mean_pos = .8
var_pos = .5
mean_neg = 0
var_neg = .5

x_p = mean_pos + var_pos * np.random.randn(num_ch, num_x_p, dim_x)
x_n = mean_neg + var_neg * np.random.randn(num_ch, num_x_n, dim_x)
y_p = [1] * num_x_p
y_n = [0] * num_x_n

x = np.concatenate((x_p, x_n), 1)
y = np.concatenate(np.asarray([y_p, y_n]), 0)
permutation = np.random.permutation(x.shape[1])
x = x[:, permutation, :]
y = y[permutation]

k_folds = 10
model, _ = train_pca_rda_kde_model(x, y, k_folds=k_folds)

alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
       'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', '<', '_']

num_x_p = 1
num_x_n = 9

x_p_s = mean_pos + var_pos * np.random.randn(num_ch, num_x_p, dim_x)
x_n_s = mean_neg + var_neg * np.random.randn(num_ch, num_x_n, dim_x)
x_s = np.concatenate((x_n_s, x_p_s), 1)

idx_let = np.random.permutation(len(alp))
letters = [alp[i] for i in idx_let[0:(num_x_p + num_x_n)]]

print(letters)
print('target letter: {}'.format(letters[-1]))
lik_r = inference(x_s, letters, model, alp)

plt.plot(np.array(list(range(len(alp)))), lik_r, 'ro')
plt.xticks(np.array(list(range(len(alp)))), alp)
plt.show()
