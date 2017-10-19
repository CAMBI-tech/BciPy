import sys

sys.path.append('.\eeg_model\mach_learning_dir\dimensionality_reduction')
from function_dim_reduction import PrincipalComponentAnalysis, \
    ChannelWisePrincipalComponentAnalysis
import numpy as np


def test_pca(x):
    pca = PrincipalComponentAnalysis()
    y = pca.fit_transform(x)

    return [y]

# pca = PrincipalComponentAnalysis()
#
# dim_x = 16
# num_x_p = 2000
# num_x_n = 1000
# var_tol = 0.80
#
# # We only require the data not labels
# x_p = 2 * np.random.randn(num_x_p, dim_x)
# x_n = np.random.randn(num_x_n, dim_x)
#
# x = np.concatenate(np.asarray([x_p, x_n]), 0)
# len_x = x.shape[0]
#
# pca.fit(x[0:int(len_x * 9 / 10), :])
# y = pca.transform(x[int(len_x * 9 / 10):-1, :])
# print(y.shape)
#
# pca.fit(x[0:int(len_x * 9 / 10), :], var_tol=var_tol)
# y = pca.transform(x[0:int(len_x * 9 / 10), :])
# print(y.shape)
#
# y2 = pca.fit_transform(x[0:int(len_x * 9 / 10), :], var_tol=var_tol)
# print(np.sum(y2 - y))

##

num_ch = 16
dim_x = 20
num_x_p = 100
num_x_n = 0
var_tol = 0.2

# We only require the data not labels
x_p = 2 * np.random.randn(num_ch, num_x_p, dim_x)
x_n = np.random.randn(num_ch, num_x_n, dim_x)
x = np.concatenate((x_n, x_p), axis=1)
len_x = x.shape[1]

cw_pca = ChannelWisePrincipalComponentAnalysis(num_ch=num_ch)
y = cw_pca.fit_transform(x, var_tol=var_tol)
y_2 = cw_pca.transform(x)
print('x: {}, y: {}'.format(x.shape, y.shape))

