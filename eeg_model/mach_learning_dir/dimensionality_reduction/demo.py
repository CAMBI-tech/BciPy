import sys

sys.path.append('.\eeg_model\mach_learning_dir\dimensionality_reduction')
from function_dim_reduction import PrincipalComponentAnalysis
import numpy as np

pca = PrincipalComponentAnalysis()

dim_x = 16
num_x_p = 2000
num_x_n = 1000
var_tol = 0.80

# We only require the data not labels
x_p = 2 * np.random.randn(num_x_p, dim_x)
x_n = np.random.randn(num_x_n, dim_x)

x = np.concatenate(np.asarray([x_p, x_n]), 0)
len_x = x.shape[0]

pca.fit(x[0:int(len_x * 9 / 10), :])
y = pca.transform(x[int(len_x * 9 / 10):-1, :])
print(y.shape)

pca.fit(x[0:int(len_x * 9 / 10), :], var_tol=var_tol)
y = pca.transform(x[0:int(len_x * 9 / 10), :])
print(y.shape)

y2 = pca.fit_transform(x[0:int(len_x * 9 / 10), :], var_tol=var_tol)
print(np.sum(y2 - y))
