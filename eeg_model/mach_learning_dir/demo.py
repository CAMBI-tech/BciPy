from __init__ import PipelineWrapper
import sys

sys.path.append('.\eeg_model\mach_learning_dir\dimensionality_reduction')
sys.path.append('.\eeg_model\mach_learning_dir\classifier')
from function_classifier import RegularizedDiscriminantAnalysis
from function_dim_reduction import PrincipalComponentAnalysis
import numpy as np

dim_x = 16
num_x_p = 2000
num_x_n = 1000
var_tol = 0.80

x_p = 2 * np.random.randn(num_x_p, dim_x)
x_n = 2 + np.random.randn(num_x_n, dim_x)
y_p = [1] * num_x_p
y_n = [0] * num_x_n

x = np.concatenate(np.asarray([x_p, x_n]), 0)
y = np.concatenate(np.asarray([y_p, y_n]), 0)

pca = PrincipalComponentAnalysis()
rda = RegularizedDiscriminantAnalysis()
model = PipelineWrapper()
model.add(pca)
model.add(rda)
arg = model.fit_transform(x, y)
