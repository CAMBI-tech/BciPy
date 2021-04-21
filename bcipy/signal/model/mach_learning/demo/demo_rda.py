import numpy as np
from bcipy.signal.model.mach_learning.classifier import RegularizedDiscriminantAnalysis


def _demo_rda():
    dim_x = 2
    num_x_p = 2000
    num_x_n = 500

    x_p = 2 * np.random.randn(num_x_p, dim_x)
    x_n = 4 + np.random.randn(num_x_n, dim_x)
    y_p = [1] * num_x_p
    y_n = [0] * num_x_n

    x = np.concatenate(np.asarray([x_p, x_n]), 0)
    y = np.concatenate(np.asarray([y_p, y_n]), 0)

    rda = RegularizedDiscriminantAnalysis()

    z = rda.fit_transform(x, y)
    print('Successfully used RDA')
    rda.fit(x, y)
    z_2 = rda.transform(x)
    print(f'MSE:{np.sum(np.abs(z_2 - z))} for fit_transform')


if __name__ == "__main__":
    _demo_rda()
