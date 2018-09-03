from bcipy.signal.model.mach_learning.dimensionality_reduction.function_dim_reduction \
    import ChannelWisePrincipalComponentAnalysis
import numpy as np


def _demo_cw_pca():
    num_ch = 16
    dim_x = 20
    num_x_p = 100
    num_x_n = 0
    var_tol = 0.2

    # We only require the data not labels
    x_p = 2 * np.random.randn(num_ch, num_x_p, dim_x)
    x_n = np.random.randn(num_ch, num_x_n, dim_x)
    x = np.concatenate((x_n, x_p), axis=1)

    cw_pca = ChannelWisePrincipalComponentAnalysis(num_ch=num_ch)
    y = cw_pca.fit_transform(x, var_tol=var_tol)
    print('CW-PCA Results')
    print('CW-PCA flows!')
    y_2 = cw_pca.transform(x)
    print('X: {} |--[CW-PCA]--> Y: {}'.format(x.shape, y.shape))
    print('MSE:{} fit_transform'.format(np.sum(np.abs(y_2 - y))))

    return 0


if __name__ == "__main__":
    _demo_cw_pca()