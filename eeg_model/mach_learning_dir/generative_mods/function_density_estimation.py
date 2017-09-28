from sklearn.neighbors.kde import KernelDensity


class KernelDensityEstimate(KernelDensity):
    """ Kernel density estimate implementation using scikit learn
    library. For further reference, please check scikit learn website.
    Attr:
        bandwidth(float): bandwidth of the kernel
        algorithm(string): algorithm type
        kernel(string): element toform the actual fitted pdf.
        metric(string): distance metric used by algorithm to insert kernels
            onto samples in the given domain.
        atol(float): absolute tolerance to fit a probability distribution
        rtol(float): relative tolerance
        breadth_first(bool): Flag to use breadth first search
        leaf_size(int): Uses a tree model to fit probability density.
        metric_params(dictionary): Used by tree model

        """
    def __init(self, bandwidth=1.0, algorithm ='auto', kernel='gaussian',
               metric='euclidean', atol=0, rtol=0, breadth_first=True,
               leaf_size=40, metric_params=None):
        super(KernelDensityEstimate,
              self).__init__(bandwidth=bandwidth, algorithm=algorithm,
                             kernel=kernel, metric=metric, atol=atol,
                             rtol=rtol, breadth_first=breadth_first,
                             leaf_size=leaf_size, metric_params=metric_params)
