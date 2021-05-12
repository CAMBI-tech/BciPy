import numpy as np

from bcipy.signal.model.neural_net.utils import get_decay_rate, pad


def test_get_decay_rate():
    assert np.allclose(0.1, get_decay_rate(1e-4, 1e-8, 4))


def test_pad():
    x1 = np.ones((3, 5))
    length = 6
    assert pad(x1, length).shape == (3, 6)

    x2 = np.ones((3, 4, 5))
    length = 6
    assert pad(x2, length).shape == (3, 4, 6)
