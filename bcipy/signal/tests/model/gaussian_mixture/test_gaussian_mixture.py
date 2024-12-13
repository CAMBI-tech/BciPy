import unittest

from bcipy.signal.model.gaussian_mixture import (
    GaussianProcess,
    GMCentralized,
    GMIndividual,
    GazeModelResolver
)


class TestGazeModelResolver(unittest.TestCase):

    def test_resolve(self):
        response = GazeModelResolver.resolve('GaussianProcess')
        self.assertIsInstance(response, GaussianProcess)

    def test_resolve_centralized(self):
        response = GazeModelResolver.resolve('GMCentralized')
        self.assertIsInstance(response, GMCentralized)

    def test_resolve_individual(self):
        response = GazeModelResolver.resolve('GMIndividual')
        self.assertIsInstance(response, GMIndividual)

    def test_resolve_raises_value_error_on_invalid_model(self):
        with self.assertRaises(ValueError):
            GazeModelResolver.resolve('InvalidModel')


class TestModelInit(unittest.TestCase):

    def test_gaussian_process(self):
        model = GaussianProcess()
        self.assertIsInstance(model, GaussianProcess)

    def test_centrailized(self):
        model = GMCentralized()
        self.assertIsInstance(model, GMCentralized)

    def test_individual(self):
        model = GMIndividual()
        self.assertIsInstance(model, GMIndividual)


if __name__ == "__main__":
    unittest.main()
