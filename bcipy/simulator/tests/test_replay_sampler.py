import unittest

from bcipy.simulator.data.replay_sampler import compute_inquiry_index


class TestReplaySamplerHelpers(unittest.TestCase):
    """Helper function tests."""

    def test_compute_inquiry(self):
        """Test inquiry_n computation."""
        self.assertEqual(
            0, compute_inquiry_index([10, 5, 3], series=1, inquiry=0))
        self.assertEqual(
            9, compute_inquiry_index([10, 5, 3], series=1, inquiry=9))

        self.assertRaises(
            AssertionError,
            lambda: compute_inquiry_index([10, 5, 3], series=1, inquiry=10))

        self.assertEqual(
            10, compute_inquiry_index([10, 5, 3], series=2, inquiry=0))
        self.assertEqual(
            14, compute_inquiry_index([10, 5, 3], series=2, inquiry=4))

        self.assertEqual(
            15, compute_inquiry_index([10, 5, 3], series=3, inquiry=0))
        self.assertEqual(
            17, compute_inquiry_index([10, 5, 3], series=3, inquiry=2))


if __name__ == '__main__':
    unittest.main()
