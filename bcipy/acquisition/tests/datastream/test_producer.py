"""Tests for Datastream Producer"""
import queue
import time
import unittest

import pytest

from bcipy.acquisition.datastream.producer import Producer


# These tests are not slow but they sometimes fail during continuous
# integration checks. Producers are used in mock development servers,
# but are not critical to production use of BciPy so they are being
# temporarily marked as slow to ensure that they are skipped during
# integration. The tests may be run locally to ensure correct behavior.
@pytest.mark.slow
class TestProducer(unittest.TestCase):
    """Tests for Producer"""

    @pytest.mark.skip(reason="Skipping due to CI failures. Run locally to test.")
    def test_frequency(self):
        """Data should be generated at the provided frequency"""
        sample_hz = 300
        runtime = 0.2
        data_queue = queue.Queue()
        producer = Producer(data_queue, freq=1 / sample_hz)
        producer.start()
        time.sleep(runtime)
        producer.stop()

        data_n = data_queue.qsize()
        expected_n = sample_hz * runtime
        tolerance = 10
        print(f'Expected: {expected_n}; Actual: {data_n}')
        self.assertTrue(data_n + tolerance >= expected_n)
        self.assertTrue(data_n <= expected_n + tolerance)

    def test_custom_generator(self):
        """Producer should be able to take a custom generator."""
        def gen():
            counter = 0
            while True:
                counter += 1
                yield counter

        data_queue = queue.Queue()
        producer = Producer(data_queue, freq=1 / 300, generator=gen())
        producer.start()
        time.sleep(0.1)
        producer.stop()

        lst = list(data_queue.queue)
        self.assertTrue(len(lst) > 0)
        self.assertEqual(lst[0], 1)
        self.assertEqual(lst[-1], len(lst))
        print(lst[-1])

    def test_max_iters(self):
        """Producer should stop producing data after maxiters if param is
        provided."""

        sample_hz = 300
        runtime = 0.2
        maxiters = 10
        data_queue = queue.Queue()
        producer = Producer(data_queue, freq=1 / sample_hz, maxiters=maxiters)
        producer.start()
        time.sleep(runtime)
        producer.stop()

        expected_n = sample_hz * runtime
        tolerance = 10
        self.assertTrue(expected_n - tolerance > maxiters)
        data_n = data_queue.qsize()
        self.assertEqual(data_n, maxiters)

    def test_producer_end(self):
        data_queue = queue.Queue()

        def stopiteration_generator(n):
            i = 0
            while True:
                yield i
                i += 1
                if i >= n:
                    raise StopIteration

        def simple_generator(n):
            for i in range(n):
                yield i

        with self.assertRaises(Exception):
            with Producer(data_queue,
                          generator=stopiteration_generator(10)) as p:
                print("stopiteration_generator")
                p.run()

        with self.assertRaises(Exception):
            with Producer(data_queue, generator=simple_generator(10)) as p:
                print("simple generator")
                p.run()

        with self.assertRaises(Exception):
            with Producer(data_queue, generator=(1 for _ in range(10))) as p:
                print("generator expression")
                p.run()
