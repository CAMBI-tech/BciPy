from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import queue
import time

from datastream.producer import Producer
import unittest


class TestProducer(unittest.TestCase):
    """Tests for Producer"""

    def test_frequency(self):
        """Data should be generated at the provided frequency"""
        fs = 300
        runtime = 0.2
        q = queue.Queue()
        p = Producer(q, freq=1 / fs)
        p.start()
        time.sleep(runtime)
        p.stop()

        n = q.qsize()
        expected_n = fs * runtime
        tolerance = 5
        self.assertTrue(n + tolerance >= expected_n)
        self.assertTrue(n <= expected_n + tolerance)

    def test_custom_generator(self):
        """Producer should be able to take a custom generator."""

        def gen():
            counter = 0
            while True:
                counter += 1
                yield counter

        q = queue.Queue()
        p = Producer(q, freq=1 / 300, generator=gen())
        p.start()
        time.sleep(0.1)
        p.stop()

        lst = list(q.queue)
        self.assertTrue(len(lst) > 0)
        self.assertEqual(lst[0], 1)
        self.assertEqual(lst[-1], len(lst))
        print(lst[-1])

    def test_max_iters(self):
        """Producer should stop producing data after maxiters if param is
        provided."""

        fs = 300
        runtime = 0.2
        maxiters = 10
        q = queue.Queue()
        p = Producer(q, freq=1 / fs, maxiters=maxiters)
        p.start()
        time.sleep(runtime)
        p.stop()

        expected_n = fs * runtime
        tolerance = 10
        self.assertTrue(expected_n - tolerance > maxiters)
        n = q.qsize()
        self.assertEqual(n, maxiters)
