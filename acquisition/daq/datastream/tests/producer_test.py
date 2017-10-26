from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import Queue
import time

from daq.datastream.producer import Producer


def test_frequency():
    """Data should be generated at the provided frequency"""
    fs = 300
    runtime = 0.2
    q = Queue.Queue()
    p = Producer(q, freq=1 / fs)
    p.start()
    time.sleep(runtime)
    p.stop()

    n = q.qsize()
    expected_n = fs * runtime
    tolerance = 5
    assert n + tolerance >= expected_n
    assert n <= expected_n + tolerance


def test_custom_generator():
    """Producer should be able to take a custom generator."""

    def gen():
        counter = 0
        while True:
            counter += 1
            yield counter

    q = Queue.Queue()
    p = Producer(q, freq=1 / 300, generator=gen())
    p.start()
    time.sleep(0.1)
    p.stop()

    lst = list(q.queue)
    assert len(lst) > 0
    assert lst[0] == 1
    assert lst[-1] == len(lst)
    print(lst[-1])


def test_max_iters():
    """Producer should stop producing data after maxiters if param is
    provided."""

    fs = 300
    runtime = 0.2
    maxiters = 10
    q = Queue.Queue()
    p = Producer(q, freq=1 / fs, maxiters=maxiters)
    p.start()
    time.sleep(runtime)
    p.stop()

    expected_n = fs * runtime
    tolerance = 10
    assert expected_n - tolerance > maxiters
    n = q.qsize()
    assert n == maxiters
