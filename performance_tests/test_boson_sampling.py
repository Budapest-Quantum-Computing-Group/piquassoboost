#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import time
import pytest
import itertools

import numpy as np

import piquasso as pq


@pytest.fixture
def print_histogram():
    def func(samples):
        hist = dict()

        for sample in samples:
            key = tuple(sample)
            if key in hist.keys():
                hist[key] += 1
            else:
                hist[key] = 1

        for key in hist.keys():
            print(f"{key}: {hist[key]}")

    return func


def test_two_mode_sampling(print_histogram):
    """
    NOTE: Expected distribution probabilities:
    (1, 1):  0.250
    (0, 2):  0.375
    (2, 0):  0.375
    """

    shots = 10000
    for _ in itertools.repeat(None, 10):

        pq.constants.seed()

        with pq.Program() as program:
            pq.Q() | pq.SamplingState(1, 1)

            pq.Q(0, 1) | pq.Beamsplitter(np.pi / 3)

            pq.Q() | pq.Sampling()

        t0 = time.time()

        result = program.execute(shots=shots)

        print("C++ time elapsed:", time.time() - t0, "s")

        print_histogram(result.samples)


def test_complex_sampling(print_histogram):
    """
    NOTE: Expected distribution probabilities:
    (0, 2, 0, 1, 1): 0.1875
    (0, 2, 1, 0, 1): 0.1875
    (1, 1, 0, 1, 1): 0.1250
    (1, 1, 1, 0, 1): 0.1250
    (2, 0, 0, 1, 1): 0.1875
    (2, 0, 1, 0, 1): 0.1875
    """

    for _ in itertools.repeat(None, 10):
        shots = 10000

        with pq.Program() as program:
            pq.Q() | pq.SamplingState(1, 1, 1, 0, 1)

            pq.Q(0, 1) | pq.Beamsplitter(np.pi / 3)
            pq.Q(2)    | pq.Fourier()
            pq.Q(2, 3) | pq.Beamsplitter(np.pi / 4)

            pq.Q() | pq.Sampling()

        t0 = time.time()

        result = program.execute(shots=shots)

        print("C++ time elapsed:", time.time() - t0, "s")

        print_histogram(result.samples)
