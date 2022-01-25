#
#  Copyright 2021 Budapest Quantum Computing Group
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import time
import pytest
import itertools
import scipy

import numpy as np

import piquasso as pq

from scipy.optimize import root_scalar
from scipy import stats

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
            pq.Q(0, 1) | pq.Beamsplitter(np.pi / 3)

            pq.Q() | pq.Sampling()

        state = pq.SamplingState(1, 1)

        t0 = time.time()

        result = state.apply(program)

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
            pq.Q(0, 1) | pq.Beamsplitter(np.pi / 3)
            pq.Q(2)    | pq.Fourier()
            pq.Q(2, 3) | pq.Beamsplitter(np.pi / 4)

            pq.Q() | pq.Sampling()

        state = pq.SamplingState(1, 1, 1, 0, 1)

        t0 = time.time()

        result = state.apply(program, shots=shots)

        print("C++ time elapsed:", time.time() - t0, "s")

        print_histogram(result.samples)


