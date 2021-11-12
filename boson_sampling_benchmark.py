

import time
import pytest
import itertools
import scipy

import numpy as np

import piquasso as pq

from scipy import stats

# make piquassoboost to run
from piquassoboost import patch

patch()

def print_histogram(samples):
    hist = dict()

    for sample in samples:
        key = tuple(sample)
        if key in hist.keys():
            hist[key] += 1
        else:
            hist[key] = 1

    for key in hist.keys():
        print(f"{key}: {hist[key]}")


def complex_sampling_interferometer():
    """
    NOTE: with random interferometer matrix with the given dimension
    """

    def _generate_parameters(d):

        return {
            "interferometer": scipy.stats.unitary_group.rvs(d)
        }




    dimension = 20
    
    params = _generate_parameters(dimension)

    for _ in itertools.repeat(None, 1):
        shots = 1

        with pq.Program() as program:

            pq.Q(all) | pq.Interferometer(params["interferometer"])

            pq.Q(all) | pq.Sampling()

        #state = pq.SamplingState(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)#10
        #state = pq.SamplingState(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)#14
        state = pq.SamplingState(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)#20
        t0 = time.time()

        result = state.apply(program, shots=shots)
        
        t1 = time.time()
        
        print("histogram:")
        print_histogram(result.samples)
        
        print("C++ time elapsed:", t1 - t0, "s")
        
complex_sampling_interferometer()
