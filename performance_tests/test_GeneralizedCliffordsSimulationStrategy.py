# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np
import pytest

import time
from piquasso.sampling.backend import SamplingBackend
from piquasso.sampling.state import SamplingState
from scipy.stats import unitary_group


def calc_histogram( samples ):

    hist = dict()

    for sample in samples:
        key = tuple(sample)
        if key in hist.keys():
            hist[key] += 1
        else:
            hist[key] = 1

    return hist


class TestGeneralizedCliffordSimulationStrategy:


    def test_sampling(self):

        interferometer_mtx = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, complex(0.5,0.7), 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ], dtype=complex)
    
        # generate random unitary for the interferometer matrix 
        problem_size = 10 
        #interferometer_mtx = unitary_group.rvs(problem_size)


     

        for i in range(10):

            # create inputs for the permanent calculations
            initial_state = SamplingState(1, 1, 1, 0, 1)
            #initial_state = SamplingState(1, 1, 1, 0, 1, 0, 1, 0, 0, 0)

            backend = SamplingBackend(initial_state)
            backend.state.interferometer = interferometer_mtx


            shots = 10000

            t0 = time.time()

            backend.sampling((shots,))

            print('C++ time elapsed: ' + str( time.time() - t0) + 's' )
       
            
            hist = calc_histogram( backend.state.results )
            for key in hist.keys():
               print( str(key) + ': ', str(hist[key])) 



