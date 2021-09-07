import numpy as np

import piquasso as pq

from piquassoboost import patch

import time
from scipy.stats import unitary_group





class TestThresholdBosonSampling:
    """bechmark tests for threshold boson sampling calculations
    """

    def test_value(self):
        """Check threshold boson sampling calculated by piquasso python and piquasso boost"""

        # Use piquasso python code
        pq.use(pq._DefaultPlugin)

        # Number of modes
        d = 10

        # Number of shots
        shots = 10000

        # rundom parameters for squeezing gates
        squeezing_params_r = np.random.random(d)
        squeezing_params_phi = np.random.random(d)

        # random unitary matrix for perform interferometer
        interferometer_param = unitary_group.rvs(d)

        ###################################

        # Piquasso python program
        with pq.Program() as pq_program:
            # Apply random squeezings
            for idx in range(d):
                pq.Q(idx) | pq.Squeezing(r=squeezing_params_r[idx], phi=squeezing_params_phi[idx])

            # Apply random interferometer
            pq.Q() | pq.Interferometer(interferometer_param)

            # Measure all modes with shots shots
            pq.Q() | pq.ThresholdMeasurement()

        state = pq.GaussianState(d=d)

        # Measuring runtime
        startTime = time.time()
        result = state.apply(program=pq_program, shots=shots)
        pypq_results = np.array(result.samples)
        endTime = time.time()

        piquasso_time = endTime - startTime

        ###################################

        # Use piquasso boost library
        patch()

        # Piquasso boost program
        with pq.Program() as pq_program:
            # Apply random squeezings
            for idx in range(d):
                pq.Q(idx) | pq.Squeezing(r=squeezing_params_r[idx], phi=squeezing_params_phi[idx])

            # Apply random interferometer
            pq.Q() | pq.Interferometer(interferometer_param)

            # Measure all modes with shots shots
            pq.Q() | pq.ThresholdMeasurement()

        state = pq.GaussianState(d=d)

        # Measuring runtime
        startTime = time.time()
        result = state.apply(program=pq_program, shots=shots)
        cpq_results = np.array(result.samples)
        endTime = time.time()

        piquasso_boost_time = endTime - startTime

        ###################################

        print(' ')
        print('*******************************************')
        print('Number of modes: ', d)
        print('Time elapsed with piquasso      : ' + str(piquasso_time))
        print('Time elapsed with piquasso boost: ' + str(piquasso_boost_time))
        print('The result of piquasso python: \n' , pypq_results)
        print('The result of piquasso C++:    \n' , cpq_results)
        print( "speedup: " + str(piquasso_time/piquasso_boost_time) )
