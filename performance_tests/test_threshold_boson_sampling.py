import numpy as np

import piquasso as pq

from cpiquasso import patch

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
        d = 12

        # Number of shots
        shots = 1000

        # rundom parameters for squeezing gates
        squeezing_params_r = np.random.random(d)
        squeezing_params_phi = np.random.random(d)

        # random unitary matrix for perform interferometer
        interferometer_param = unitary_group.rvs(d)

        ###################################

        # Piquasso python program
        with pq.Program() as pq_program:
            pq.Q() | pq.GaussianState(d=d)

            # Apply random squeezings
            for idx in range(d):
                pq.Q(idx) | pq.Squeezing(r=squeezing_params_r[idx], phi=squeezing_params_phi[idx])

            # Apply random interferometer
            pq.Q() | pq.Interferometer(interferometer_param)

            # Measure all modes with shots shots
            pq.Q() | pq.ThresholdMeasurement(shots=shots)

        # Measuring runtime
        startTime = time.time()
        pq_results = np.array(pq_program.execute()[0].samples)
        endTime = time.time()

        piquasso_time = endTime - startTime

        ###################################

        # Use piquasso boost library
        patch()

        # Piquasso boost program
        with pq.Program() as pq_program:
            pq.Q() | pq.GaussianState(d=d)

            # Apply random squeezings
            for idx in range(d):
                pq.Q(idx) | pq.Squeezing(r=squeezing_params_r[idx], phi=squeezing_params_phi[idx])

            # Apply random interferometer
            pq.Q() | pq.Interferometer(interferometer_param)

            # Measure all modes with shots shots
            pq.Q() | pq.ThresholdMeasurement(shots=shots)

        # Measuring runtime
        startTime = time.time()
        pq_results = np.array(pq_program.execute()[0].samples)
        endTime = time.time()

        piquasso_boost_time = endTime - startTime

        ###################################

        print(' ')
        print('*******************************************')
        print('Number of modes: ', d)
        print('Time elapsed with piquasso      : ' + str(piquasso_time))
        print('Time elapsed with piquasso boost: ' + str(piquasso_boost_time))
        print( "speedup: " + str(piquasso_time/piquasso_boost_time) )
