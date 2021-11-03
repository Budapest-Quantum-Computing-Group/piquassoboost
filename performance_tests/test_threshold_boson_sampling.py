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

import numpy as np

import piquasso as pq

import piquassoboost as pqb

import time
from scipy.stats import unitary_group





class TestThresholdBosonSampling:
    """bechmark tests for threshold boson sampling calculations
    """

    def test_value(self):
        """Check threshold boson sampling calculated by piquasso python and piquasso boost"""

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

        simulator = pq.GaussianSimulator(d=d)

        # Measuring runtime
        startTime = time.time()
        result = simulator.execute(program=pq_program, shots=shots)
        pypq_results = np.array(result.samples)
        endTime = time.time()

        piquasso_time = endTime - startTime

        ###################################

        # Piquasso boost program
        with pq.Program() as pq_program:
            # Apply random squeezings
            for idx in range(d):
                pq.Q(idx) | pq.Squeezing(r=squeezing_params_r[idx], phi=squeezing_params_phi[idx])

            # Apply random interferometer
            pq.Q() | pq.Interferometer(interferometer_param)

            # Measure all modes with shots shots
            pq.Q() | pq.ThresholdMeasurement()

        simulator = pqb.BoostedGaussianSimulator(d=d)

        # Measuring runtime
        startTime = time.time()
        result = simulator.execute(program=pq_program, shots=shots)
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
