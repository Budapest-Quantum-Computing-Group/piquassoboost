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
from piquassoboost.sampling.Boson_Sampling_Utilities import PowerTraceHafnian
import time


def generate_random_matrix( dim ):
    """ creating symetric random matrix A of dimension dim x dim """

    A = np.complex128(np.random.random([dim, dim]))
    A += 1j * np.random.random([dim, dim])
    A += A.T

    return A





class TestHafnian:
    """bechmark tests for hafnian calculations
    """

    def test_power_trace(self):
        """Calculate the hafnian of a 40x40 random complex matrix"""

        # generate the random matrix
        A = generate_random_matrix(40)


        # calculate the hafnian with the power trace method using the piquasso library
        start = time.time()  
        hafnian_calculator_Cpp = PowerTraceHafnian( A )
        hafnian_Cpp = hafnian_calculator_Cpp.calculate()
        time_Cpp = time.time() - start
       




        print(' ')
        print('*******************************************')
        print('Time elapsed with the power trace method: ' + str(time_Cpp))
        print(' ')
        print(' ')


