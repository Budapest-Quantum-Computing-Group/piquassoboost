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


