import numpy as np

from cpiquasso.sampling.Boson_Sampling_Utilities import Torontonian, TorontonianRecursive
import time


def generate_random_matrix( dim ):
    """ creading positive definite selfadjoint matrix A of dimension dim and with eigenvalues between 0 and 1 """

    A = np.complex128(np.random.random([dim, dim]))
    A_adjoint = A.conj().T

    P = A @ A_adjoint
    P += np.identity(len(P))

    P_inverse = np.linalg.inv(P)
    
    return P_inverse





class TestTorontonian:
    """bechmark tests for torontonian calculations
    """

    def test_value(self):
        """Check torontonian of thewalrus and piquasso"""


        # generate the random matrix
        dim = 40
        mtxs = []

        iter_loops = 10
        for idx in range(iter_loops):
            A = generate_random_matrix(dim)
            mtxs.append(A)


        # calculate the torontonian using the piquasso library
        time_cpiq = 10000
        time_sum_cpiq = 0
        for idx in range(iter_loops):
            A = mtxs[idx]

            start = time.time()   

            #perform calculation
            cpiquasso_tor = Torontonian(A).calculate()

            time_loc = time.time() - start
            start = time.time()   
       
            if time_cpiq > time_loc:
                time_cpiq = time_loc
            
            time_sum_cpiq += time_loc


        # calculate the torontonian using the recursive algorithm of the piquasso library
        time_cpiq_recursive = 10000
        time_sum_cpiq_recursive = 0
        for idx in range(iter_loops):
            A = mtxs[idx]

            start = time.time()   

            #perform calculation
            cpiquasso_tor_recursive = TorontonianRecursive(A).calculate()

            time_loc = time.time() - start
            start = time.time()   
       
            if time_cpiq_recursive > time_loc:
                time_cpiq_recursive = time_loc
            
            time_sum_cpiq_recursive += time_loc
            
        
        
        print(' ')
        print('cpiquasso torontonian value: ' + str(cpiquasso_tor) )
        print('cpiquasso recursive torontonian value: ' + str(cpiquasso_tor_recursive) )


        print(' ')
        print('*******************************************')
        print('Time elapsed with piquasso: ' + str(time_cpiq))
        print('Time elapsed with piquasso recursive: ' + str(time_cpiq_recursive))
        print('All time elapsed with piquasso: ' + str(time_sum_cpiq))
        print('All time elapsed with piquasso recursive: ' + str(time_sum_cpiq_recursive))
        print( "speedup: " + str(time_cpiq/time_cpiq_recursive) )
        print( "overall speedup: " + str(time_sum_cpiq/time_sum_cpiq_recursive) )
        print(' ')
        print(' ')

        print( 'Relative difference between the two piquasso result: ' + str(abs(cpiquasso_tor_recursive-cpiquasso_tor)/abs(cpiquasso_tor)*100) + '%')
