import pytest
import numpy as np

from piquasso.gaussian.state import GaussianState
from piquasso import constants
import random



def tolerance():
    return 1E-9


def generate_symmetric_matrix(N):
    A = np.random.rand(N, N)

    return A + A.T



def generate_complex_symmetric_matrix(N):
    real = generate_symmetric_matrix(N)
    imaginary = generate_symmetric_matrix(N)

    return real + 1j * imaginary


def generate_unitary_matrix(N):
    from scipy.stats import unitary_group

    return np.array(unitary_group.rvs(N), dtype=complex)



def generate_hermitian_matrix(N):
    from scipy.linalg import logm

    U = generate_unitary_matrix(N)

    return 1j * logm(U)



class TestGaussianState:

    def test_Apply_to_C_and_g(self):

   
        C_size = 10
        C = np.ascontiguousarray(generate_hermitian_matrix(C_size))
        G = np.ascontiguousarray(generate_complex_symmetric_matrix(C_size))
        mean = np.ascontiguousarray(np.random.rand(C_size) + 1j * np.random.rand(C_size))

        T_size = 4
        T = np.random.rand(T_size, T_size) + 1j * np.random.rand(T_size, T_size)
        if not T.flags['C_CONTIGUOUS']:
            T = np.ascontiguousarray(T)

        modes = list(random.sample(range(0, C_size), T_size))
        modes.sort()

        import time
        loop_max = 100

        # calculate the transformation by numpy calculations
        time_numpy = 10
        for iter in range(0, loop_max ):

            C_numpy = np.copy(C)
            G_numpy = np.copy(G)

            start = time.time()  
            columns = np.array([modes] * len(modes))
            rows = columns.transpose()
            index = rows, columns


            C_numpy[index] = (
               T.conjugate() @ C_numpy[index] @ T.transpose()
            )
            G_numpy[index] = (
                T @ G_numpy[index] @ T.transpose()
            )

            all_other_modes = np.delete(np.arange(C_size), modes)
            if all_other_modes.any():
                all_other_rows = np.array([modes] * len(all_other_modes)).transpose()

                C_numpy[all_other_rows, all_other_modes] = (
                    T.conjugate() @ C_numpy[all_other_rows, all_other_modes]
                )

                G_numpy[all_other_rows, all_other_modes] = (
                    T @ G_numpy[all_other_rows, all_other_modes]
                )

                C_numpy[:, modes] = np.conj(C_numpy[modes, :]).transpose()
                G_numpy[:, modes] = G_numpy[modes, :].transpose()

            time_loc = time.time() - start
            start = time.time()   
       
            if time_numpy > time_loc:
                time_numpy = time_loc



        # Calculating the transformation by the C++ library
        time_Cpp = 10
        for iter in range(0, loop_max ):

            # make copy of the numpy arrays
            C_cpp = np.copy( C )
            G_cpp = np.copy( G )

            # create an instance of the Gaussioan state
            state = GaussianState(
                    C=C_cpp,
                    G=G_cpp,
                    mean=mean,
            )

            # start measuring the time
            start = time.time()   
        
            # call the transformation
            state.apply_to_C_and_G(T, modes)

            # get the run time
            time_loc = time.time() - start
    
            if time_Cpp > time_loc:
                time_Cpp = time_loc


        print(' ')
        print('*******************************************')
        print('Time elapsed with numpy: ' + str(time_numpy))
        print('Time elapsed with C++: ' + str(time_Cpp))
        print( "speedup: " + str(time_numpy/time_Cpp) )
        print(' ')
        print(' ')

        print( 'Difference between numpy and C++ result for C: ' + str(np.linalg.norm(C_numpy - C_cpp)))
        print( 'Difference between numpy and C++ result for G: ' + str(np.linalg.norm(G_numpy - G_cpp)))

        #assert np.linalg.norm(C_numpy - C_cpp) < 1e-13
        #assert np.linalg.norm(G_numpy - G_cpp) < 1e-13


