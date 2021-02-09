import piquasso.common.dot_wrapper as dot
from scipy.stats import unitary_group
import numpy as np
import random
import time

class Test_Dot:
    """Call to test cases for multiplication of two matrices on the C++ side"""


    matrix_size = 20
    iteration_num = 100


    def test_dot(self):
        r"""
        This method tests the functionality of calculating the dot product A*B of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A*B')


        # creating a numpy matrices
        A_numpy = np.ascontiguousarray(unitary_group.rvs(self.matrix_size))
        B_numpy = np.ascontiguousarray(unitary_group.rvs(self.matrix_size))
        A_numpy[0,1] = 0.2
        B_numpy[0,1] = 0.2

        time_python = 10
        time_Cpp = 10

        # calculations on the C++ side
        for idx in range(self.iteration_num):

            start = time.time()
            C = dot.dot( A_numpy, B_numpy)
            time_loc = time.time() - start      

            if time_Cpp > time_loc:
                time_Cpp = time_loc

        for idx in range(self.iteration_num):

            # calculate the matrix product by numpy
            start = time.time()
            C_numpy = np.dot( A_numpy, B_numpy )
            time_loc = time.time() - start

            if time_python > time_loc:
                time_python = time_loc
      

        # now calculate the difference between C++ and numpy
        diff = np.linalg.norm(C_numpy - C)

        print( 'Difference between numpy and C++ result: ' + str(diff))
        print('Time elapsed with python: ' + str(time_python))
        print('Time elapsed with C++: ' + str(time_Cpp))
        print( "speedup: " + str(time_python/time_Cpp) )

        assert diff < 1e-13



    def test_dot2(self):
        r"""
        This method tests the functionality of calculating the dot product A^T * B of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A^T * B')


        # creating a numpy matrices
        A_numpy = np.ascontiguousarray(unitary_group.rvs(self.matrix_size))
        B_numpy = np.ascontiguousarray(unitary_group.rvs(self.matrix_size))
        A_numpy[0,1] = 0.2
        B_numpy[0,1] = 0.2

        time_python = 10
        time_Cpp = 10

        # calculations on the C++ side
        for idx in range(self.iteration_num):
            A = A_numpy.copy()

            start = time.time()
            C = dot.dot( A.transpose(), B_numpy)
            time_loc = time.time() - start      

            if time_Cpp > time_loc:
                time_Cpp = time_loc

        for idx in range(self.iteration_num):

            # calculate the matrix product by numpy
            start = time.time()
            C_numpy = np.dot( A_numpy.transpose(), B_numpy )
            time_loc = time.time() - start

            if time_python > time_loc:
                time_python = time_loc
      

        # now calculate the difference between C++ and numpy
        diff = np.linalg.norm(C_numpy - C)

        print( 'Difference between numpy and C++ result: ' + str(diff))
        print('Time elapsed with python: ' + str(time_python))
        print('Time elapsed with C++: ' + str(time_Cpp))
        print( "speedup: " + str(time_python/time_Cpp) )

        assert diff < 1e-13




    def ctest_dote(self):
        r"""
        This method test the dot product of two matrices. 
        The call is forwarded to a cython class to test some functionalities of a C++ class. 

        """

        # creating an instance of wrapping class to call the test function by pytest
        wrapper_for_matrix_instance = wrapper_for_test_matrix( matrix_size = 16 )

        # call the test function A*B
        wrapper_for_matrix_instance.dot_test_function()

        # call the test function A * B^*
        wrapper_for_matrix_instance.dot_test_function5()

        # call the test function A^T * B
        wrapper_for_matrix_instance.dot_test_function2()

        # call the test function A^+ * B
        wrapper_for_matrix_instance.dot_test_function3()

        # call the test function A^* * B
        wrapper_for_matrix_instance.dot_test_function4()

        # call the test function A * B^T
        wrapper_for_matrix_instance.dot_test_function6()



