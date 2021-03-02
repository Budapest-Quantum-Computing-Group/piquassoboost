import cpiquasso.common.dot_wrapper as dot
from scipy.stats import unitary_group
import numpy as np
import random
import time

class Test_Dot:
    """Call to test cases for multiplication of two matrices on the C++ side"""


    matrix_size = 400
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
        This method tests the functionality of calculating the dot product A * B^* of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A * B^*')


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
            C = dot.dot2( A_numpy, B_numpy)
            time_loc = time.time() - start      

            if time_Cpp > time_loc:
                time_Cpp = time_loc

        for idx in range(self.iteration_num):

            # calculate the matrix product by numpy
            start = time.time()
            C_numpy = np.dot( A_numpy, B_numpy.conjugate() )
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


    def test_dot3(self):
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
            start = time.time()
            C = dot.dot3( A_numpy, B_numpy)
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




    def test_dot4(self):
        r"""
        This method tests the functionality of calculating the dot product A^+ * B of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A^+ * B')


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
            C = dot.dot4( A_numpy, B_numpy)
            time_loc = time.time() - start      

            if time_Cpp > time_loc:
                time_Cpp = time_loc

        for idx in range(self.iteration_num):

            # calculate the matrix product by numpy
            start = time.time()
            C_numpy = np.dot( A_numpy.transpose().conjugate(), B_numpy )
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






    def test_dot5(self):
        r"""
        This method tests the functionality of calculating the dot product A^* * B of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A^* * B')


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
            C = dot.dot5( A_numpy, B_numpy)
            time_loc = time.time() - start      

            if time_Cpp > time_loc:
                time_Cpp = time_loc

        for idx in range(self.iteration_num):

            # calculate the matrix product by numpy
            start = time.time()
            C_numpy = np.dot( A_numpy.conjugate(), B_numpy )
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




    def test_dot6(self):
        r"""
        This method tests the functionality of calculating the dot product A * B^T of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A * B^T')


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
            C = dot.dot6( A_numpy, B_numpy)
            time_loc = time.time() - start      

            if time_Cpp > time_loc:
                time_Cpp = time_loc

        for idx in range(self.iteration_num):

            # calculate the matrix product by numpy
            start = time.time()
            C_numpy = np.dot( A_numpy, B_numpy.transpose() )
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



