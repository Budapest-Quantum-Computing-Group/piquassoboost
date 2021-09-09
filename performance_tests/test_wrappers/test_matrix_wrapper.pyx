# distutils: language = c++
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
cimport numpy as np
np.import_array()


import time

from piquasso.common.matrix cimport matrix # import the C++ version of class matrix
from piquasso.common.dot cimport dot # import the C++ version of the dot product
from piquasso.common.numpy_interface cimport numpy_interface
from scipy.stats import unitary_group
import random
from libcpp cimport bool #import bool type
import time


# create an interface to numpy-C++ wrapper functions
cdef numpy_interface np_interface = numpy_interface()


cdef class wrapper_for_test_matrix:
    """This is a class to test low level algorithms of class matrix."""

    cdef int matrix_size
    cdef int iteration_num


    def __init__(self, matrix_size=120):

        self.matrix_size = matrix_size
        self.iteration_num = 100

    def dot_test_function(self):
        r"""
        This method tests the functionality of calculating the dot product A*B of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A*B')


        # creating a numpy matrices
        A_numpy = unitary_group.rvs(self.matrix_size)
        B_numpy = unitary_group.rvs(self.matrix_size)
        A_numpy[0,1] = 0.2
        B_numpy[0,1] = 0.2
        #B_numpy = np.delete(B_numpy, 0,1)
        #B_numpy = np.delete(B_numpy, 0,1)

        # make input arrays contigous in the memory for C if needed
        if not A_numpy.flags['C_CONTIGUOUS']:
            A_numpy = np.ascontiguousarray(A_numpy)

        # make input arrays contigous in the memory for C if needed
        if not B_numpy.flags['C_CONTIGUOUS']:
            B_numpy = np.ascontiguousarray(B_numpy)


        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cA_numpy
        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cB_numpy
        cdef matrix A_mtx
        cdef matrix B_mtx
        cdef matrix C_mtx


        time_python = 10
        time_Cpp = 10

        for idx in range(self.iteration_num):
            # creating the C++ variant of the numpy array


      
            # creating C-type objects of matrices
            cA_numpy = A_numpy
            shape = A_numpy.shape
            A_mtx = matrix( <double complex*> cA_numpy.data, shape[0], shape[1])

            cB_numpy = B_numpy
            shape = B_numpy.shape
            B_mtx = matrix( <double complex*> cB_numpy.data, shape[0], shape[1])

            # calculate the matrix product by C++ library
            start = time.time()
            C_mtx = dot(A_mtx, B_mtx)
            time_loc = time.time() - start

            # transforming the resulted C++ type matrix to numpy array
            C_mtx.set_owner(False)
            C = np_interface.matrix_to_numpy( C_mtx )

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



    def dot_test_function2(self):
        r"""
        This method tests the functionality of calculating the dot product A^T * B of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A^T * B')

        # creating a numpy matrices
        A_numpy = unitary_group.rvs(self.matrix_size)
        B_numpy = unitary_group.rvs(self.matrix_size)
        #A_numpy = np.delete(A_numpy, 0,1)

        # make input arrays contigous in the memory for C if needed
        if not A_numpy.flags['C_CONTIGUOUS']:
            A_numpy = np.ascontiguousarray(A_numpy)

        # make input arrays contigous in the memory for C if needed
        if not B_numpy.flags['C_CONTIGUOUS']:
            B_numpy = np.ascontiguousarray(B_numpy)


        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cA_numpy
        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cB_numpy
        cdef matrix A_mtx
        cdef matrix B_mtx
        cdef matrix C_mtx


        time_python = 10
        time_Cpp = 10

        for idx in range(self.iteration_num):
            # creating the C++ variant of the numpy array


      
            # creating C-type objects of matrices
            cA_numpy = A_numpy
            shape = A_numpy.shape
            A_mtx = matrix( <double complex*> cA_numpy.data, shape[0], shape[1])
            A_mtx.transpose()

            cB_numpy = B_numpy
            shape = B_numpy.shape
            B_mtx = matrix( <double complex*> cB_numpy.data, shape[0], shape[1])

            # calculate the matrix product by C++ library
            start = time.time()
            C_mtx = dot(A_mtx, B_mtx)
            time_loc = time.time() - start

            # transforming the resulted C++ type matrix to numpy array
            C_mtx.set_owner(False)
            C = np_interface.matrix_to_numpy( C_mtx )

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




    def dot_test_function3(self):
        r"""
        This method tests the functionality of calculating the dot product A^+ * B of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A^+ * B')

        # creating a numpy matrices
        A_numpy = unitary_group.rvs(self.matrix_size)
        B_numpy = unitary_group.rvs(self.matrix_size)

        # make input arrays contigous in the memory for C if needed
        if not A_numpy.flags['C_CONTIGUOUS']:
            A_numpy = np.ascontiguousarray(A_numpy)

        # make input arrays contigous in the memory for C if needed
        if not B_numpy.flags['C_CONTIGUOUS']:
            B_numpy = np.ascontiguousarray(B_numpy)


        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cA_numpy
        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cB_numpy
        cdef matrix A_mtx
        cdef matrix B_mtx
        cdef matrix C_mtx


        time_python = 10
        time_Cpp = 10

        for idx in range(self.iteration_num):
            # creating the C++ variant of the numpy array


      
            # creating C-type objects of matrices
            cA_numpy = A_numpy
            shape = A_numpy.shape
            A_mtx = matrix( <double complex*> cA_numpy.data, shape[0], shape[1])
            A_mtx.transpose()
            A_mtx.conjugate()

            cB_numpy = B_numpy
            shape = B_numpy.shape
            B_mtx = matrix( <double complex*> cB_numpy.data, shape[0], shape[1])

            # calculate the matrix product by C++ library
            start = time.time()
            C_mtx = dot(A_mtx, B_mtx)
            time_loc = time.time() - start

            # transforming the resulted C++ type matrix to numpy array
            C_mtx.set_owner(False)
            C = np_interface.matrix_to_numpy( C_mtx )

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


    def dot_test_function4(self):
        r"""
        This method tests the functionality of calculating the dot product A^* * B of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A^* * B')

        # creating a numpy matrices
        A_numpy = unitary_group.rvs(self.matrix_size)
        B_numpy = unitary_group.rvs(self.matrix_size)

        # make input arrays contigous in the memory for C if needed
        if not A_numpy.flags['C_CONTIGUOUS']:
            A_numpy = np.ascontiguousarray(A_numpy)

        # make input arrays contigous in the memory for C if needed
        if not B_numpy.flags['C_CONTIGUOUS']:
            B_numpy = np.ascontiguousarray(B_numpy)


        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cA_numpy
        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cB_numpy
        cdef matrix A_mtx
        cdef matrix B_mtx
        cdef matrix C_mtx


        time_python = 10
        time_Cpp = 10

        for idx in range(self.iteration_num):
            # creating the C++ variant of the numpy array


      
            # creating C-type objects of matrices
            cA_numpy = A_numpy
            shape = A_numpy.shape
            A_mtx = matrix( <double complex*> cA_numpy.data, shape[0], shape[1])
            A_mtx.conjugate()

            cB_numpy = B_numpy
            shape = B_numpy.shape
            B_mtx = matrix( <double complex*> cB_numpy.data, shape[0], shape[1])

            # calculate the matrix product by C++ library
            start = time.time()
            C_mtx = dot(A_mtx, B_mtx)
            time_loc = time.time() - start

            # transforming the resulted C++ type matrix to numpy array
            C_mtx.set_owner(False)
            C = np_interface.matrix_to_numpy( C_mtx )

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

   
        

    def dot_test_function5(self):
        r"""
        This method tests the functionality of calculating the dot product A * B^* of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A * B^*')

        # creating a numpy matrices
        A_numpy = unitary_group.rvs(self.matrix_size)
        B_numpy = unitary_group.rvs(self.matrix_size)

        # make input arrays contigous in the memory for C if needed
        if not A_numpy.flags['C_CONTIGUOUS']:
            A_numpy = np.ascontiguousarray(A_numpy)

        # make input arrays contigous in the memory for C if needed
        if not B_numpy.flags['C_CONTIGUOUS']:
            B_numpy = np.ascontiguousarray(B_numpy)


        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cA_numpy
        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cB_numpy
        cdef matrix A_mtx
        cdef matrix B_mtx
        cdef matrix C_mtx


        time_python = 10
        time_Cpp = 10

        for idx in range(self.iteration_num):
            # creating the C++ variant of the numpy array


      
            # creating C-type objects of matrices
            cA_numpy = A_numpy
            shape = A_numpy.shape
            A_mtx = matrix( <double complex*> cA_numpy.data, shape[0], shape[1])

            cB_numpy = B_numpy
            shape = B_numpy.shape
            B_mtx = matrix( <double complex*> cB_numpy.data, shape[0], shape[1])
            B_mtx.conjugate()

            # calculate the matrix product by C++ library
            start = time.time()
            C_mtx = dot(A_mtx, B_mtx)
            time_loc = time.time() - start

            # transforming the resulted C++ type matrix to numpy array
            C_mtx.set_owner(False)
            C = np_interface.matrix_to_numpy( C_mtx )

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




    def dot_test_function6(self):
        r"""
        This method tests the functionality of calculating the dot product A * B^T of two matrices and compares it to numpy. 
        It is called from a test file ../test_matrix.py runned by pytest

        """

        print(' ')
        print('*******************************************')
        print('Calculating matrix product A * B^T')

        # creating a numpy matrices
        A_numpy = unitary_group.rvs(self.matrix_size)
        B_numpy = unitary_group.rvs(self.matrix_size)

        # make input arrays contigous in the memory for C if needed
        if not A_numpy.flags['C_CONTIGUOUS']:
            A_numpy = np.ascontiguousarray(A_numpy)

        # make input arrays contigous in the memory for C if needed
        if not B_numpy.flags['C_CONTIGUOUS']:
            B_numpy = np.ascontiguousarray(B_numpy)


        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cA_numpy
        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cB_numpy
        cdef matrix A_mtx
        cdef matrix B_mtx
        cdef matrix C_mtx


        time_python = 10
        time_Cpp = 10

        for idx in range(self.iteration_num):
            # creating the C++ variant of the numpy array


      
            # creating C-type objects of matrices
            cA_numpy = A_numpy
            shape = A_numpy.shape
            A_mtx = matrix( <double complex*> cA_numpy.data, shape[0], shape[1])

            cB_numpy = B_numpy
            shape = B_numpy.shape
            B_mtx = matrix( <double complex*> cB_numpy.data, shape[0], shape[1])
            B_mtx.transpose()

            # calculate the matrix product by C++ library
            start = time.time()
            C_mtx = dot(A_mtx, B_mtx)
            time_loc = time.time() - start

            # transforming the resulted C++ type matrix to numpy array
            C_mtx.set_owner(False)
            C = np_interface.matrix_to_numpy( C_mtx )

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

