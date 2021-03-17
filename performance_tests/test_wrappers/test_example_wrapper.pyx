# distutils: language = c++
#
# Copyright (C) 2020 by TODO - All rights reserved.
#


import numpy as np
cimport numpy as np
np.import_array()


import time

from piquasso.common.matrix cimport matrix # import the C++ version of class matrix
from piquasso.common.numpy_interface cimport numpy_interface
from libcpp.vector cimport vector
from scipy.stats import unitary_group
import random
from libcpp cimport bool #import bool type


#cimport TBB headers
cimport piquasso.common.tbb as tbb

# cimport the header of the class to be tested
from piquasso.gaussian.source.tasks_apply_to_C_and_G.extract_rows cimport Extract_Rows


# create an interface to numpy-C++ wrapper functions
cdef numpy_interface np_interface = numpy_interface()


cdef class wrapper_for_test_example:
    """This is an example class to demonstrate how to interface with a C++ class using the data structures of the piquasso framework."""


    def some_wrapped_test_function(self):
        r"""
        This method tests the functionality of the class Extract_Rows and compares it to numpy. 
        It is called from a test file ../test_example.py runned by pytest

        """

        # creating a numpy matrix
        matrix_size = 5
        numpy_matrix = unitary_group.rvs(matrix_size)

        # creating the C++ variant of the numpy array

        # make input arrays contigous in the memory for C if needed
        if not numpy_matrix.flags['C_CONTIGUOUS']:
            numpy_matrix = np.ascontiguousarray(numpy_matrix)

        
        # creating C-type structures referencing the transformation matrix        
        cdef np.ndarray[np.complex_t, ndim=2, mode='c'] cmatrix = numpy_matrix
        shape = numpy_matrix.shape
        cdef matrix matrix_mtx = matrix( <double complex*> cmatrix.data, shape[0], shape[1])


        # selecting rows 1,3,4 from the matrix (indices start with 0)
        row_indices = list(random.sample(range(0, matrix_size), matrix_size-1))
        row_indices.sort()
        cdef vector[size_t] crow_indices = row_indices

        
        # allocate memory for the extracted rows
        cdef matrix rows_mtx = matrix(crow_indices.size(), matrix_mtx.cols)

        # the ownership of the resulting data would be given to the python side
        rows_mtx.set_owner( False )

        

        # create an instance of C++ class to be tested
        cdef Extract_Rows extract_rows = Extract_Rows( matrix_mtx, rows_mtx, crow_indices) 

        # now call the test function to extract rows from the matrix
        extract_rows(tbb.continue_msg() )


        # transforming C++ type matrix to numpy array
        numpy_rows = np_interface.matrix_to_numpy( rows_mtx )



        # now calculate the difference between C++ and numpy
        diff = np.linalg.norm(numpy_rows - numpy_matrix[row_indices,:])

        print(' ')
        print('*******************************************')
        print( 'Difference between numpy and C++ result: ' + str(diff))

        assert diff < 1e-13
        


