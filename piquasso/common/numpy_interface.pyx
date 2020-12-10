# distutils: language = c++
#
# Copyright (C) 2020 by TODO - All rights reserved.
#


import numpy as np
cimport numpy as np
np.import_array()

from libcpp.vector cimport vector
from piquasso.common.matrix cimport matrix, matrix_int64
from piquasso.common.PicState cimport PicState_int64
from cpython cimport PyObject, Py_INCREF, Py_DECREF

# external declaration of the memory release function from TBB library
cdef extern from "tbb/scalable_allocator.h":
    void scalable_aligned_free (void* ptr)

cdef class Sentinel:
    """Object to watch over raw pointers when coated by a numpy array."""

    # the raw pointer to the data
    cdef void* ptr

    def __cinit__(self):
        r"""
        Default constructor of the class

        """
        self.ptr = NULL

    def __dealloc__(self):
        r"""
        This method is called when the numpy array loses its last reference. 
        The raw pointer should be released by method corresponding to the allocation.

        """
        scalable_aligned_free(self.ptr)

    @staticmethod
    cdef create(void* ptr):
        r"""
        This static method is called to make the class instance to watch over the raw pinter.

        Args:
            ptr (void*): a void pointer

        """
        cdef Sentinel keeper = Sentinel()
        keeper.ptr = ptr
        return keeper

# external declaration of the function PyArray_SetBaseObject from the Numpy API
cdef extern from "numpy/arrayobject.h":
    int PyArray_SetBaseObject(np.ndarray arr, PyObject *obj) except -1


cdef class numpy_interface:

    cdef np.ndarray PicState_int64_to_numpy(self, PicState_int64 &cstate ):
        r"""
        Call to make a numpy array from an external PicState_int64 class.

        Args:
        cstate (PicState_int64&): a PicState_int64 instance

        """

        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> cstate.cols

        cdef np.int64_t* data = cstate.get_data()
        return self.array_from_ptr( <void*> data, 1, shape, np.NPY_INT64 ) 




    cdef np.ndarray matrix_to_numpy(self, matrix &cmtx ):
        r"""
        Call to make a numpy array from an external matrix class.

        Args:
        cstate (PicState_int64&): a PicState_int64 instance

        """

        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> cmtx.rows
        shape[1] = <np.npy_intp> cmtx.cols

        cdef double complex* data = cmtx.get_data()
        return self.array_from_ptr( <void*> data, 2, shape, np.NPY_COMPLEX128 ) 


        

    cdef np.ndarray array_from_ptr(self, void * ptr, int dim, np.npy_intp* shape, int np_type):
        r"""
        Call to make a two-dimensional numpy array from a raw pinter.

        Args:
        ptr (void*): a void pointer
        rows (int) number of rows in the array
        cols (int) number of columns in the array
        np_type (int) The data type stored in the numpy array (see possible values at https://numpy.org/doc/1.17/reference/c-api.dtype.html)

        """


        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(dim, shape, np_type, ptr)

        keeper = Sentinel.create(ptr)
 
        Py_INCREF(keeper)

        PyArray_SetBaseObject(arr, <PyObject*>keeper)

        return arr
