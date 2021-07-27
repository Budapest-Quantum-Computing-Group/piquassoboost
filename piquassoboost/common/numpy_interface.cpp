#ifndef NUMPY_INTERFACE_H
#define NUMPY_INTERFACE_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include "matrix.h"
#include "matrix_real.h"
#include "PicState.h"


/**
@brief Method to cleanup the memory when the python object becomes released
@param capsule Pointer to the memory capsule
*/
void capsule_cleanup(PyObject* capsule) {

    void *memory = PyCapsule_GetPointer(capsule, NULL);
    // I'm going to assume your memory needs to be freed with free().
    // If it needs different cleanup, perform whatever that cleanup is
    // instead of calling free().
    scalable_aligned_free(memory);


}




/**
@brief Call to make a numpy array from data stored via void pointer.
@param ptr pointer pointing to the data
@param dim The number of dimensions
@param shape array containing the dimensions.
@param np_type The data type stored in the numpy array (see possible values at https://numpy.org/doc/1.17/reference/c-api.dtype.html)
*/
PyObject* array_from_ptr(void * ptr, int dim, npy_intp* shape, int np_type) {

        if (PyArray_API == NULL) {
            import_array();
        }


        // create numpy array
        PyObject* arr = PyArray_SimpleNewFromData(dim, shape, np_type, ptr);

        // set memory keeper for the numpy array
        PyObject *capsule = PyCapsule_New(ptr, NULL, capsule_cleanup);
        PyArray_SetBaseObject((PyArrayObject *) arr, capsule);

        return arr;

}



/**
@brief Call to make a numpy array from an instance of matrix class.
@param mtx a matrix instance
*/
PyObject* matrix_to_numpy( pic::matrix &mtx ) {
        // initialize Numpy API
        import_array();


        npy_intp shape[2];
        shape[0] = (npy_intp) mtx.rows;
        shape[1] = (npy_intp) mtx.cols;

        pic::Complex16* data = mtx.get_data();
        return array_from_ptr( (void*) data, 2, shape, NPY_COMPLEX128);


}


/**
@brief Call to make a numpy array from an external PicState_int64 class.
@param cstate a PicState_int64 instance
*/
PyObject* PicState_int64_to_numpy( pic::PicState_int64 &cstate ) {


        npy_intp shape[1];
        shape[0] = (npy_intp) cstate.cols;

        int64_t* data = cstate.get_data();
        return array_from_ptr( (void*) data, 1, shape, NPY_INT64);


}




/**
@brief Call to create a PIC matrix representation of a numpy array
*/
pic::matrix
numpy2matrix(PyObject *arr) {

    if ( arr == Py_None ) {
        return pic::matrix(0,0);
    }

#ifdef DEBUG
    // test C-style contiguous memory allocation of the arrays
    // in production this case has to be handled outside
    assert( PyArray_IS_C_CONTIGUOUS(arr) && "array is not memory contiguous" );
#endif

    // get the pointer to the data stored in the input matrices
    pic::Complex16* data = (pic::Complex16*)PyArray_DATA(arr);

    // get the dimensions of the array self->C
    int dim_num = PyArray_NDIM( arr );
    npy_intp* dims = PyArray_DIMS(arr);

    // create PIC version of the input matrices
    if (dim_num == 2) {
        pic::matrix mtx = pic::matrix(data, dims[0], dims[1]);
        return mtx;
    }
    else if (dim_num == 1) {
        pic::matrix mtx = pic::matrix(data, dims[0], 1);
        return mtx;
    }
    else {
        std::cerr << "numpy2matrix: Wrong matrix dimension was given" << std::endl;
        exit(EXIT_FAILURE);
    }



}


/**
@brief Call to create a PIC matrix_real representation of a numpy array
*/
pic::matrix_real
numpy2matrix_real(PyObject *arr) {


    if ( arr == Py_None ) {
        return pic::matrix_real(0,0);
    }

#ifdef DEBUG
    // test C-style contiguous memory allocation of the arrays
    // in production this case has to be handled outside
    assert( PyArray_IS_C_CONTIGUOUS(arr) && "array is not memory contiguous" );
#endif

    // get the pointer to the data stored in the input matrices
    double *data = (double *)PyArray_DATA(arr);

    // get the dimensions of the array self->C
    int dim_num = PyArray_NDIM( arr );
    npy_intp* dims = PyArray_DIMS(arr);

    // create PIC version of the input matrices
    if (dim_num == 2) {
        pic::matrix_real mtx = pic::matrix_real(data, dims[0], dims[1]);
        return mtx;
    }
    else if (dim_num == 1) {
        pic::matrix_real mtx = pic::matrix_real(data, dims[0], 1);
        return mtx;
    }
    else {
        std::cerr << "numpy2matrix_real: Wrong matrix dimension was given" << std::endl;
        exit(EXIT_FAILURE);
    }



}


/**
@brief Call to create a PicState_int64 representation of a numpy array
*/
pic::PicState_int64
numpy2PicState_int64(PyObject *arr) {


    // test C-style contiguous memory allocation of the arrays
    if ( !PyArray_IS_C_CONTIGUOUS(arr) ) {
        std::cout << "array is not memory contiguous" << std::endl;
    }

    // get the pointer to the data stored in the input matrices
    int64_t* data = (int64_t*)PyArray_DATA(arr);

    // get the dimensions of the array self->C
    int dim_num = PyArray_NDIM( arr );
    npy_intp* dims = PyArray_DIMS(arr);

    // create PIC version of the input matrices
    if (dim_num == 1) {
        pic::PicState_int64 state = pic::PicState_int64(data, dims[0]);
        return state;
    }
    else {
        std::cout << "numpy2PicState_int64: Wrong matrix dimension was given" << std::endl;
        return pic::PicState_int64(0);
    }



}




#endif
