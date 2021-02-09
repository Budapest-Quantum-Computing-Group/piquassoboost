#define PY_SSIZE_T_CLEAN

#define CPYTHON

#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "dot.h"
#include "tbb/scalable_allocator.h"


void capsule_cleanup(PyObject* capsule) {

    void *memory = PyCapsule_GetPointer(capsule, NULL);
    // I'm going to assume your memory needs to be freed with free().
    // If it needs different cleanup, perform whatever that cleanup is
    // instead of calling free().
    scalable_aligned_free(memory);


}


PyObject* array_from_ptr(void * ptr, int dim, npy_intp* shape, int np_type) {

        // create numpy array
        PyObject* arr = PyArray_SimpleNewFromData(dim, shape, np_type, ptr);

        // set memory keeper for the numpy array
        PyObject *capsule = PyCapsule_New(ptr, NULL, capsule_cleanup);
        PyArray_SetBaseObject((PyArrayObject *) arr, capsule);
        
        return arr;

}



/**
r"""
        Call to make a numpy array from an external pic::matrix class.

        Args:
        mtx (matrix&): a matrix instance
*/
PyObject* matrix_to_numpy( pic::matrix &mtx ) {
 
     
 
        npy_intp shape[2];
        shape[0] = (npy_intp) mtx.rows;
        shape[1] = (npy_intp) mtx.cols;

        pic::Complex16* data = mtx.get_data();
        return array_from_ptr( (void*) data, 2, shape, NPY_COMPLEX128);


}




/**
@brief Call to create a PIC matrix representation of a numpy array
*/
pic::matrix 
numpy2matrix(PyObject *arr) {

    // test C-style contiguous memory allocation of the arrays
    if ( !PyArray_IS_C_CONTIGUOUS(arr) ) {
        std::cout << "array is not memory contiguous" << std::endl;
    }

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
        std::cout << "numpy2matrix: Wrong matrix dimension was given" << std::endl;
        return pic::matrix(0,0);
    }



} 




extern "C"
{




/**
@brief Wrapper function to call the simulate method of C++ class CGaussianState
@param self A pointer pointing to an instance of the class GaussianState_Wrapper.
@param args A tuple of the input arguments: ??????????????
*/
static PyObject *
dot_wrapper_dot(PyObject *self, PyObject *args)
{

    // initiate variables for input arguments
    PyObject *A_arg = NULL;
    PyObject *B_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OO",
                                     &A_arg, &B_arg))
        return Py_BuildValue("i", -1);

  
    // establish memory contiguous arrays for C calculations
    PyObject* A = NULL, *B = NULL;  

    if ( PyArray_IS_C_CONTIGUOUS(A_arg) ) {
        A = A_arg;
        Py_INCREF(A); 
    }
    else {
        A = PyArray_FROM_OTF(A_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(B_arg) ) {
        B = B_arg;
        Py_INCREF(B); 
    }
    else {
        B = PyArray_FROM_OTF(B_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix A_mtx = numpy2matrix(A);     
    pic::matrix B_mtx = numpy2matrix(B);     

    // calculate the matrix product on the C++ side
    pic::matrix C_mtx = dot(A_mtx, B_mtx);

    // release C++ matrix from the ownership of the data. (Python would handle the release of the data)
    C_mtx.set_owner( false );      

    // convert C++ matrix into numpy array
    PyObject* C = matrix_to_numpy( C_mtx );


    Py_DECREF(A);   
    Py_DECREF(B);  


    return C;//Py_BuildValue("i", -1);;
}


static PyMethodDef dot_wrapper_Methods[] = {
    {"dot", (PyCFunction) dot_wrapper_dot, METH_VARARGS,
     "Method to calculate the product of two matrices"
    },
    {NULL}  /* Sentinel */
};



/**
@brief Structure containing metadata about the module.
*/
static PyModuleDef dot_wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "dot_wrapper",
    .m_doc = "Python binding for calculating matrix product by C++ code",
    .m_size = -1,
    dot_wrapper_Methods 
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_dot_wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;

    m = PyModule_Create(&dot_wrapper_Module);
    if (m == NULL)
        return NULL;


    return m;
}



}
