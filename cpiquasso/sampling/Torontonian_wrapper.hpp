#ifndef Torontonian_wrapper_H
#define Torontonian_wrapper_H




#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "Torontonian.h"
#include "numpy_interface.h"

/**
This file contains the implementation of the python wrapper object for the C++ class Torontonian_wrapper. 
It is included by the file Torontonian_wrapper.cpp
*/

/**
@brief Type definition of the Torontonian_wrapper Python class of the Torontonian_wrapper module
*/
typedef struct Torontonian_wrapper {
    PyObject_HEAD
    /// pointer to numpy matrix to keep it alive
    PyObject *matrix = NULL;
    /// The C++ variant of class Torontonian
    pic::Torontonian* calculator;
} Torontonian_wrapper;


/**
@brief Creates an instance of class Torontonian and return with a pointer pointing to the class instance (C++ linking is needed)
@param matrix_mtx The matrix for which the hafnain should be calculated
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
pic::Torontonian*
create_Torontonian( pic::matrix &matrix_mtx ) {

    return new pic::Torontonian(matrix_mtx);
}

/**
@brief Call to deallocate an instance of Torontonian class
@param ptr A pointer pointing to an instance of Torontonian class.
*/
void
release_Torontonian( pic::Torontonian*  instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}





extern "C"
{




/**
@brief Method called when a python instance of the class Torontonian_wrapper is destroyed
@param self A pointer pointing to an instance of class Torontonian_wrapper.
*/
static void
Torontonian_wrapper_dealloc(Torontonian_wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_Torontonian( self->calculator );

    // release numpy arrays
    Py_DECREF(self->matrix);

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class Torontonian_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class Torontonian_wrapper.
*/
static PyObject *
Torontonian_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Torontonian_wrapper *self;
    self = (Torontonian_wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->matrix = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class Torontonian_wrapper is initialized
@param self A pointer pointing to an instance of the class Torontonian_wrapper.
@param args A tuple of the input arguments: matrix (np.ndarray) The matrix for which the Torontonian should be calculated.
@param kwds A tuple of keywords
*/
static int
Torontonian_wrapper_init(Torontonian_wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"matrix", NULL};

    // initiate variables for input arguments
    PyObject *matrix_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist,
                                     &matrix_arg))
        return -1;

    // convert python object array to numpy C API array
    if ( matrix_arg == NULL ) return -1;

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg)  && PyArray_TYPE(matrix_arg) == NPY_COMPLEX128 ) {
        self->matrix = matrix_arg;
        Py_INCREF(self->matrix);
    }
    else {
        self->matrix = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(self->matrix);

    // create instance of class Torontonian
    self->calculator = create_Torontonian( matrix_mtx );

    return 0;
}



/**
@brief Wrapper function to call the calculate method of C++ class Torontonian
@param self A pointer pointing to an instance of the class Torontonian_wrapper.
@return Returns with a PyObject containing the calculated torontonian.
*/
static PyObject *
Torontonian_wrapper_calculate(Torontonian_wrapper *self)
{

    // start the calculation of the permanent
    pic::Complex16 ret = self->calculator->calculate();

    return Py_BuildValue("D", &ret);
}




/**
@brief Method to call get attribute matrix
@param self A pointer pointing to an instance of the class Torontonian_wrapper.
@param closure Set to NULL pointer
@return Returns with a PyObject containing matrix.
*/
static PyObject *
Torontonian_wrapper_getmatrix(Torontonian_wrapper *self, void *closure)
{
    Py_INCREF(self->matrix);
    return self->matrix;
}

/**
@brief Method to call set attribute matrix
@param self A pointer pointing to an instance of the class Torontonian_wrapper.
@param matrix_arg A PyObject containing the matrix.
@param closure Set to NULL pointer
@return Returns with 0 in case of success.
*/
static int
Torontonian_wrapper_setmatrix(Torontonian_wrapper *self, PyObject *matrix_arg, void *closure)
{
    // set the array on the Python side
    Py_DECREF(self->matrix);

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        self->matrix = matrix_arg;
        Py_INCREF(self->matrix);
    }
    else {
        self->matrix = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(self->matrix);

    // update data on the C++ side
    self->calculator->Update_mtx( matrix_mtx );


    return 0;
}



/**
@brief list of set and get function implementations for the python object Torontonian_wrapper
*/
static PyGetSetDef Torontonian_wrapper_getsetters[] = {
    {"matrix", (getter) Torontonian_wrapper_getmatrix, (setter) Torontonian_wrapper_setmatrix,
     "matrix", NULL},
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class Torontonian_wrapper.
*/
static PyMemberDef Torontonian_wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef Torontonian_wrapper_Methods[] = {
    {"calculate", (PyCFunction) Torontonian_wrapper_calculate, METH_NOARGS,
     "Method to calculate the permanent."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class Torontonian_wrapper.
*/
static PyTypeObject Torontonian_wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "Torontonian_wrapper.Torontonian_wrapper", /*tp_name*/
  sizeof(Torontonian_wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) Torontonian_wrapper_dealloc, /*tp_dealloc*/
  #if PY_VERSION_HEX < 0x030800b4
  0, /*tp_print*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b4
  0, /*tp_vectorcall_offset*/
  #endif
  0, /*tp_getattr*/
  0, /*tp_setattr*/
  #if PY_MAJOR_VERSION < 3
  0, /*tp_compare*/
  #endif
  #if PY_MAJOR_VERSION >= 3
  0, /*tp_as_async*/
  #endif
  0, /*tp_repr*/
  0, /*tp_as_number*/
  0, /*tp_as_sequence*/
  0, /*tp_as_mapping*/
  0, /*tp_hash*/
  0, /*tp_call*/
  0, /*tp_str*/
  0, /*tp_getattro*/
  0, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "Object to represent a Operation_block class of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  Torontonian_wrapper_Methods, /*tp_methods*/
  Torontonian_wrapper_Members, /*tp_members*/
  Torontonian_wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) Torontonian_wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  Torontonian_wrapper_new, /*tp_new*/
  0, /*tp_free*/
  0, /*tp_is_gc*/
  0, /*tp_bases*/
  0, /*tp_mro*/
  0, /*tp_cache*/
  0, /*tp_subclasses*/
  0, /*tp_weaklist*/
  0, /*tp_del*/
  0, /*tp_version_tag*/
  #if PY_VERSION_HEX >= 0x030400a1
  0, /*tp_finalize*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b1
  0, /*tp_vectorcall*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
  0, /*tp_print*/
  #endif
};




} // extern C



#endif //Torontonian_wrapper_H
