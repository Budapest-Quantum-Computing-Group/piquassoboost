/**
 * Copyright 2021 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PowerTraceHafnian_wrapper_H
#define PowerTraceHafnian_wrapper_H


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "PowerTraceHafnian.h"
#include "numpy_interface.h"

/**
This file contains the implementation of the python wrapper object for the C++ class PowerTraceHafnian_wrapper.
It is included by the file Boson_Sampling_Utilities_wrapper.cpp
*/

/**
@brief Type definition of the PowerTraceHafnian_wrapper Python class of the PowerTraceHafnian_wrapper module
*/
typedef struct PowerTraceHafnian_wrapper {
    PyObject_HEAD
    /// pointer to numpy matrix to keep it alive
    PyObject *matrix = NULL;
    /// The C++ variant of class CPowerTraceHafnian
    pic::PowerTraceHafnian* calculator;
} PowerTraceHafnian_wrapper;


/**
@brief Creates an instance of class PowerTraceHafnian and return with a pointer pointing to the class instance (C++ linking is needed)
@param matrix_mtx The matrix for which the hafnain should be calculated
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
pic::PowerTraceHafnian*
create_PowerTraceHafnian( pic::matrix &matrix_mtx ) {

    return new pic::PowerTraceHafnian(matrix_mtx);
}

/**
@brief Call to deallocate an instance of PowerTraceHafnian class
@param ptr A pointer pointing to an instance of PowerTraceHafnian class.
*/
void
release_PowerTraceHafnian( pic::PowerTraceHafnian*  instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}





extern "C"
{




/**
@brief Method called when a python instance of the class PowerTraceHafnian_wrapper is destroyed
@param self A pointer pointing to an instance of class PowerTraceHafnian_wrapper.
*/
static void
PowerTraceHafnian_wrapper_dealloc(PowerTraceHafnian_wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_PowerTraceHafnian( self->calculator );

    // release numpy arrays
    Py_DECREF(self->matrix);

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class PowerTraceHafnian_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class PowerTraceHafnian_wrapper.
*/
static PyObject *
PowerTraceHafnian_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PowerTraceHafnian_wrapper *self;
    self = (PowerTraceHafnian_wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->matrix = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class PowerTraceHafnian_wrapper is initialized
@param self A pointer pointing to an instance of the class PowerTraceHafnian_wrapper.
@param args A tuple of the input arguments: matrix (np.ndarray) The matrix for which the hafnain should be calculated
matrix: The matrix for which the hafnain should be calculated
@param kwds A tuple of keywords
*/
static int
PowerTraceHafnian_wrapper_init(PowerTraceHafnian_wrapper *self, PyObject *args, PyObject *kwds)
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

    // create instance of class PowerTraceHafnian
    self->calculator = create_PowerTraceHafnian( matrix_mtx );

    return 0;
}



/**
@brief Wrapper function to call the calculate method of C++ class CPowerTraceHafnian
@param self A pointer pointing to an instance of the class PowerTraceHafnian_Wrapper.
@return Returns with a PyObject containing the calculated hafnian.
*/
static PyObject *
PowerTraceHafnian_Wrapper_calculate(PowerTraceHafnian_wrapper *self)
{

    // start the calculation of the hafnian
    pic::Complex16 ret = self->calculator->calculate();

    return Py_BuildValue("D", &ret);
}




/**
@brief Method to call get attribute matrix
@param self A pointer pointing to an instance of the class PowerTraceHafnian_Wrapper.
@param closure Set to NULL pointer
@return Returns with a PyObject containing matrix.
*/
static PyObject *
PowerTraceHafnian_wrapper_getmatrix(PowerTraceHafnian_wrapper *self, void *closure)
{
    Py_INCREF(self->matrix);
    return self->matrix;
}

/**
@brief Method to call set attribute matrix
@param self A pointer pointing to an instance of the class PowerTraceHafnian_Wrapper.
@param matrix_arg A PyObject containing the matrix.
@param closure Set to NULL pointer
@return Returns with 0 in case of success.
*/
static int
PowerTraceHafnian_wrapper_setmatrix(PowerTraceHafnian_wrapper *self, PyObject *matrix_arg, void *closure)
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
@brief list of set and get function implementations for the python object PowerTraceHafnian_wrapper
*/
static PyGetSetDef PowerTraceHafnian_wrapper_getsetters[] = {
    {"matrix", (getter) PowerTraceHafnian_wrapper_getmatrix, (setter) PowerTraceHafnian_wrapper_setmatrix,
     "matrix", NULL},
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class PowerTraceHafnian_wrapper.
*/
static PyMemberDef PowerTraceHafnian_wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef PowerTraceHafnian_wrapper_Methods[] = {
    {"calculate", (PyCFunction) PowerTraceHafnian_Wrapper_calculate, METH_NOARGS,
     "Method to calculate the hafnian."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class PowerTraceHafnian_wrapper.
*/
static PyTypeObject PowerTraceHafnian_wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "PowerTraceHafnian_wrapper.PowerTraceHafnian_wrapper", /*tp_name*/
  sizeof(PowerTraceHafnian_wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) PowerTraceHafnian_wrapper_dealloc, /*tp_dealloc*/
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
  PowerTraceHafnian_wrapper_Methods, /*tp_methods*/
  PowerTraceHafnian_wrapper_Members, /*tp_members*/
  PowerTraceHafnian_wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) PowerTraceHafnian_wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  PowerTraceHafnian_wrapper_new, /*tp_new*/
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



#endif //PowerTraceHafnian_wrapper
