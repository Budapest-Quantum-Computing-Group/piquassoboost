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

#ifndef PowerTraceHafnianRecursive_wrapper_H
#define PowerTraceHafnianRecursive_wrapper_H


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "PowerTraceHafnianRecursive.h"
#include "numpy_interface.h"

#define Hybrid 0
#define Double 1
#define LongDouble 2

#ifdef __MPFR__
#define InfPrec 3
#endif

/**
This file contains the implementation of the python wrapper object for the C++ class PowerTraceHafnianRecursive_wrapper. 
It is included by the file Boson_Sampling_Utilities_wrapper.cpp
*/

/**
@brief Type definition of the PowerTraceHafnianRecursive_wrapper Python class of the PowerTraceHafnianRecursive_wrapper module
*/
typedef struct PowerTraceHafnianRecursive_wrapper {
    PyObject_HEAD
    int lib;
    /// pointer to numpy matrix to keep it alive (this stores the symmetric matrix for which the hafnian is calculated)
    PyObject *matrix = NULL;
    /// pointer to numpy matrix to keep it alive (this stores the occupancy of the individual modes)
    PyObject *occupancy = NULL;
    /// The C++ variant of class CPowerTraceHafnianRecursive
    union {
        pic::PowerTraceHafnianRecursiveHybrid* calculator;
        pic::PowerTraceHafnianRecursiveDouble* calculatorDouble;
        pic::PowerTraceHafnianRecursiveLongDouble* calculatorLongDouble;
#ifdef __MPFR__
        pic::PowerTraceHafnianRecursiveInf* calculatorInf;
#endif
    };    
} PowerTraceHafnianRecursive_wrapper;


/**
@brief Creates an instance of class PowerTraceHafnianRecursive and return with a pointer pointing to the class instance (C++ linking is needed)
@param matrix_mtx The matrix for which the hafnain should be calculated
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
pic::PowerTraceHafnianRecursiveHybrid*
create_PowerTraceHafnianRecursive( pic::matrix &matrix_mtx,  pic::PicState_int64& occupancy) {

    return new pic::PowerTraceHafnianRecursiveHybrid(matrix_mtx, occupancy);
}

/**
@brief Call to deallocate an instance of PowerTraceHafnianRecursive class
@param ptr A pointer pointing to an instance of PowerTraceHafnianRecursive class.
*/
void
release_PowerTraceHafnianRecursive( pic::PowerTraceHafnianRecursiveHybrid*  instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}





extern "C"
{




/**
@brief Method called when a python instance of the class PowerTraceHafnianRecursive_wrapper is destroyed
@param self A pointer pointing to an instance of class PowerTraceHafnianRecursive_wrapper.
*/
static void
PowerTraceHafnianRecursive_wrapper_dealloc(PowerTraceHafnianRecursive_wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition    
    if (self->lib == Hybrid)
        release_PowerTraceHafnianRecursive( self->calculator );
    else if (self->lib == Double)
        delete self->calculatorDouble;
    else if (self->lib == LongDouble)
        delete self->calculatorLongDouble;
#ifdef __MPFR__
    else if (self->lib == InfPrec)
        delete self->calculatorInf;
#endif

    if ( self->matrix != NULL ) {
        // release numpy arrays
        Py_DECREF(self->matrix);
        self->matrix = NULL;
    }

    if ( self->occupancy != NULL ) {
        // release numpy arrays
        Py_DECREF(self->occupancy);
        self->occupancy = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class PowerTraceHafnianRecursive_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class PowerTraceHafnianRecursive_wrapper.
*/
static PyObject *
PowerTraceHafnianRecursive_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PowerTraceHafnianRecursive_wrapper *self;
    self = (PowerTraceHafnianRecursive_wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->matrix = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class PowerTraceHafnianRecursive_wrapper is initialized
@param self A pointer pointing to an instance of the class PowerTraceHafnianRecursive_wrapper.
@param args A tuple of the input arguments: matrix (np.ndarray) The matrix for which the hafnain should be calculated
matrix: The matrix for which the hafnain should be calculated
@param kwds A tuple of keywords
*/
static int
PowerTraceHafnianRecursive_wrapper_init(PowerTraceHafnianRecursive_wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"lib", (char*)"matrix", (char*)"occupancy", NULL};

    // initiate variables for input arguments
    PyObject *matrix_arg = NULL;
    PyObject *occupancy_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iOO", kwlist,
                                     &self->lib, &matrix_arg, &occupancy_arg))
        return -1;

    // convert python object array to numpy C API array
    if ( matrix_arg == NULL ) return -1;

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) && PyArray_TYPE(matrix_arg) == NPY_COMPLEX128 ) {
        self->matrix = matrix_arg;
        Py_INCREF(self->matrix);
    }
    else {
        self->matrix = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // convert python object array to numpy C API array
    if ( occupancy_arg == NULL ) return -1;

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(occupancy_arg) && PyArray_TYPE(occupancy_arg) == NPY_INT64) {
        self->occupancy = occupancy_arg;
        Py_INCREF(self->occupancy);
    }
    else {
        self->occupancy = PyArray_FROM_OTF(occupancy_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(self->matrix);

    // create PIC version of the occupancy matrix
    pic::PicState_int64 occupancy_mtx = numpy2PicState_int64( self->occupancy);

    // create instance of class PowerTraceHafnianRecursive
    
    if (self->lib == Hybrid)
        self->calculator = create_PowerTraceHafnianRecursive( matrix_mtx, occupancy_mtx );
    else if (self->lib == Double)
        self->calculatorDouble = new pic::PowerTraceHafnianRecursiveDouble(matrix_mtx, occupancy_mtx);
    else if (self->lib == LongDouble)
        self->calculatorLongDouble = new pic::PowerTraceHafnianRecursiveLongDouble(matrix_mtx, occupancy_mtx);
#ifdef __MPFR__
    else if (self->lib == InfPrec)
        self->calculatorInf = new pic::PowerTraceHafnianRecursiveInf(matrix_mtx, occupancy_mtx);
#endif
    else {
        PyErr_SetString(PyExc_Exception, "Wrong value set for hafnian library.");
        return -1;
    }    

    return 0;
}



/**
@brief Wrapper function to call the calculate method of C++ class CPowerTraceHafnianRecursive
@param self A pointer pointing to an instance of the class PowerTraceHafnianRecursive_Wrapper.
@return Returns with a PyObject containing the calculated hafnian.
*/
static PyObject *
PowerTraceHafnianRecursive_Wrapper_calculate(PowerTraceHafnianRecursive_wrapper *self)
{

    // start the calculation of the hafnian
    pic::Complex16 ret = self->calculator->calculate();

    return Py_BuildValue("D", &ret);
}




/**
@brief Method to call get attribute matrix
@param self A pointer pointing to an instance of the class PowerTraceHafnianRecursive_Wrapper.
@param closure Set to NULL pointer
@return Returns with a PyObject containing matrix.
*/
static PyObject *
PowerTraceHafnianRecursive_wrapper_getmatrix(PowerTraceHafnianRecursive_wrapper *self, void *closure)
{
    Py_INCREF(self->matrix);
    return self->matrix;
}

/**
@brief Method to call set attribute matrix
@param self A pointer pointing to an instance of the class PowerTraceHafnianRecursive_Wrapper.
@param matrix_arg A PyObject containing the matrix.
@param closure Set to NULL pointer
@return Returns with 0 in case of success.
*/
static int
PowerTraceHafnianRecursive_wrapper_setmatrix(PowerTraceHafnianRecursive_wrapper *self, PyObject *matrix_arg, void *closure)
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
@brief list of set and get function implementations for the python object PowerTraceHafnianRecursive_wrapper
*/
static PyGetSetDef PowerTraceHafnianRecursive_wrapper_getsetters[] = {
    {"matrix", (getter) PowerTraceHafnianRecursive_wrapper_getmatrix, (setter) PowerTraceHafnianRecursive_wrapper_setmatrix,
     "matrix", NULL},
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class PowerTraceHafnianRecursive_wrapper.
*/
static PyMemberDef PowerTraceHafnianRecursive_wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef PowerTraceHafnianRecursive_wrapper_Methods[] = {
    {"calculate", (PyCFunction) PowerTraceHafnianRecursive_Wrapper_calculate, METH_NOARGS,
     "Method to calculate the hafnian."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class PowerTraceHafnianRecursive_wrapper.
*/
static PyTypeObject PowerTraceHafnianRecursive_wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "PowerTraceHafnianRecursive_wrapper.PowerTraceHafnianRecursive_wrapper", /*tp_name*/
  sizeof(PowerTraceHafnianRecursive_wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) PowerTraceHafnianRecursive_wrapper_dealloc, /*tp_dealloc*/
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
  PowerTraceHafnianRecursive_wrapper_Methods, /*tp_methods*/
  PowerTraceHafnianRecursive_wrapper_Members, /*tp_members*/
  PowerTraceHafnianRecursive_wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) PowerTraceHafnianRecursive_wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  PowerTraceHafnianRecursive_wrapper_new, /*tp_new*/
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



#endif //PowerTraceHafnianRecursive_wrapper
