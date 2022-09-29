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

#ifndef PowerTraceLoopHafnian_wrapper_H
#define PowerTraceLoopHafnian_wrapper_H


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "PowerTraceLoopHafnian.h"
#include "numpy_interface.h"

#define Hybrid 0
#define Double 1
#define LongDouble 2
#define InfPrec 3

/**
@brief Type definition of the PowerTraceLoopHafnian_wrapper Python class of the PowerTraceLoopHafnian_wrapper module
*/
typedef struct PowerTraceLoopHafnian_wrapper {
    PyObject_HEAD
    int lib;
    /// pointer to numpy matrix to keep it alive
    PyObject *matrix = NULL;
    /// The C++ variant of class CPowerTraceLoopHafnian
    union {
        pic::PowerTraceLoopHafnianHybrid* calculator;
        pic::PowerTraceLoopHafnianDouble* calculatorDouble;
        pic::PowerTraceLoopHafnianLongDouble* calculatorLongDouble;
        pic::PowerTraceLoopHafnianInf* calculatorInf;
    };    
} PowerTraceLoopHafnian_wrapper;


/**
@brief Creates an instance of class PowerTraceLoopHafnian and return with a pointer pointing to the class instance (C++ linking is needed)
@param C
@param G
@param mean
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
pic::PowerTraceLoopHafnianHybrid*
create_PowerTraceLoopHafnian( pic::matrix &matrix_mtx ) {

    return new pic::PowerTraceLoopHafnianHybrid(matrix_mtx);
}

/**
@brief Call to deallocate an instance of PowerTraceLoopHafnian class
@param ptr A pointer pointing to an instance of PowerTraceLoopHafnian class.
*/
void
release_PowerTraceLoopHafnian( pic::PowerTraceLoopHafnianHybrid*  instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}





extern "C"
{




/**
@brief Method called when a python instance of the class PowerTraceLoopHafnian_wrapper is destroyed
@param self A pointer pointing to an instance of class PowerTraceLoopHafnian_wrapper.
*/
static void
PowerTraceLoopHafnian_wrapper_dealloc(PowerTraceLoopHafnian_wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    if (self->lib == Hybrid)
        release_PowerTraceLoopHafnian( self->calculator );
    else if (self->lib == Double)
        delete self->calculatorDouble;
    else if (self->lib == LongDouble)
        delete self->calculatorLongDouble;
    else if (self->lib == InfPrec)
        delete self->calculatorInf;
    

    // release numpy arrays
    Py_DECREF(self->matrix);

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class PowerTraceLoopHafnian_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class PowerTraceLoopHafnian_wrapper.
*/
static PyObject *
PowerTraceLoopHafnian_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PowerTraceLoopHafnian_wrapper *self;
    self = (PowerTraceLoopHafnian_wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->matrix = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class PowerTraceLoopHafnian_wrapper is initialized
@param self A pointer pointing to an instance of the class PowerTraceLoopHafnian_wrapper.
@param args A tuple of the input arguments: qbit_num (integer)
qbit_num: the number of qubits spanning the operations
@param kwds A tuple of keywords
*/
static int
PowerTraceLoopHafnian_wrapper_init(PowerTraceLoopHafnian_wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"lib", (char*)"matrix", NULL};

    // initiate variables for input arguments
    PyObject *matrix_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO", kwlist,
                                     &self->lib, &matrix_arg))
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

    // create instance of class PowerTraceLoopHafnian
    if (self->lib == Hybrid)
        self->calculator = create_PowerTraceLoopHafnian( matrix_mtx );
    else if (self->lib == Double)
        self->calculatorDouble = new pic::PowerTraceLoopHafnianDouble(matrix_mtx);
    else if (self->lib == LongDouble)
        self->calculatorLongDouble = new pic::PowerTraceLoopHafnianLongDouble(matrix_mtx);
    else if (self->lib == InfPrec)
        self->calculatorInf = new pic::PowerTraceLoopHafnianInf(matrix_mtx);
    else {
        PyErr_SetString(PyExc_Exception, "Wrong value set for hafnian library.");
        return -1;
    }
    

    return 0;
}



/**
@brief Wrapper function to call the calculate method of C++ class CPowerTraceLoopHafnian
@param self A pointer pointing to an instance of the class PowerTraceLoopHafnian_Wrapper.
@param args A tuple of the input arguments: ??????????????
@param kwds A tuple of keywords
*/
static PyObject *
PowerTraceLoopHafnian_Wrapper_calculate(PowerTraceLoopHafnian_wrapper *self)
{

    // start the calculation of the loop hafnian
    pic::Complex16 ret = self->calculator->calculate();

    return Py_BuildValue("D", &ret);
}




/**
@brief Method to call get attribute matrix
*/
static PyObject *
PowerTraceLoopHafnian_wrapper_getmatrix(PowerTraceLoopHafnian_wrapper *self, void *closure)
{
    Py_INCREF(self->matrix);
    return self->matrix;
}

/**
@brief Method to call set attribute matrix
*/
static int
PowerTraceLoopHafnian_wrapper_setmatrix(PowerTraceLoopHafnian_wrapper *self, PyObject *matrix_arg, void *closure)
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





static PyGetSetDef PowerTraceLoopHafnian_wrapper_getsetters[] = {
    {"matrix", (getter) PowerTraceLoopHafnian_wrapper_getmatrix, (setter) PowerTraceLoopHafnian_wrapper_setmatrix,
     "matrix", NULL},
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class PowerTraceLoopHafnian_wrapper.
*/
static PyMemberDef PowerTraceLoopHafnian_wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef PowerTraceLoopHafnian_wrapper_Methods[] = {
    {"calculate", (PyCFunction) PowerTraceLoopHafnian_Wrapper_calculate, METH_NOARGS,
     "Method to calculate the loop hafnian."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class PowerTraceLoopHafnian_wrapper.
*/
static PyTypeObject PowerTraceLoopHafnian_wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "PowerTraceLoopHafnian_wrapper.PowerTraceLoopHafnian_wrapper", /*tp_name*/
  sizeof(PowerTraceLoopHafnian_wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) PowerTraceLoopHafnian_wrapper_dealloc, /*tp_dealloc*/
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
  PowerTraceLoopHafnian_wrapper_Methods, /*tp_methods*/
  PowerTraceLoopHafnian_wrapper_Members, /*tp_members*/
  PowerTraceLoopHafnian_wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) PowerTraceLoopHafnian_wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  PowerTraceLoopHafnian_wrapper_new, /*tp_new*/
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



#endif //PowerTraceLoopHafnian_wrapper
