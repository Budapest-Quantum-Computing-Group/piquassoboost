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

#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include "numpy_interface.h"
#include "GlynnPermanentCalculatorRepeated.hpp"


/** Python interface method for calculating the permanent of a matrix based on 
 *  pic::GlynnPermanentCalculatorRepeatedLongDouble
 */
static PyObject *
permanent_CPU_repeated_long_double(PyObject *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"matrix", (char*)"input_state", (char*)"output_state", NULL};


    PyObject *matrix_arg = NULL;
    PyObject *input_state_arg = NULL;
    PyObject *output_state_arg = NULL;

    PyObject *matrix_pyobj = NULL;
    PyObject *input_state_pyobj = NULL;
    PyObject *output_state_pyobj = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO", kwlist,
                                     &matrix_arg, &input_state_arg, &output_state_arg))
        return -1;

    // convert python object array to numpy C API array
    if ( matrix_arg == NULL ) return -1;
    if ( input_state_arg == NULL ) return -1;
    if ( output_state_arg == NULL ) return -1;

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        matrix_pyobj = matrix_arg;
        Py_INCREF(matrix_pyobj);
    }
    else {
        matrix_pyobj = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(input_state_arg) ) {
        input_state_pyobj = input_state_arg;
        Py_INCREF(input_state_pyobj);
    }
    else {
        input_state_pyobj = PyArray_FROM_OTF(input_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(output_state_arg) ) {
        output_state_pyobj = output_state_arg;
        Py_INCREF(output_state_pyobj);
    }
    else {
        output_state_pyobj = PyArray_FROM_OTF(output_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the parameters
    pic::matrix matrix_cpp = numpy2matrix(matrix_pyobj);
    pic::PicState_int64 input_state = numpy2PicState_int64(input_state_pyobj);
    pic::PicState_int64 output_state = numpy2PicState_int64(output_state_pyobj);

    pic::GlynnPermanentCalculatorRepeatedLongDouble calculator = pic::GlynnPermanentCalculatorRepeatedLongDouble();


    pic::Complex16 ret = calculator.calculate(matrix_cpp, input_state, output_state);

    // release numpy arrays
    Py_DECREF(matrix_pyobj);
    Py_DECREF(input_state_pyobj);
    Py_DECREF(output_state_pyobj);

    return Py_BuildValue("D", &ret);
}


/** Python interface method for calculating the permanent of a matrix based on
 *  pic::GlynnPermanentCalculatorRepeatedDouble
 */
static PyObject *
permanent_CPU_repeated_double(PyObject *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"matrix", (char*)"input_state", (char*)"output_state", NULL};


    PyObject *matrix_arg = NULL;
    PyObject *input_state_arg = NULL;
    PyObject *output_state_arg = NULL;

    PyObject *matrix_pyobj = NULL;
    PyObject *input_state_pyobj = NULL;
    PyObject *output_state_pyobj = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO", kwlist,
                                     &matrix_arg, &input_state_arg, &output_state_arg))
        return -1;

    // convert python object array to numpy C API array
    if ( matrix_arg == NULL ) return -1;
    if ( input_state_arg == NULL ) return -1;
    if ( output_state_arg == NULL ) return -1;

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        matrix_pyobj = matrix_arg;
        Py_INCREF(matrix_pyobj);
    }
    else {
        matrix_pyobj = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(input_state_arg) ) {
        input_state_pyobj = input_state_arg;
        Py_INCREF(input_state_pyobj);
    }
    else {
        input_state_pyobj = PyArray_FROM_OTF(input_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(output_state_arg) ) {
        output_state_pyobj = output_state_arg;
        Py_INCREF(output_state_pyobj);
    }
    else {
        output_state_pyobj = PyArray_FROM_OTF(output_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the parameters
    pic::matrix matrix_cpp = numpy2matrix(matrix_pyobj);
    pic::PicState_int64 input_state = numpy2PicState_int64(input_state_pyobj);
    pic::PicState_int64 output_state = numpy2PicState_int64(output_state_pyobj);

    pic::GlynnPermanentCalculatorRepeatedDouble calculator = pic::GlynnPermanentCalculatorRepeatedDouble();


    pic::Complex16 ret = calculator.calculate(matrix_cpp, input_state, output_state);

    // release numpy arrays
    Py_DECREF(matrix_pyobj);
    Py_DECREF(input_state_pyobj);
    Py_DECREF(output_state_pyobj);

    return Py_BuildValue("D", &ret);
}



extern "C"
{


static PyMethodDef permanent_calculators_functions[] = {
    {
        "permanent_CPU_repeated_double",
        (PyCFunction)(void(*)(void)) permanent_CPU_repeated_double,
        METH_VARARGS | METH_KEYWORDS,
        "Calculate the permanent of the parameter matrix with double precision."
    },
    {
        "permanent_CPU_repeated_long_double",
        (PyCFunction)(void(*)(void)) permanent_CPU_repeated_long_double,
        METH_VARARGS | METH_KEYWORDS,
        "Calculate the permanent of the parameter matrix with long double precision."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef permanent_calculators_module = {
    PyModuleDef_HEAD_INIT,
    "permanent_calculators",
    NULL,
    -1,
    permanent_calculators_functions
};

PyMODINIT_FUNC
PyInit_permanent_calculators(void)
{
    import_array();
    return PyModule_Create(&permanent_calculators_module);
}




} //extern C
