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
#include "CalculationHelpers.h"


static PyObject *
apply_passive_linear_to_C_and_G(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {
        (char*)"C", (char*)"G", (char*)"T", (char*)"modes",
        NULL
    };

    PyObject *C = NULL;
    PyObject *G = NULL;
    PyObject *T = NULL;
    PyObject *modes = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO", kwlist,
                                     &C, &G, &T, &modes))
        return Py_BuildValue("i", -1);

    if ( C == NULL ) return Py_BuildValue("i", -1);
    if ( G == NULL ) return Py_BuildValue("i", -1);

    if (PyArray_IS_C_CONTIGUOUS(C))
    {
        Py_INCREF(C);
    }
    else
    {
        C = PyArray_FROM_OTF(C, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if (PyArray_IS_C_CONTIGUOUS(G))
    {
        Py_INCREF(G);
    }
    else
    {
        G = PyArray_FROM_OTF(G, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if (PyArray_IS_C_CONTIGUOUS(T))
    {
        Py_INCREF(T);
    }
    else
    {
        T = PyArray_FROM_OTF(T, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    pic::matrix C_mtx = numpy2matrix(C);
    pic::matrix G_mtx = numpy2matrix(G);
    pic::matrix T_mtx = numpy2matrix(T);

    Py_ssize_t element_num = PyTuple_GET_SIZE(modes);

    std::vector<size_t> modes_vector;

    modes_vector.reserve((int) element_num);

    for (Py_ssize_t idx = 0; idx < element_num; idx++)
        modes_vector.push_back((int) PyLong_AsLong(PyTuple_GetItem(modes, idx)));

    pic::apply_to_C_and_G(C_mtx, G_mtx, T_mtx, modes_vector);

    Py_DECREF(C);
    Py_DECREF(G);
    Py_DECREF(T);

    return Py_BuildValue("i", 0);
}


extern "C"
{

static PyMethodDef calculation_helper_functions[] = {
    {
        "apply_passive_linear_to_C_and_G",
        (PyCFunction)(void(*)(void)) apply_passive_linear_to_C_and_G,
        METH_VARARGS | METH_KEYWORDS,
        "Apply passive linear to the C and G matrices."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef calculations_module = {
    PyModuleDef_HEAD_INIT,
    "calculation_extension",
    NULL,
    -1,
    calculation_helper_functions
};

PyMODINIT_FUNC
PyInit_calculation_extension(void)
{
    import_array();
    return PyModule_Create(&calculations_module);
}


} // extern "C"