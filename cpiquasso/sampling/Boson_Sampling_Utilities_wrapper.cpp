#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include "ChinHuhPermanentCalculator_Wrapper.hpp"
#include "PowerTraceHafnian_Wrapper.hpp"
#include "PowerTraceHafnianRecursive_Wrapper.hpp"
#include "PowerTraceLoopHafnian_Wrapper.hpp"
#include "PowerTraceLoopHafnianRecursive_Wrapper.hpp"


extern "C"
{


/**
@brief Structure containing metadata about the module.
*/
static PyModuleDef Boson_Sampling_Utilities_wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "Boson_Sampling_Utilities_wrapper",
    .m_doc = "Python binding for class Boson_Sampling_Utilities",
    .m_size = -1,
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_Boson_Sampling_Utilities_wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;

    if (PyType_Ready(&ChinHuhPermanentCalculator_wrapper_Type) < 0)
        return NULL;

    if (PyType_Ready(&PowerTraceHafnian_wrapper_Type) < 0)
        return NULL;

    if (PyType_Ready(&PowerTraceHafnianRecursive_wrapper_Type) < 0)
        return NULL;

    if (PyType_Ready(&PowerTraceLoopHafnian_wrapper_Type) < 0)
        return NULL;

    if (PyType_Ready(&PowerTraceLoopHafnianRecursive_wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&Boson_Sampling_Utilities_wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&ChinHuhPermanentCalculator_wrapper_Type);
    if (PyModule_AddObject(m, "ChinHuhPermanentCalculator_wrapper", (PyObject *) &ChinHuhPermanentCalculator_wrapper_Type) < 0) {
        Py_DECREF(&ChinHuhPermanentCalculator_wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }


    Py_INCREF(&PowerTraceHafnian_wrapper_Type);
    if (PyModule_AddObject(m, "PowerTraceHafnian_wrapper", (PyObject *) &PowerTraceHafnian_wrapper_Type) < 0) {
        Py_DECREF(&PowerTraceHafnian_wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }


    Py_INCREF(&PowerTraceHafnianRecursive_wrapper_Type);
    if (PyModule_AddObject(m, "PowerTraceHafnianRecursive_wrapper", (PyObject *) &PowerTraceHafnianRecursive_wrapper_Type) < 0) {
        Py_DECREF(&PowerTraceHafnianRecursive_wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }


    Py_INCREF(&PowerTraceLoopHafnian_wrapper_Type);
    if (PyModule_AddObject(m, "PowerTraceLoopHafnian_wrapper", (PyObject *) &PowerTraceLoopHafnian_wrapper_Type) < 0) {
        Py_DECREF(&PowerTraceLoopHafnian_wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&PowerTraceLoopHafnianRecursive_wrapper_Type);
    if (PyModule_AddObject(m, "PowerTraceLoopHafnianRecursive_wrapper", (PyObject *) &PowerTraceLoopHafnianRecursive_wrapper_Type) < 0) {
        Py_DECREF(&PowerTraceLoopHafnianRecursive_wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



} //extern C
