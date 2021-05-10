#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include "Torontonian_wrapper.hpp"


extern "C"
{


/**
@brief Structure containing metadata about the module.
*/
static PyModuleDef Torontonian_wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "Torontonian_wrapper_Module",
    .m_doc = "Python binding for class Torontonian",
    .m_size = -1,
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_Torontonian_wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;

    if (PyType_Ready(&Torontonian_wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&Torontonian_wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&Torontonian_wrapper_Type);
    if (PyModule_AddObject(m, "Torontonian_wrapper", (PyObject *) &Torontonian_wrapper_Type) < 0) {
        Py_DECREF(&Torontonian_wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }


    return m;
}



} //extern C
