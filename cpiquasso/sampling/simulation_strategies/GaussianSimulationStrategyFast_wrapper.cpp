#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "GaussianSimulationStrategyFast.h"
#include "tbb/scalable_allocator.h"
#include "numpy_interface.h"



/**
@brief Type definition of the GaussianSimulationStrategyFast_wrapper Python class of the GaussianSimulationStrategyFast_wrapper module
*/
typedef struct GaussianSimulationStrategyFast_wrapper {
    PyObject_HEAD
    /// pointer to numpy matrix of the correlation matrix to keep it alive
    PyObject *covariance_matrix = NULL;
    /// pointer to numpy matrix to the displacement to keep it alive
    PyObject *m = NULL;
    /// The C++ variant of class GaussianSimulationStrategyFast
    pic::GaussianSimulationStrategyFast* simulation_strategy = NULL;

} GaussianSimulationStrategyFast_wrapper;


/**
@brief Creates an instance of class ChinHuhPermanentCalculator and return with a pointer pointing to the class instance (C++ linking is needed)
@param covariance_matrix The covariance matrix describing the gaussian state
@param displacement The mean (displacement) of the Gaussian state
@param cutoff the Fock basis truncation.
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
pic::GaussianSimulationStrategyFast*
create_ChinHuhPermanentCalculator( pic::matrix &covariance_matrix_mtx, pic::matrix &displacement, const size_t& cutoff ) {

    return new pic::GaussianSimulationStrategyFast(covariance_matrix_mtx, displacement, cutoff );

}

/**
@brief Call to deallocate an instance of ChinHuhPermanentCalculator class
@param ptr A pointer pointing to an instance of ChinHuhPermanentCalculator class.
*/
void
release_ChinHuhPermanentCalculator( pic::GaussianSimulationStrategyFast*  instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}




extern "C"
{




/**
@brief Method called when a python instance of the class GaussianSimulationStrategyFast_wrapper is destroyed
@param self A pointer pointing to an instance of class GaussianSimulationStrategyFast_wrapper.
*/
static void
GaussianSimulationStrategyFast_wrapper_dealloc(GaussianSimulationStrategyFast_wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_ChinHuhPermanentCalculator( self->simulation_strategy );

    // release numpy arrays
    if (self->covariance_matrix != NULL) {
        Py_DECREF(self->covariance_matrix);
        self->covariance_matrix = NULL;
    }

    // release numpy arrays
    if (self->m != NULL) {
        Py_DECREF(self->m);
        self->m = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class GaussianSimulationStrategyFast_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class GaussianSimulationStrategyFast_wrapper.
*/
static PyObject *
GaussianSimulationStrategyFast_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    GaussianSimulationStrategyFast_wrapper *self;
    self = (GaussianSimulationStrategyFast_wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->covariance_matrix = NULL;
    self->m = NULL;
    self->simulation_strategy = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class GaussianSimulationStrategyFast_wrapper is initialized
@param self A pointer pointing to an instance of the class GaussianSimulationStrategyFast_wrapper.
@param args A tuple of the input arguments: qbit_num (integer)
qbit_num: the number of qubits spanning the operations
@param kwds A tuple of keywords
*/
static int
GaussianSimulationStrategyFast_wrapper_init(GaussianSimulationStrategyFast_wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"covariance_matrix", (char*)"m", (char*)"fock_cutoff", NULL};

    // initiate variables for input arguments
    PyObject *covariance_matrix_arg = NULL;
    PyObject *m_arg = NULL;
    int fock_cutoff = 0;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOi", kwlist,
                                     &covariance_matrix_arg, &m_arg, &fock_cutoff ))
        return -1;

    // convert python object array to numpy C API array
    if ( covariance_matrix_arg == NULL ) return -1;


    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(covariance_matrix_arg) && PyArray_TYPE(covariance_matrix_arg) == NPY_COMPLEX128) {
        self->covariance_matrix = covariance_matrix_arg;
        Py_INCREF(self->covariance_matrix);
    }
    else {
        self->covariance_matrix = PyArray_FROM_OTF(covariance_matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    // establish memory contiguous arrays for C calculations
    if ( m_arg != Py_None && PyArray_IS_C_CONTIGUOUS(m_arg) && PyArray_TYPE(covariance_matrix_arg) == NPY_COMPLEX128 ) {
        self->m = m_arg;
        Py_INCREF(self->m);
    }
    else if ( m_arg != Py_None ) {
        self->m = PyArray_FROM_OTF(m_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }
    else {
        self->m = m_arg;
        Py_INCREF(self->m);
    }


    // create PIC version of the input matrices
    pic::matrix covariance_matrix_mtx = numpy2matrix(self->covariance_matrix);
    pic::matrix m_mtx = numpy2matrix(self->m);

    // create instance of class ChinHuhPermanentCalculator
    self->simulation_strategy = create_ChinHuhPermanentCalculator( covariance_matrix_mtx, m_mtx, fock_cutoff );

    return 0;
}



/**
@brief Wrapper function to call the simulate method of C++ class CGaussianState
@param self A pointer pointing to an instance of the class GaussianState_Wrapper.
@param args A tuple of the input arguments: ??????????????
*/
static PyObject *
GaussianSimulationStrategyFast_wrapper_simulate(GaussianSimulationStrategyFast_wrapper *self, PyObject *args)
{

    // initiate variables for input arguments
    int sample_num = 0;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|i",
                                     &sample_num))
        return Py_BuildValue("i", -1);


    // call the C++ variant of the sampling method
    std::vector<pic::PicState_int64> samples = self->simulation_strategy->simulate(sample_num);


    // preallocate Python list to hold the calculated samples
    PyObject* PySamples = PyTuple_New( (Py_ssize_t) samples.size() );

    for ( int idx = 0; idx < samples.size(); idx++ ) {
        // release the C++ array from the ownership of the calculated data
        samples[idx].set_owner(false);

        // convert output samples to numpy arrays
        PyObject *PySample = PicState_int64_to_numpy( samples[idx] );
        PyTuple_SetItem(PySamples, idx, PySample);

    }

    return PySamples;

}



/**
@brief Method to get matrix covariance_matrix
*/
static PyObject *
GaussianSimulationStrategyFast_wrapper_getcovariance_matrix(GaussianSimulationStrategyFast_wrapper *self, void *closure)
{
    Py_INCREF(self->covariance_matrix);
    return self->covariance_matrix;
}

/**
@brief Method to set matrix covariance_matrix
*/
static int
GaussianSimulationStrategyFast_wrapper_setcovariance_matrix(GaussianSimulationStrategyFast_wrapper *self, PyObject *covariance_matrix_arg, void *closure)
{
    // set matrin on the Pyhon side
    Py_DECREF(self->covariance_matrix);

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(covariance_matrix_arg) ) {
        self->covariance_matrix = covariance_matrix_arg;
        Py_INCREF(self->covariance_matrix);
    }
    else {
        self->covariance_matrix = PyArray_FROM_OTF(covariance_matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    // create PIC version of the input matrices
    pic::matrix covariance_matrix_mtx = numpy2matrix(self->covariance_matrix);

    // update data on the C++ side
    self->simulation_strategy->Update_covariance_matrix( covariance_matrix_mtx );


    return 0;
}



static PyGetSetDef GaussianSimulationStrategyFast_wrapper_getsetters[] = {
    {"covariance_matrix", (getter) GaussianSimulationStrategyFast_wrapper_getcovariance_matrix, (setter) GaussianSimulationStrategyFast_wrapper_setcovariance_matrix,
     "covariance_matrix", NULL},
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class GaussianSimulationStrategyFast_wrapper.
*/
static PyMemberDef GaussianSimulationStrategyFast_wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef GaussianSimulationStrategyFast_wrapper_Methods[] = {
    {"simulate", (PyCFunction) GaussianSimulationStrategyFast_wrapper_simulate, METH_VARARGS,
     "Method to calculate boson sampling output samples"
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class GaussianSimulationStrategyFast_wrapper.
*/
static PyTypeObject GaussianSimulationStrategyFast_wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "GaussianSimulationStrategyFast_wrapper.GaussianSimulationStrategyFast_wrapper", /*tp_name*/
  sizeof(GaussianSimulationStrategyFast_wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) GaussianSimulationStrategyFast_wrapper_dealloc, /*tp_dealloc*/
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
  GaussianSimulationStrategyFast_wrapper_Methods, /*tp_methods*/
  GaussianSimulationStrategyFast_wrapper_Members, /*tp_members*/
  GaussianSimulationStrategyFast_wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) GaussianSimulationStrategyFast_wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  GaussianSimulationStrategyFast_wrapper_new, /*tp_new*/
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

/**
@brief Structure containing metadata about the module.
*/
static PyModuleDef GaussianSimulationStrategyFast_wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "GaussianSimulationStrategyFast_wrapper",
    .m_doc = "Python binding for class ChinHuhPermanentCalculator",
    .m_size = -1,
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_GaussianSimulationStrategyFast_wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&GaussianSimulationStrategyFast_wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&GaussianSimulationStrategyFast_wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&GaussianSimulationStrategyFast_wrapper_Type);
    if (PyModule_AddObject(m, "GaussianSimulationStrategyFast_wrapper", (PyObject *) &GaussianSimulationStrategyFast_wrapper_Type) < 0) {
        Py_DECREF(&GaussianSimulationStrategyFast_wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
