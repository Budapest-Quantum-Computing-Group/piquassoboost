/**
 * Copyright 2022 Budapest Quantum Computing Group
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
#include "structmember.h"
#include "CGeneralizedCliffordsBLossySimulationStrategy.h"
#include "tbb/scalable_allocator.h"
#include "numpy_interface.h"
#ifdef _DFE_
#include "GlynnPermanentCalculatorDFE.h"
#endif


/**
@brief Type definition of the GeneralizedCliffordsBLossySimulationStrategy_wrapper Python class of the GeneralizedCliffordsBLossySimulationStrategy_wrapper module
*/
typedef struct GeneralizedCliffordsBLossySimulationStrategy_wrapper {
    PyObject_HEAD
    int lib;
    /// pointer to numpy matrix to keep it alive
    PyObject *interferometer_matrix = NULL;
    /// The number of approximated modes
    int number_of_approximated_modes = 0;
    /// The C++ variant of class CGeneralizedCliffordsBLossySimulationStrategy
    pic::CGeneralizedCliffordsBLossySimulationStrategy* simulation_strategy = NULL;

} GeneralizedCliffordsBLossySimulationStrategy_wrapper;


/**
@brief Creates an instance of class CGeneralizedCliffordsBLossySimulationStrategy and return with a pointer pointing to the class instance (C++ linking is needed)
@param interferometer_matrix
@param number_of_approximated_modes
@param lib
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
pic::CGeneralizedCliffordsBLossySimulationStrategy*
create_CGeneralizedCliffordsBLossySimulationStrategy( pic::matrix &interferometer_matrix_mtx, int number_of_approximated_modes, int lib ) {

    return new pic::CGeneralizedCliffordsBLossySimulationStrategy(interferometer_matrix_mtx, number_of_approximated_modes, lib);

}

/**
@brief Call to deallocate an instance of CGeneralizedCliffordsBLossySimulationStrategy class
@param ptr A pointer pointing to an instance of CGeneralizedCliffordsBLossySimulationStrategy class.
*/
void
release_CGeneralizedCliffordsBLossySimulationStrategy( pic::CGeneralizedCliffordsBLossySimulationStrategy *instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}




extern "C"
{




/**
@brief Method called when a python instance of the class GeneralizedCliffordsBLossySimulationStrategy_wrapper is destroyed
@param self A pointer pointing to an instance of class GeneralizedCliffordsBLossySimulationStrategy_wrapper.
*/
static void
GeneralizedCliffordsBLossySimulationStrategy_wrapper_dealloc(GeneralizedCliffordsBLossySimulationStrategy_wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_CGeneralizedCliffordsBLossySimulationStrategy( self->simulation_strategy );

#ifdef _DFE_
    dec_dfe_lib_count();
#endif

    // release numpy arrays
    if (self->interferometer_matrix != NULL)
        Py_DECREF(self->interferometer_matrix);

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class GeneralizedCliffordsBLossySimulationStrategy_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class GeneralizedCliffordsBLossySimulationStrategy_wrapper.
*/
static PyObject *
GeneralizedCliffordsBLossySimulationStrategy_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    GeneralizedCliffordsBLossySimulationStrategy_wrapper *self;
    self = (GeneralizedCliffordsBLossySimulationStrategy_wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->interferometer_matrix = NULL;
    self->simulation_strategy = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class GeneralizedCliffordsBLossySimulationStrategy_wrapper is initialized
@param self A pointer pointing to an instance of the class GeneralizedCliffordsBLossySimulationStrategy_wrapper.
@param args A tuple of the input arguments: qbit_num (integer)
qbit_num: the number of qubits spanning the operations
@param kwds A tuple of keywords
*/
static int
GeneralizedCliffordsBLossySimulationStrategy_wrapper_init(GeneralizedCliffordsBLossySimulationStrategy_wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {
        (char*)"interferometer_matrix",
        (char*)"approximated_modes",
        (char*)"seed",
        (char*)"lib",
        NULL};

    // initiate variables for input arguments
    PyObject *interferometer_matrix_arg = NULL;
    int number_of_approximated_modes = 0;
    PyObject *seed = NULL;

    // deafult value for the permanent library
    self->lib = GlynnRep;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OiOi", kwlist,
                                     &interferometer_matrix_arg, &number_of_approximated_modes, &seed, &self->lib))
        return -1;

    // convert python object array to numpy C API array
    if ( interferometer_matrix_arg == NULL ) return -1;



    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(interferometer_matrix_arg) && PyArray_TYPE(interferometer_matrix_arg) == NPY_COMPLEX128 ) {
        self->interferometer_matrix = interferometer_matrix_arg;
        Py_INCREF(self->interferometer_matrix);
    }
    else {
        self->interferometer_matrix = PyArray_FROM_OTF(interferometer_matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix interferometer_matrix_mtx = numpy2matrix(self->interferometer_matrix);

    // create instance of class CGeneralizedCliffordsBLossySimulationStrategy
    self->simulation_strategy = create_CGeneralizedCliffordsBLossySimulationStrategy(
        interferometer_matrix_mtx, number_of_approximated_modes, self->lib
    );

    // set custom seed for sampling
    if ( seed != NULL && seed != Py_None) {
        unsigned long long int seed_C = PyLong_AsUnsignedLongLong(seed);
        self->simulation_strategy->seed(seed_C);
    }

#ifdef _DFE_
        inc_dfe_lib_count();
#endif
    

    return 0;
}

static PyObject *
GeneralizedCliffordsBLossySimulationStrategy_wrapper_seed(GeneralizedCliffordsBLossySimulationStrategy_wrapper *self, PyObject *args)
{
    // initiate variables for input arguments
    unsigned long long seed = 0;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|K",
                                     &seed))
        return Py_BuildValue("i", -1);
    self->simulation_strategy->seed(seed);
    Py_INCREF(Py_None);
    return Py_None;
}

/**
@brief Wrapper function to call the simulate method of C++ class CGaussianState
@param self A pointer pointing to an instance of the class GaussianState_Wrapper.
@param args A tuple of the input arguments: ??????????????
*/
static PyObject *
GeneralizedCliffordsBLossySimulationStrategy_wrapper_simulate(GeneralizedCliffordsBLossySimulationStrategy_wrapper *self, PyObject *args)
{

    // initiate variables for input arguments
    PyObject *input_state_arg = NULL;
    int sample_num = 0;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|Oi",
                                     &input_state_arg, &sample_num))
        return Py_BuildValue("i", -1);

    // convert python object array to numpy C API array
    if ( input_state_arg == NULL ) return Py_BuildValue("i", -1);

    // establish memory contiguous arrays for C calculations
    PyObject* input_state = NULL;

    if ( PyArray_IS_C_CONTIGUOUS(input_state_arg) ) {
        input_state = input_state_arg;
        Py_INCREF(input_state);
    }
    else {
        input_state = PyArray_FROM_OTF(input_state_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::PicState_int64 input_state_mtx = numpy2PicState_int64(input_state);


    
    // call the C++ variant of the sampling method
    std::vector<pic::PicState_int64> samples;
    try {
        samples = self->simulation_strategy->simulate(input_state_mtx, sample_num);
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
    }


    // preallocate Python list to hold the calculated samples
    PyObject* PySamples = PyTuple_New( (Py_ssize_t) sample_num );


    for ( int idx = 0; idx < samples.size(); idx++ ) {
        // release the C++ array from the ownership of the calculated data
        samples[idx].set_owner(false);

        // convert output samples to numpy arrays
        PyObject *PySample = PicState_int64_to_numpy( samples[idx] );
        PyTuple_SetItem(PySamples, idx, PySample);

    }

    Py_DECREF(input_state);

    return PySamples;

}



/**
@brief Method to get matrix interferometer_matrix
*/
static PyObject *
GeneralizedCliffordsBLossySimulationStrategy_wrapper_getinterferometer_matrix(GeneralizedCliffordsBLossySimulationStrategy_wrapper *self, void *closure)
{
    Py_INCREF(self->interferometer_matrix);
    return self->interferometer_matrix;
}

/**
@brief Method to set matrix interferometer_matrix
*/
static int
GeneralizedCliffordsBLossySimulationStrategy_wrapper_setinterferometer_matrix(GeneralizedCliffordsBLossySimulationStrategy_wrapper *self, PyObject *interferometer_matrix_arg, void *closure)
{
    // set matrin on the Pyhon side
    Py_DECREF(self->interferometer_matrix);

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(interferometer_matrix_arg) ) {
        self->interferometer_matrix = interferometer_matrix_arg;
        Py_INCREF(self->interferometer_matrix);
    }
    else {
        self->interferometer_matrix = PyArray_FROM_OTF(interferometer_matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    // create PIC version of the input matrices
    pic::matrix interferometer_matrix_mtx = numpy2matrix(self->interferometer_matrix);

    // update data on the C++ side
    self->simulation_strategy->Update_interferometer_matrix( interferometer_matrix_mtx );


    return 0;
}



static PyGetSetDef GeneralizedCliffordsBLossySimulationStrategy_wrapper_getsetters[] = {
    {"interferometer_matrix", (getter) GeneralizedCliffordsBLossySimulationStrategy_wrapper_getinterferometer_matrix, (setter) GeneralizedCliffordsBLossySimulationStrategy_wrapper_setinterferometer_matrix,
     "interferometer_matrix", NULL},
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class GeneralizedCliffordsBLossySimulationStrategy_wrapper.
*/
static PyMemberDef GeneralizedCliffordsBLossySimulationStrategy_wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef GeneralizedCliffordsBLossySimulationStrategy_wrapper_Methods[] = {
    {"seed", (PyCFunction) GeneralizedCliffordsBLossySimulationStrategy_wrapper_seed, METH_VARARGS,
     "Method to set random number generator seed for boson sampling"
    },
    {"simulate", (PyCFunction) GeneralizedCliffordsBLossySimulationStrategy_wrapper_simulate, METH_VARARGS,
     "Method to calculate boson sampling output samples"
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class GeneralizedCliffordsBLossySimulationStrategy_wrapper.
*/
static PyTypeObject GeneralizedCliffordsBLossySimulationStrategy_wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "GeneralizedCliffordsBLossySimulationStrategy_wrapper.GeneralizedCliffordsBLossySimulationStrategy_wrapper", /*tp_name*/
  sizeof(GeneralizedCliffordsBLossySimulationStrategy_wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) GeneralizedCliffordsBLossySimulationStrategy_wrapper_dealloc, /*tp_dealloc*/
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
  GeneralizedCliffordsBLossySimulationStrategy_wrapper_Methods, /*tp_methods*/
  GeneralizedCliffordsBLossySimulationStrategy_wrapper_Members, /*tp_members*/
  GeneralizedCliffordsBLossySimulationStrategy_wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) GeneralizedCliffordsBLossySimulationStrategy_wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  GeneralizedCliffordsBLossySimulationStrategy_wrapper_new, /*tp_new*/
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
static PyModuleDef GeneralizedCliffordsBLossySimulationStrategy_wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "GeneralizedCliffordsBLossySimulationStrategy_wrapper",
    .m_doc = "Python binding for class CGeneralizedCliffordsBLossySimulationStrategy",
    .m_size = -1,
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_GeneralizedCliffordsBLossySimulationStrategy_wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&GeneralizedCliffordsBLossySimulationStrategy_wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&GeneralizedCliffordsBLossySimulationStrategy_wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&GeneralizedCliffordsBLossySimulationStrategy_wrapper_Type);
    if (PyModule_AddObject(m, "GeneralizedCliffordsBLossySimulationStrategy_wrapper", (PyObject *) &GeneralizedCliffordsBLossySimulationStrategy_wrapper_Type) < 0) {
        Py_DECREF(&GeneralizedCliffordsBLossySimulationStrategy_wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
