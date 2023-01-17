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
#include "structmember.h"
#include "ThresholdBosonSampling.h"
#include "tbb/scalable_allocator.h"
#include "numpy_interface.h"



/**
@brief Type definition of the ThresholdBosonSampling_wrapper Python class of the ThresholdBosonSampling module
*/
typedef struct ThresholdBosonSampling_wrapper {
    PyObject_HEAD
    /// pointer to numpy matrix of the correlation matrix to keep it alive
    PyObject *covariance_matrix = NULL;
    /// pointer to numpy matrix to the displacement to keep it alive
    PyObject *m = NULL;
    /// The C++ variant of class ThresholdBosonSampling
    pic::ThresholdBosonSampling* simulation_strategy = NULL;

} ThresholdBosonSampling_wrapper;



/**
@brief Creates an instance of class ThresholdBosonSampling and return with a pointer pointing to the class instance (C++ linking is needed)
@param covariance_matrix The covariance matrix describing the gaussian state
@return Return with a void pointer pointing to the new instance of ThresholdBosonSampling class.
*/
pic::ThresholdBosonSampling*
create_ThresholdBosonSamplingCalculator( pic::matrix_real& covariance_matrix_mtx ) {

    return new pic::ThresholdBosonSampling(covariance_matrix_mtx);

}

/**
@brief Call to deallocate an instance of ThresholdBosonSampling calculator class
@param ptr A pointer pointing to an instance of ThresholdBosonSampling calculator class.
*/
void
release_ThresholdBosonSamplingCalculator( pic::ThresholdBosonSampling *instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}




extern "C"
{




/**
@brief Method called when a python instance of the class ThresholdBosonSampling_wrapper is destroyed
@param self A pointer pointing to an instance of class ThresholdBosonSampling_wrapper.
*/
static void
ThresholdBosonSampling_wrapper_dealloc(ThresholdBosonSampling_wrapper *self)
{

    // deallocate the instance of class ThresholdBosonSampling
    release_ThresholdBosonSamplingCalculator( self->simulation_strategy );

    // release numpy arrays
    if (self->covariance_matrix != NULL) {
        Py_DECREF(self->covariance_matrix);
        self->covariance_matrix = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class ThresholdBosonSampling_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class ThresholdBosonSampling_wrapper.
*/
static PyObject *
ThresholdBosonSampling_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    ThresholdBosonSampling_wrapper *self;
    self = (ThresholdBosonSampling_wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->covariance_matrix = NULL;
    self->simulation_strategy = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class ThresholdBosonSampling_wrapper is initialized
@param self A pointer pointing to an instance of the class ThresholdBosonSampling_wrapper.
@param args A tuple of the input arguments: covariance_matrix (matrix)
covariance_matrix: the covariance matrix of the current state
@param kwds A tuple of keywords
*/
static int
ThresholdBosonSampling_wrapper_init(ThresholdBosonSampling_wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"covariance_matrix", NULL};

    // initiate variables for input arguments
    PyObject *covariance_matrix_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist,
                                     &covariance_matrix_arg))
        return -1;

    // convert python object array to numpy C API array
    if ( covariance_matrix_arg == NULL ) return -1;

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(covariance_matrix_arg) && PyArray_TYPE(covariance_matrix_arg) == NPY_FLOAT64 ){
        self->covariance_matrix = covariance_matrix_arg;
        Py_INCREF(self->covariance_matrix);
    }
    else if (PyArray_TYPE(covariance_matrix_arg) == NPY_FLOAT64 ) {
        self->covariance_matrix = PyArray_FROM_OTF(covariance_matrix_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    }
    else {
        std::string err( "The covariance matrix should be real (given in float64 format)");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return -1;
    }

    // create PIC version of the input matrices
    pic::matrix_real covariance_matrix_pq = numpy2matrix_real(self->covariance_matrix);
    
    // create instance of class ThresholdBosonSampling_wrapperCalculator
    self->simulation_strategy = create_ThresholdBosonSamplingCalculator( covariance_matrix_pq );

    return 0;
}



/**
@brief Wrapper function to call the simulate method of C++ class ThresholdBosonSampling
@param self A pointer pointing to an instance of the class ThresholdBosonSampling_wrapper.
@param args A tuple of the input arguments: sample_num (number of samples to be calculated)
*/
static PyObject *
ThresholdBosonSampling_wrapper_simulate(ThresholdBosonSampling_wrapper *self, PyObject *args)
{

    // initiate variables for input arguments
    int sample_num = 0;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|i", &sample_num))
        return Py_BuildValue("i", -1);



    std::vector<pic::PicState_int64> samples;
    
    try {

        // call the C++ variant of the sampling method    
        samples = self->simulation_strategy->simulate(sample_num);
        
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        std::cout << err << std::endl;
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to sampling trategy object class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }


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
ThresholdBosonSampling_wrapper_getcovariance_matrix(ThresholdBosonSampling_wrapper *self, void *closure)
{
    Py_INCREF(self->covariance_matrix);
    return self->covariance_matrix;
}

/**
@brief Method to set matrix covariance_matrix
*/
static int
ThresholdBosonSampling_wrapper_setcovariance_matrix(ThresholdBosonSampling_wrapper *self, PyObject *covariance_matrix_arg, void *closure)
{
    // set matrin on the Pyhon side
    Py_DECREF(self->covariance_matrix);

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(covariance_matrix_arg) ) {
        self->covariance_matrix = covariance_matrix_arg;
        Py_INCREF(self->covariance_matrix);
    }
    else {
        self->covariance_matrix = PyArray_FROM_OTF(covariance_matrix_arg, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    }

    // create PIC version of the input matrices
    pic::matrix_real covariance_matrix_pq = numpy2matrix_real(self->covariance_matrix);

    // update data on the C++ side
    self->simulation_strategy->Update_covariance_matrix( covariance_matrix_pq );


    return 0;
}



/**
@brief Method to get the upper bound of photons to be sampled.
*/
static PyObject *
ThresholdBosonSampling_wrapper_get_photon_upper_bound(ThresholdBosonSampling_wrapper *self)
{


    int upbound = -1;


    try {
        upbound = self->simulation_strategy->get_photon_upper_bound();
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        std::cout << err << std::endl;
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to sampling trategy object class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    
    // set the upper bound of photon count in the C++ class  
    
    
    
    return Py_BuildValue("i", upbound);
}

/**
@brief Method to set the upper bound of photons to be sampled.
*/
static PyObject *
ThresholdBosonSampling_wrapper_set_photon_upper_bound(ThresholdBosonSampling_wrapper *self, PyObject *args)
{

    int upbound = 0;


    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|i", &upbound))
        return Py_BuildValue("i", -1);


    try {
        // set the upper bound of photon count in the C++ class  
        self->simulation_strategy->set_photon_upper_bound(upbound);
    }
    catch (std::string err) {
        PyErr_SetString(PyExc_Exception, err.c_str());
        std::cout << err << std::endl;
        return NULL;
    }
    catch(...) {
        std::string err( "Invalid pointer to sampling trategy object class");
        PyErr_SetString(PyExc_Exception, err.c_str());
        return NULL;
    }
    

    



    return Py_BuildValue("i", 0);

}



static PyGetSetDef ThresholdBosonSampling_wrapper_getsetters[] = {
    {"covariance_matrix", (getter) ThresholdBosonSampling_wrapper_getcovariance_matrix, (setter) ThresholdBosonSampling_wrapper_setcovariance_matrix,
     "covariance_matrix", NULL},
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class ThresholdBosonSampling_wrapper.
*/
static PyMemberDef ThresholdBosonSampling_wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef ThresholdBosonSampling_wrapper_Methods[] = {
    {"simulate", (PyCFunction) ThresholdBosonSampling_wrapper_simulate, METH_VARARGS,
     "Method to calculate boson sampling output samples"
    },
    {"set_photon_upper_bound", (PyCFunction) ThresholdBosonSampling_wrapper_set_photon_upper_bound, METH_VARARGS,
     "Method to set the upper bound of photons to be sampled."
    },
    {"get_photon_upper_bound", (PyCFunction) ThresholdBosonSampling_wrapper_get_photon_upper_bound, METH_NOARGS,
     "Method to get the upper bound of photons to be sampled."
    },        
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class ThresholdBosonSampling_wrapper.
*/
static PyTypeObject ThresholdBosonSampling_wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "ThresholdBosonSampling_wrapper.ThresholdBosonSampling_wrapper", /*tp_name*/
  sizeof(ThresholdBosonSampling_wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) ThresholdBosonSampling_wrapper_dealloc, /*tp_dealloc*/
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
  ThresholdBosonSampling_wrapper_Methods, /*tp_methods*/
  ThresholdBosonSampling_wrapper_Members, /*tp_members*/
  ThresholdBosonSampling_wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) ThresholdBosonSampling_wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  ThresholdBosonSampling_wrapper_new, /*tp_new*/
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
static PyModuleDef ThresholdBosonSampling_wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ThresholdBosonSampling_wrapper",
    .m_doc = "Python binding for class ThresholdBosonSampling",
    .m_size = -1,
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_ThresholdBosonSampling_wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&ThresholdBosonSampling_wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&ThresholdBosonSampling_wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&ThresholdBosonSampling_wrapper_Type);
    if (PyModule_AddObject(m, "ThresholdBosonSampling_wrapper", (PyObject *) &ThresholdBosonSampling_wrapper_Type) < 0) {
        Py_DECREF(&ThresholdBosonSampling_wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
