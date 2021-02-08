#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "CGeneralizedCliffordsSimulationStrategy.h"
#include "tbb/scalable_allocator.h"

/**
@brief """Object to watch over raw pointers when coated by a numpy array."""
*/
class Sentinel {

public:    

    // the raw pointer to the data
    void* ptr;

/**
@brief Default constructor of the class
*/
Sentinel() {

    ptr = NULL;

}


/**
@brief Constructor of the class. This method is called to make the class instance to watch over the raw pinter.
*/
Sentinel( void *ptr_in) {

    ptr = ptr_in;

}

/**
This method is called when the numpy array loses its last reference. 
        The raw pointer should be released by method corresponding to the allocation.
*/
~Sentinel() {

        scalable_aligned_free(ptr);

}

};

/*

# external declaration of the function PyArray_SetBaseObject from the Numpy API
cdef extern from "numpy/arrayobject.h":
    int PyArray_SetBaseObject(np.ndarray arr, PyObject *obj) except -1
*/


void capsule_cleanup(PyObject* capsule) {

    void *memory = PyCapsule_GetPointer(capsule, NULL);
    // I'm going to assume your memory needs to be freed with free().
    // If it needs different cleanup, perform whatever that cleanup is
    // instead of calling free().
    scalable_aligned_free(memory);


}


PyObject* array_from_ptr(void * ptr, int dim, npy_intp* shape, int np_type) {

        // create numpy array
        PyObject* arr = PyArray_SimpleNewFromData(dim, shape, np_type, ptr);

        // set memory keeper for the numpy array
        PyObject *capsule = PyCapsule_New(ptr, NULL, capsule_cleanup);
        PyArray_SetBaseObject((PyArrayObject *) arr, capsule);
        
        return arr;

}



/**
r"""
        Call to make a numpy array from an external PicState_int64 class.

        Args:
        cstate (PicState_int64&): a PicState_int64 instance
*/
PyObject* PicState_int64_to_numpy( pic::PicState_int64 &cstate ) {
        

        npy_intp shape[1];
        shape[0] = (npy_intp) cstate.cols;

        int64_t* data = cstate.get_data();
        return array_from_ptr( (void*) data, 1, shape, NPY_INT64);


}


/*

    cdef np.ndarray matrix_to_numpy(self, matrix &cmtx ):
        r"""
        Call to make a numpy array from an external matrix class.

        Args:
        cstate (PicState_int64&): a PicState_int64 instance

        """

        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> cmtx.rows
        shape[1] = <np.npy_intp> cmtx.cols

        cdef double complex* data = cmtx.get_data()
        return self.array_from_ptr( <void*> data, 2, shape, np.NPY_COMPLEX128 ) 

*/
        
/**
        ptr (void*): a void pointer
        rows (int) number of rows in the array
        cols (int) number of columns in the array
        np_type (int) The data type stored in the numpy array (see possible values at https://numpy.org/doc/1.17/reference/c-api.dtype.html)
*/



/**
@brief Type definition of the GeneralizedCliffordsSimulationStrategy_wrapper Python class of the GeneralizedCliffordsSimulationStrategy_wrapper module
*/
typedef struct GeneralizedCliffordsSimulationStrategy_wrapper {
    PyObject_HEAD
    /// pointer to numpy matrix to keep it alive
    PyObject *interferometer_matrix = NULL;
    /// The C++ variant of class CGeneralizedCliffordsSimulationStrategy
    pic::CGeneralizedCliffordsSimulationStrategy* simulation_strategy = NULL;

} GeneralizedCliffordsSimulationStrategy_wrapper;


/**
@brief Creates an instance of class ChinHuhPermanentCalculator and return with a pointer pointing to the class instance (C++ linking is needed)
@param interferometer_matrix
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
pic::CGeneralizedCliffordsSimulationStrategy* 
cerate_ChinHuhPermanentCalculator( pic::matrix &interferometer_matrix_mtx ) {

    return new pic::CGeneralizedCliffordsSimulationStrategy(interferometer_matrix_mtx);

}

/**
@brief Call to deallocate an instance of ChinHuhPermanentCalculator class
@param ptr A pointer pointing to an instance of ChinHuhPermanentCalculator class.
*/
void
release_ChinHuhPermanentCalculator( pic::CGeneralizedCliffordsSimulationStrategy*  instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}


/**
@brief Call to create a PIC matrix representation of a numpy array
*/
pic::matrix 
numpy2matrix(PyObject *arr) {

    // test C-style contiguous memory allocation of the arrays
    if ( !PyArray_IS_C_CONTIGUOUS(arr) ) {
        std::cout << "array is not memory contiguous" << std::endl;
    }

    // get the pointer to the data stored in the input matrices
    pic::Complex16* data = (pic::Complex16*)PyArray_DATA(arr);

    // get the dimensions of the array self->C
    int dim_num = PyArray_NDIM( arr );
    npy_intp* dims = PyArray_DIMS(arr);

    // create PIC version of the input matrices   
    if (dim_num == 2) {
        pic::matrix mtx = pic::matrix(data, dims[0], dims[1]);  
        return mtx;
    }
    else if (dim_num == 1) {
        pic::matrix mtx = pic::matrix(data, dims[0], 1);  
        return mtx;
    }
    else {
        std::cout << "numpy2matrix: Wrong matrix dimension was given" << std::endl;
        return pic::matrix(0,0);
    }



} 



/**
@brief Call to create a PicState_int64 representation of a numpy array
*/
pic::PicState_int64 
numpy2PicState_int64(PyObject *arr) {


    // test C-style contiguous memory allocation of the arrays
    if ( !PyArray_IS_C_CONTIGUOUS(arr) ) {
        std::cout << "array is not memory contiguous" << std::endl;
    }

    // get the pointer to the data stored in the input matrices
    int64_t* data = (int64_t*)PyArray_DATA(arr);

    // get the dimensions of the array self->C
    int dim_num = PyArray_NDIM( arr );
    npy_intp* dims = PyArray_DIMS(arr);

    // create PIC version of the input matrices   
    if (dim_num == 1) {
        pic::PicState_int64 state = pic::PicState_int64(data, dims[0]);  
        return state;
    }
    else {
        std::cout << "numpy2PicState_int64: Wrong matrix dimension was given" << std::endl;
        return pic::PicState_int64(0);
    }



} 





extern "C"
{




/**
@brief Method called when a python instance of the class GeneralizedCliffordsSimulationStrategy_wrapper is destroyed
@param self A pointer pointing to an instance of class GeneralizedCliffordsSimulationStrategy_wrapper.
*/
static void
GeneralizedCliffordsSimulationStrategy_wrapper_dealloc(GeneralizedCliffordsSimulationStrategy_wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_ChinHuhPermanentCalculator( self->simulation_strategy );

    // release numpy arrays
    Py_DECREF(self->interferometer_matrix);   

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class GeneralizedCliffordsSimulationStrategy_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class GeneralizedCliffordsSimulationStrategy_wrapper.
*/
static PyObject *
GeneralizedCliffordsSimulationStrategy_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    GeneralizedCliffordsSimulationStrategy_wrapper *self;
    self = (GeneralizedCliffordsSimulationStrategy_wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->interferometer_matrix = NULL;
    self->simulation_strategy = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class GeneralizedCliffordsSimulationStrategy_wrapper is initialized
@param self A pointer pointing to an instance of the class GeneralizedCliffordsSimulationStrategy_wrapper.
@param args A tuple of the input arguments: qbit_num (integer)
qbit_num: the number of qubits spanning the operations
@param kwds A tuple of keywords
*/
static int
GeneralizedCliffordsSimulationStrategy_wrapper_init(GeneralizedCliffordsSimulationStrategy_wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"interferometer_matrix", NULL};

    // initiate variables for input arguments
    PyObject *interferometer_matrix_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist,
                                     &interferometer_matrix_arg))
        return -1;

    // convert python object array to numpy C API array
    if ( interferometer_matrix_arg == NULL ) return -1;

    self->interferometer_matrix = PyArray_FROM_OTF(interferometer_matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);

    

    // create PIC version of the input matrices
    pic::matrix interferometer_matrix_mtx = numpy2matrix(self->interferometer_matrix); 

    // create instance of class ChinHuhPermanentCalculator
    self->simulation_strategy = cerate_ChinHuhPermanentCalculator( interferometer_matrix_mtx );
   
    return 0;
}



/**
@brief Wrapper function to call the simulate method of C++ class CGaussianState
@param self A pointer pointing to an instance of the class GaussianState_Wrapper.
@param args A tuple of the input arguments: ??????????????
*/
static PyObject *
GeneralizedCliffordsSimulationStrategy_wrapper_simulate(GeneralizedCliffordsSimulationStrategy_wrapper *self, PyObject *args)
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

    PyObject* input_state = PyArray_FROM_OTF(input_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);

    

    // create PIC version of the input matrices
    pic::PicState_int64 input_state_mtx = numpy2PicState_int64(input_state);  
   

    
    // call the C++ variant of the sampling method
    std::vector<pic::PicState_int64> samples = self->simulation_strategy->simulate(input_state_mtx, sample_num);


    // preallocate Python list to hold the calculated samples
    PyObject* PySamples = PyTuple_New( (Py_ssize_t) sample_num );


    for ( int idx = 0; idx < samples.size(); idx++ ) {
        PyObject *PySample = PicState_int64_to_numpy( samples[idx] );
        PyTuple_SetItem(PySamples, idx, PySample);

    }

    Py_DECREF(input_state);   

    return PySamples;  

}





static PyGetSetDef GeneralizedCliffordsSimulationStrategy_wrapper_getsetters[] = {
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class GeneralizedCliffordsSimulationStrategy_wrapper.
*/
static PyMemberDef GeneralizedCliffordsSimulationStrategy_wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef GeneralizedCliffordsSimulationStrategy_wrapper_Methods[] = {
    {"simulate", (PyCFunction) GeneralizedCliffordsSimulationStrategy_wrapper_simulate, METH_VARARGS,
     "Method to calculate boson sampling output samples"
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class GeneralizedCliffordsSimulationStrategy_wrapper.
*/
static PyTypeObject GeneralizedCliffordsSimulationStrategy_wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "GeneralizedCliffordsSimulationStrategy_wrapper.GeneralizedCliffordsSimulationStrategy_wrapper", /*tp_name*/
  sizeof(GeneralizedCliffordsSimulationStrategy_wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) GeneralizedCliffordsSimulationStrategy_wrapper_dealloc, /*tp_dealloc*/
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
  GeneralizedCliffordsSimulationStrategy_wrapper_Methods, /*tp_methods*/
  GeneralizedCliffordsSimulationStrategy_wrapper_Members, /*tp_members*/
  GeneralizedCliffordsSimulationStrategy_wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) GeneralizedCliffordsSimulationStrategy_wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  GeneralizedCliffordsSimulationStrategy_wrapper_new, /*tp_new*/
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
static PyModuleDef GeneralizedCliffordsSimulationStrategy_wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "GeneralizedCliffordsSimulationStrategy_wrapper",
    .m_doc = "Python binding for class ChinHuhPermanentCalculator",
    .m_size = -1,
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_GeneralizedCliffordsSimulationStrategy_wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&GeneralizedCliffordsSimulationStrategy_wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&GeneralizedCliffordsSimulationStrategy_wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&GeneralizedCliffordsSimulationStrategy_wrapper_Type);
    if (PyModule_AddObject(m, "GeneralizedCliffordsSimulationStrategy_wrapper", (PyObject *) &GeneralizedCliffordsSimulationStrategy_wrapper_Type) < 0) {
        Py_DECREF(&GeneralizedCliffordsSimulationStrategy_wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
