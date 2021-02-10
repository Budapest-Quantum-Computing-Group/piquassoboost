#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "CGaussianState.h"
#include "numpy_interface.h"


/**
@brief Type definition of the GaussianState_Wrapper Python class of the GaussianState_Wrapper module
*/
typedef struct GaussianState_Wrapper {
    PyObject_HEAD
    /// pointer to numpy matrix C to keep it alive
    PyObject *C=NULL;
    /// pointer to numpy matrix G to keep it alive
    PyObject *G=NULL;
    /// pointer to numpy matrix mean to keep it alive
    PyObject *mean=NULL;
    /// The C++ variant of class GaussianState
    pic::CGaussianState* state=NULL;
} GaussianState_Wrapper;


/**
@brief Creates an instance of class CGaussianState and return with a pointer pointing to the class instance (C++ linking is needed)
@param C
@param G
@param mean
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
pic::CGaussianState* 
create_GaussianState( pic::matrix &C, pic::matrix &G, pic::matrix &mean ) {

    return new pic::CGaussianState(C, G, mean);
}

/**
@brief Call to deallocate an instance of CGaussianState class
@param ptr A pointer pointing to an instance of CGaussianState class.
*/
void
release_CGaussianState( pic::CGaussianState*  instance ) {

    if (instance != NULL ) {
        delete instance;
        instance = NULL;
    }
    return;
}




extern "C"
{




/**
@brief Method called when a python instance of the class GaussianState_Wrapper is destroyed
@param self A pointer pointing to an instance of class GaussianState_Wrapper.
*/
static void
GaussianState_Wrapper_dealloc(GaussianState_Wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_CGaussianState( self->state );

    // release numpy arrays
    Py_DECREF(self->C);    
    Py_DECREF(self->G);    
    Py_DECREF(self->mean);    

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class GaussianState_Wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class GaussianState_Wrapper.
*/
static PyObject *
GaussianState_Wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    GaussianState_Wrapper *self;
    self = (GaussianState_Wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->state = NULL;
    self->C = NULL;
    self->G = NULL;
    self->mean = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class GaussianState_Wrapper is initialized
@param self A pointer pointing to an instance of the class GaussianState_Wrapper.
@param args A tuple of the input arguments: qbit_num (integer)
qbit_num: the number of qubits spanning the operations
@param kwds A tuple of keywords
*/
static int
GaussianState_Wrapper_init(GaussianState_Wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"C", (char*)"G", (char*)"mean", NULL};

    // initiate variables for input arguments
    PyObject *C_arg = NULL;
    PyObject *G_arg = NULL;
    PyObject *mean_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", kwlist,
                                     &C_arg, &G_arg, &mean_arg))
        return -1;

    // convert python object array to numpy C API array
    if ( C_arg == NULL ) return -1;
    if ( G_arg == NULL ) return -1;
    if ( mean_arg == NULL ) return -1;

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(C_arg) ) {
        self->C = C_arg;
        Py_INCREF(self->C); 
    }
    else {
        self->C = PyArray_FROM_OTF(C_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(G_arg) ) {
        self->G = G_arg;
        Py_INCREF(self->G); 
    }
    else {
        self->G = PyArray_FROM_OTF(G_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    if ( PyArray_IS_C_CONTIGUOUS(mean_arg) ) {
        self->mean = mean_arg;
        Py_INCREF(self->mean); 
    }
    else {
        self->mean = PyArray_FROM_OTF(mean_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }
    

    // create PIC version of the input matrices
    pic::matrix C_mtx = numpy2matrix(self->C); 
    pic::matrix G_mtx = numpy2matrix(self->G);  
    pic::matrix mean_mtx = numpy2matrix(self->mean);  

    // create instance of class CGaussianState
    self->state = create_GaussianState( C_mtx, G_mtx, mean_mtx );

    
    return 0;
}


/**
@brief Wrapper function to call the apply_to_C_and_G method of C++ class CGaussianState
@param self A pointer pointing to an instance of the class GaussianState_Wrapper.
@param args A tuple of the input arguments: ??????????????
*/
static PyObject *
GaussianState_Wrapper_apply_to_C_and_G(GaussianState_Wrapper *self, PyObject *args)
{
   

    // initiate variables for input arguments
    PyObject* T_arg = NULL; 
    PyObject* modes = NULL; 

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OO",
                                     &T_arg, &modes) )
        return Py_BuildValue("i", -1);

    // establish memory contiguous arrays for C calculations
    PyObject* T = NULL;
    if ( PyArray_IS_C_CONTIGUOUS(T_arg) ) {
        T = T_arg;
        Py_INCREF(T); 
    }
    else {
        T = PyArray_FROM_OTF(T_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    // create PIC version of the input matrix T
    pic::matrix T_mtx = numpy2matrix(T);     


    // convert python list/tuple of modes to std vector
    bool is_tuple = PyTuple_Check(modes);
    bool is_list = PyList_Check(modes);

    // Check whether input is dictionary
    if (!is_list && !is_tuple) {
        printf("Input modes must be tuple or list!\n");
        return Py_BuildValue("i", -1);
    }

    // get the number of qbubits
    Py_ssize_t element_num;

    if (is_tuple) {
        element_num = PyTuple_GET_SIZE(modes);
    }
    else {
        element_num = PyList_GET_SIZE(modes);
    }


    // create C++ variant of the tuple/list
    std::vector<size_t> modes_C;
    modes_C.reserve( (int) element_num);
    for ( Py_ssize_t idx=0; idx<element_num; idx++ ) {
        if (is_tuple) {        
            modes_C.push_back( (int) PyLong_AsLong( PyTuple_GetItem(modes, idx ) ) );
        }
        else {
            modes_C.push_back( (int) PyLong_AsLong( PyList_GetItem(modes, idx ) ) );
        }
    }

    // call the C++ variant transformation on the matrices C,G
    self->state->apply_to_C_and_G( T_mtx, modes_C );

    Py_DECREF(T);

    return Py_BuildValue("i", 0);  

}


/**
@brief Method to call get matrix C
*/
static PyObject *
GaussianState_Wrapper_getC(GaussianState_Wrapper *self, void *closure)
{
    Py_INCREF(self->C);
    return self->C;
}

/**
@brief Method to call set matrix C
*/
static int
GaussianState_Wrapper_setC(GaussianState_Wrapper *self, PyObject *C_arg, void *closure)
{
    // set the mytrix on Python side
    Py_DECREF(self->C); 

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(C_arg) ) {
        self->C = C_arg;
        Py_INCREF(self->C); 
    }
    else {
        self->C = PyArray_FROM_OTF(C_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix C_mtx = numpy2matrix(self->C);     

    // update data on the C++ side
    self->state->Update_C( C_mtx );


    return 0;
}





/**
@brief Method to call get matrix G
*/
static PyObject *
GaussianState_Wrapper_getG(GaussianState_Wrapper *self, void *closure)
{
    Py_INCREF(self->G);
    return self->G;
}

/**
@brief Method to call set matrix G
*/
static int
GaussianState_Wrapper_setG(GaussianState_Wrapper *self, PyObject *G_arg, void *closure)
{
    // set the array on the Python side
    Py_DECREF(self->G); 

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(G_arg) ) {
        self->G = G_arg;
        Py_INCREF(self->G); 
    }
    else {
        self->G = PyArray_FROM_OTF(G_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    // create PIC version of the input matrices
    pic::matrix G_mtx = numpy2matrix(self->G); ;    

    // update data on the C++ side
    self->state->Update_G( G_mtx );


    return 0;
}




/**
@brief Method to call get matrix mean
*/
static PyObject *
GaussianState_Wrapper_getmean(GaussianState_Wrapper *self, void *closure)
{
    Py_INCREF(self->mean);
    return self->mean;
}

/**
@brief Method to call set matrix mean
*/
static int
GaussianState_Wrapper_setmean(GaussianState_Wrapper *self, PyObject *mean_arg, void *closure)
{
    // set the array on the Python side
    Py_DECREF(self->mean); 

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(mean_arg) ) {
        self->mean = mean_arg;
        Py_INCREF(self->mean); 
    }
    else {
        self->mean = PyArray_FROM_OTF(mean_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix mean_mtx = numpy2matrix(self->mean);    

    // update data on the C++ side
    self->state->Update_mean( mean_mtx );


    return 0;
}



static PyGetSetDef GaussianState_Wrapper_getsetters[] = {
    {"C", (getter) GaussianState_Wrapper_getC, (setter) GaussianState_Wrapper_setC,
     "C matrix", NULL},
    {"G", (getter) GaussianState_Wrapper_getG, (setter) GaussianState_Wrapper_setG,
     "G matrix", NULL},
    {"mean", (getter) GaussianState_Wrapper_getmean, (setter) GaussianState_Wrapper_setmean,
     "mean matrix", NULL},
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class GaussianState_Wrapper.
*/
static PyMemberDef GaussianState_Wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef GaussianState_Wrapper_Methods[] = {
    {"apply_to_C_and_G", (PyCFunction) GaussianState_Wrapper_apply_to_C_and_G, METH_VARARGS,
     "Method to transform amtrices C and G."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class GaussianState_Wrapper.
*/
static PyTypeObject GaussianState_Wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "GaussianState_Wrapper.GaussianState_Wrapper", /*tp_name*/
  sizeof(GaussianState_Wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) GaussianState_Wrapper_dealloc, /*tp_dealloc*/
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
  GaussianState_Wrapper_Methods, /*tp_methods*/
  GaussianState_Wrapper_Members, /*tp_members*/
  GaussianState_Wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) GaussianState_Wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  GaussianState_Wrapper_new, /*tp_new*/
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
static PyModuleDef GaussianState_Wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "GaussianState_Wrapper",
    .m_doc = "Python binding for class CGaussianState",
    .m_size = -1,
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_state_wrapper(void)
{

    // initialize Numpy API
    import_array();

    PyObject *m;
    if (PyType_Ready(&GaussianState_Wrapper_Type) < 0)
        return NULL;

    m = PyModule_Create(&GaussianState_Wrapper_Module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&GaussianState_Wrapper_Type);
    if (PyModule_AddObject(m, "GaussianState_Wrapper", (PyObject *) &GaussianState_Wrapper_Type) < 0) {
        Py_DECREF(&GaussianState_Wrapper_Type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}



}
