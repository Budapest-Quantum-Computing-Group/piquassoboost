#ifndef GlynnPermanentCalculator_wrapper_H
#define GlynnPermanentCalculator_wrapper_H


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "GlynnPermanentCalculator.h"
#include "GlynnPermanentCalculatorInf.h"
#include "GlynnPermanentCalculatorRepeated.h"
#include "GlynnPermanentCalculatorDFE.h"
#include "GlynnPermanentCalculatorRepeatedDFE.h"
#include "numpy_interface.h"
#include <dlfcn.h>


/**
@brief Type definition of the GlynnPermanentCalculator_wrapper Python class of the GlynnPermanentCalculator_wrapper module
*/
typedef struct GlynnPermanentCalculator_wrapper {
    PyObject_HEAD
    /// pointer to numpy matrix to keep it alive
    PyObject *matrix = NULL;
    ///
    void *handle = NULL;
    ///
    void *rephandle = NULL;
    /// The C++ variant of class CGlynnPermanentCalculator
    pic::GlynnPermanentCalculator* calculator;
} GlynnPermanentCalculator_wrapper;


/**
@brief Creates an instance of class GlynnPermanentCalculator and return with a pointer pointing to the class instance (C++ linking is needed)
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
pic::GlynnPermanentCalculator*
create_GlynnPermanentCalculator() {

    return new pic::GlynnPermanentCalculator();
}

/**
@brief Call to deallocate an instance of GlynnPermanentCalculator class
@param ptr A pointer pointing to an instance of GlynnPermanentCalculator class.
*/
void
release_GlynnPermanentCalculator( pic::GlynnPermanentCalculator*  instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}





extern "C"
{


/**
@brief Method called when a python instance of the class GlynnPermanentCalculator_wrapper is destroyed
@param self A pointer pointing to an instance of class GlynnPermanentCalculator_wrapper.
*/
static void
GlynnPermanentCalculator_wrapper_dealloc(GlynnPermanentCalculator_wrapper *self)
{
    // unload DFE
    if (releive_DFE) releive_DFE();
    //if (releiveRep_DFE) releiveRep_DFE();
    if (self->handle) dlclose(self->handle);
    if (self->rephandle) dlclose(self->rephandle);
    // deallocate the instance of class N_Qubit_Decomposition
    release_GlynnPermanentCalculator( self->calculator );

    // release numpy arrays
    Py_DECREF(self->matrix);

   
    Py_TYPE(self)->tp_free((PyObject *) self);
}

#define DFE_PATH_SIM "./dist/release/lib/"
#define DFE_PATH "/home/rakytap/Permanent_Project/PermanentGlynn/PermanentGlynnCPU/dist/release/lib/"
#define DFE_REP_PATH "/home/rakytap/Permanent_Project/PermanentGlynn/PermanentGlynnCPU/dist/release/lib/"
#define DFE_LIB_SIM "libPermanentGlynnSIM.so"
#define DFE_LIB "libPermanentGlynnDFE.so"
#define DFE_REP_LIB_SIM "libPermRepGlynnSIM.so"
#define DFE_REP_LIB "libPermRepGlynnDFE.so"

/**
@brief Method called when a python instance of the class GlynnPermanentCalculator_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class GlynnPermanentCalculator_wrapper.
*/
static PyObject *
GlynnPermanentCalculator_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    GlynnPermanentCalculator_wrapper *self;
    self = (GlynnPermanentCalculator_wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->matrix = NULL;

    // dynamic-loading the correct DFE permanent calculator (Simulator/DFE/single or dual) from shared libararies
    self->handle = dlopen(getenv("SLIC_CONF") ? DFE_PATH_SIM DFE_LIB_SIM : DFE_PATH DFE_LIB, RTLD_NOW); //"MAXELEROSDIR"
    if (self->handle == NULL) {
        char* pwd = getcwd(NULL, 0);
        fprintf(stderr, "%s\n'%s' (in %s mode) failed to load from working directory '%s'\n", dlerror(), getenv("SLIC_CONF") ? DFE_PATH_SIM DFE_LIB_SIM : DFE_PATH DFE_LIB, getenv("SLIC_CONF") ? "simulator" : "DFE", pwd);
        free(pwd);
    } else {
      calcPermanentGlynnDFE = (CALCPERMGLYNNDFE)dlsym(self->handle, "calcPermanentGlynnDFE");
      initialize_DFE = (INITPERMGLYNNDFE)dlsym(self->handle, "initialize_DFE");
      releive_DFE = (FREEPERMGLYNNDFE)dlsym(self->handle, "releive_DFE");
    }
/*
    // dynamic-loading the correct DFE REPEATED permanent calculator (Simulator/DFE/single or dual) from shared libararies
    self->rephandle = dlopen(getenv("SLIC_CONF") ? DFE_PATH_SIM DFE_REP_LIB_SIM : DFE_REP_PATH DFE_REP_LIB, RTLD_NOW); //"MAXELEROSDIR"
    if (self->rephandle == NULL) {
        char* pwd = getcwd(NULL, 0);
        fprintf(stderr, "%s\n'%s' (in %s mode) failed to load from working directory '%s'\n", dlerror(), getenv("SLIC_CONF") ? DFE_PATH_SIM DFE_REP_LIB_SIM : DFE_REP_PATH DFE_REP_LIB, getenv("SLIC_CONF") ? "simulator" : "DFE", pwd);
        free(pwd);
    } else {
      calcPermanentGlynnRepDFE = (CALCPERMGLYNNREPDFE)dlsym(self->rephandle, "calcPermanentGlynnRepDFE");
      initializeRep_DFE = (INITPERMGLYNNREPDFE)dlsym(self->rephandle, "initializeRep_DFE");
      releiveRep_DFE = (FREEPERMGLYNNREPDFE)dlsym(self->rephandle, "releiveRep_DFE");
    }
  */  
    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class GlynnPermanentCalculator_wrapper is initialized
@param self A pointer pointing to an instance of the class GlynnPermanentCalculator_wrapper.
@param args A tuple of the input arguments: qbit_num (integer)
qbit_num: the number of qubits spanning the operations
@param kwds A tuple of keywords
*/
static int
GlynnPermanentCalculator_wrapper_init(GlynnPermanentCalculator_wrapper *self, PyObject *args, PyObject *kwds)
{

    // initialize DFE array
    //initialize_DFE();

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"matrix", NULL};

    // initiate variables for input arguments
    PyObject *matrix_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist,
                                     &matrix_arg))
        return -1;

    // convert python object array to numpy C API array
    if ( matrix_arg == NULL ) return -1;

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        self->matrix = matrix_arg;
        Py_INCREF(self->matrix);
    }
    else {
        self->matrix = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    // create instance of class GlynnPermanentCalculator
    self->calculator = create_GlynnPermanentCalculator();

    return 0;
}



/**
@brief Wrapper function to call the calculate method of C++ class CGlynnPermanentCalculator
@param self A pointer pointing to an instance of the class GlynnPermanentCalculator_Wrapper.
@param args A tuple of the input arguments: ??????????????
@param kwds A tuple of keywords
*/
static PyObject *
GlynnPermanentCalculator_Wrapper_calculate(GlynnPermanentCalculator_wrapper *self)
{

    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(self->matrix);

    // start the calculation of the permanent
    pic::Complex16 ret = self->calculator->calculate(matrix_mtx);

    return Py_BuildValue("D", &ret);
}


/**
@brief Wrapper function to call the calculate method of C++ class CGlynnPermanentCalculator
@param self A pointer pointing to an instance of the class GlynnPermanentCalculator_Wrapper.
@param args A tuple of the input arguments: ??????????????
@param kwds A tuple of keywords
*/
static PyObject *
GlynnPermanentCalculator_Wrapper_calculateInf(GlynnPermanentCalculator_wrapper *self)
{

    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(self->matrix);

    // start the calculation of the permanent
    //pic::GlynnPermanentCalculatorInf* calcInf = new pic::GlynnPermanentCalculatorInf();
    pic::Complex16 ret;// = calcInf->calculate(matrix_mtx);

    return Py_BuildValue("D", &ret);
}


/**
@brief Wrapper function to call the calculate method of C++ class CGlynnPermanentCalculator
@param self A pointer pointing to an instance of the class GlynnPermanentCalculator_Wrapper.
@param args A tuple of the input arguments: ??????????????
@param kwds A tuple of keywords
*/
static PyObject *
GlynnPermanentCalculator_Wrapper_calculate_repeated(GlynnPermanentCalculator_wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"matrix", (char*)"input_state", (char*)"output_state", NULL};

    // initiate variables for input arguments
    PyObject *matrix_arg = NULL;
    PyObject *input_state_arg = NULL;
    PyObject *output_state_arg = NULL;
    /// pointer to numpy matrix to keep it alive
    PyObject *matrix = NULL;
    /// pointer to numpy matrix of input states to keep it alive
    PyObject *input_state = NULL;
    /// pointer to numpy matrix of output states to keep it alive
    PyObject *output_state = NULL;


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", kwlist,
                                     &matrix_arg, &input_state_arg, &output_state_arg))
        return Py_BuildValue("");

    // convert python object array to numpy C API array
    if ( matrix_arg == NULL ) return Py_BuildValue("");
    if ( input_state_arg == NULL ) return Py_BuildValue("");
    if ( output_state_arg == NULL ) return Py_BuildValue("");

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        matrix = matrix_arg;
        Py_INCREF(matrix);
    }
    else {
        matrix = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(input_state_arg) ) {
        input_state = input_state_arg;
        Py_INCREF(input_state);
    }
    else {
        input_state = PyArray_FROM_OTF(input_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(output_state_arg) ) {
        output_state = output_state_arg;
        Py_INCREF(output_state);
    }
    else {
        output_state = PyArray_FROM_OTF(output_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }

  

    // create instance of class GlynnPermanentCalculatorRepeated
    pic::GlynnPermanentCalculatorRepeated* calcRepeated = new pic::GlynnPermanentCalculatorRepeated();

    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(matrix);
    pic::PicState_int64 input_state_mtx = numpy2PicState_int64(input_state);
    pic::PicState_int64 output_state_mtx = numpy2PicState_int64(output_state);

    // start the calculation of the permanent
    pic::Complex16 ret = calcRepeated->calculate(matrix_mtx, input_state_mtx, output_state_mtx);
    delete calcRepeated;

    return Py_BuildValue("D", &ret);
}


/**
@brief Wrapper function to call the calculate the Permanent on a DFE
@param self A pointer pointing to an instance of the class GlynnPermanentCalculator_Wrapper.
@param args A tuple of the input arguments: ??????????????
@param kwds A tuple of keywords
*/
static PyObject *
GlynnPermanentCalculator_Wrapper_calculateDFE(GlynnPermanentCalculator_wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"matrix", (char*)"dual", NULL};

    // initiate variables for input arguments
    int useDual = 0;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p", kwlist,
                                     &useDual))
        return Py_BuildValue("");

    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(self->matrix);

    if (initialize_DFE) initialize_DFE(useDual);

    pic::Complex16 perm;
    
    if (calcPermanentGlynnDFE) GlynnPermanentCalculator_DFE( matrix_mtx, perm, useDual);
    else perm = self->calculator->calculate(matrix_mtx);

    return Py_BuildValue("D", &perm);
}

/**
@brief Wrapper function to call the calculate method of C++ class CGlynnPermanentCalculator
@param self A pointer pointing to an instance of the class GlynnPermanentCalculator_Wrapper.
@param args A tuple of the input arguments: ??????????????
@param kwds A tuple of keywords
*/
static PyObject *
GlynnPermanentCalculator_Wrapper_calculate_repeatedDFE(GlynnPermanentCalculator_wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"matrix", (char*)"input_state", (char*)"output_state", (char*)"dual", NULL};

    // initiate variables for input arguments
    PyObject *matrix_arg = NULL;
    PyObject *input_state_arg = NULL;
    PyObject *output_state_arg = NULL;
    int useDual = 0;
    /// pointer to numpy matrix to keep it alive
    PyObject *matrix = NULL;
    /// pointer to numpy matrix of input states to keep it alive
    PyObject *input_state = NULL;
    /// pointer to numpy matrix of output states to keep it alive
    PyObject *output_state = NULL;


    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOp", kwlist,
                                     &matrix_arg, &input_state_arg, &output_state_arg, &useDual))
        return Py_BuildValue("");

    // convert python object array to numpy C API array
    if ( matrix_arg == NULL ) return Py_BuildValue("");
    if ( input_state_arg == NULL ) return Py_BuildValue("");
    if ( output_state_arg == NULL ) return Py_BuildValue("");

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        matrix = matrix_arg;
        Py_INCREF(matrix);
    }
    else {
        matrix = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(input_state_arg) ) {
        input_state = input_state_arg;
        Py_INCREF(input_state);
    }
    else {
        input_state = PyArray_FROM_OTF(input_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(output_state_arg) ) {
        output_state = output_state_arg;
        Py_INCREF(output_state);
    }
    else {
        output_state = PyArray_FROM_OTF(output_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }

  


    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(matrix);
    pic::PicState_int64 input_state_mtx = numpy2PicState_int64(input_state);
    pic::PicState_int64 output_state_mtx = numpy2PicState_int64(output_state);

    //if (initializeRep_DFE) initializeRep_DFE(useDual);
    
    // start the calculation of the permanent
    pic::Complex16 ret;
/*
    if (calcPermanentGlynnRepDFE) GlynnPermanentCalculatorRepeated_DFE( matrix_mtx, input_state_mtx, output_state_mtx, ret, useDual);
    else {
      // create instance of class GlynnPermanentCalculatorRepeated
      pic::GlynnPermanentCalculatorRepeated* calcRepeated = new pic::GlynnPermanentCalculatorRepeated();
      ret = calcRepeated->calculate(matrix_mtx, input_state_mtx, output_state_mtx);
      delete calcRepeated;
    }
*/
    return Py_BuildValue("D", &ret);
}

static PyGetSetDef GlynnPermanentCalculator_wrapper_getsetters[] = {
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class GlynnPermanentCalculator_wrapper.
*/
static PyMemberDef GlynnPermanentCalculator_wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef GlynnPermanentCalculator_wrapper_Methods[] = {
    {"calculate", (PyCFunction) GlynnPermanentCalculator_Wrapper_calculate, METH_NOARGS,
     "Method to calculate the permanent."
    },
    {"calculateInf", (PyCFunction) GlynnPermanentCalculator_Wrapper_calculateInf, METH_NOARGS,
     "Method to calculate the permanent with GMP MPFR for infinite precision."
    },
    {"calculateDFE", (PyCFunction) GlynnPermanentCalculator_Wrapper_calculateDFE, METH_VARARGS | METH_KEYWORDS,
     "Method to calculate the permanent on single or dual DFE."
    },
    {"calculate_repeated", (PyCFunction) GlynnPermanentCalculator_Wrapper_calculate_repeated, METH_VARARGS | METH_KEYWORDS,
     "Method to calculate the permanent with repeated rows and columns."
    },
    {"calculate_repeatedDFE", (PyCFunction) GlynnPermanentCalculator_Wrapper_calculate_repeatedDFE, METH_VARARGS | METH_KEYWORDS,
     "Method to calculate the permanent with repeated rows and columns on single or dual DFE."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class GlynnPermanentCalculator_wrapper.
*/
static PyTypeObject GlynnPermanentCalculator_wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "GlynnPermanentCalculator_wrapper.GlynnPermanentCalculator_wrapper", /*tp_name*/
  sizeof(GlynnPermanentCalculator_wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) GlynnPermanentCalculator_wrapper_dealloc, /*tp_dealloc*/
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
  GlynnPermanentCalculator_wrapper_Methods, /*tp_methods*/
  GlynnPermanentCalculator_wrapper_Members, /*tp_members*/
  GlynnPermanentCalculator_wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) GlynnPermanentCalculator_wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  GlynnPermanentCalculator_wrapper_new, /*tp_new*/
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



#endif //GlynnPermanentCalculator_wrapper
