#ifndef GlynnPermanentCalculator_wrapper_H
#define GlynnPermanentCalculator_wrapper_H


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "GlynnPermanentCalculator.h"
#include "GlynnPermanentCalculatorRepeated.h"

#ifdef __MPFR__
#include "GlynnPermanentCalculatorInf.h"
#endif

#ifdef __DFE__
#include "GlynnPermanentCalculatorDFE.h"
#endif

#include "numpy_interface.h"
#include <dlfcn.h>

/// The C++ variants of class CGlynnPermanentCalculator
union CPU_glynn {
    /// long double precision calculator
    pic::GlynnPermanentCalculator* cpu_long_double;
    /// long double precision calculator using repeated rows implementation
    pic::GlynnPermanentCalculatorRepeated* cpu_long_double_repeated;
#ifdef __MPFR__
    /// infinite precision calculator
    pic::GlynnPermanentCalculatorInf* cpu_inf;
#endif
};

/**
@brief Type definition of the GlynnPermanentCalculator_wrapper Python class of the GlynnPermanentCalculator_wrapper module
*/
typedef struct GlynnPermanentCalculator_wrapper {
    PyObject_HEAD
    /// pointer to numpy matrix to keep it alive
    PyObject *matrix = NULL;
    /// pointer to numpy matrix of input states to keep it alive (used in repeated rows claculator variant)
    PyObject *input_state = NULL;
    /// pointer to numpy matrix of output states to keep it alive (used in repeated rows claculator variant)
    PyObject *output_state = NULL;
    ///
    void *handle = NULL;
    ///
    void *rephandle = NULL;
    /// set 0 to use CPU implementation, set 1 to use single DFE implementation, set 2 to use dual DFE implementation
    int DFE = 0;
    /// set 1 (default) to use long double precision, set 2 to use infinite computational precision using the GNU MPFR library. Has no effect if DFE>0 is set.
    int precision = 1;
    /// CPU permanent calculator
    CPU_glynn calculator;

} GlynnPermanentCalculator_wrapper;


/**
@brief Creates an instance of class GlynnPermanentCalculator and return with a pointer pointing to the class instance (C++ linking is needed)
@return Return with a void pointer pointing to an instance of GlynnPermanentCalculator.
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




/**
@brief Creates an instance of class GlynnPermanentCalculatorRepeated and return with a pointer pointing to the class instance (C++ linking is needed)
@return Return with a void pointer pointing to an instance of GlynnPermanentCalculatorRepeated.
*/
pic::GlynnPermanentCalculatorRepeated*
create_GlynnPermanentCalculatorRepeated() {

    return new pic::GlynnPermanentCalculatorRepeated();
}

/**
@brief Call to deallocate an instance of GlynnPermanentCalculatorRepeated class
@param ptr A pointer pointing to an instance of GlynnPermanentCalculatorRepeated class.
*/
void
release_GlynnPermanentCalculatorRepeated( pic::GlynnPermanentCalculatorRepeated*  instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}

#ifdef __MPFR__

/**
@brief Creates an instance of class GlynnPermanentCalculatorInf and return with a pointer pointing to the class instance (C++ linking is needed)
@return Return with a void pointer pointing to an instance of GlynnPermanentCalculatorInf.
*/
pic::GlynnPermanentCalculatorInf*
create_GlynnPermanentCalculatorInf() {

    return new pic::GlynnPermanentCalculatorInf();
}

/**
@brief Call to deallocate an instance of GlynnPermanentCalculatorInf class
@param ptr A pointer pointing to an instance of GlynnPermanentCalculatorInf class.
*/
void
release_GlynnPermanentCalculatorInf ( pic::GlynnPermanentCalculatorInf*  instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}


#endif


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
#ifdef __DFE__
    if (releive_DFE) releive_DFE();
    if (self->handle) dlclose(self->handle);
    if (self->rephandle) dlclose(self->rephandle);
#endif

    if ( self->precision == 1 ) {
        if ( self->input_state || self->output_state) {
            // deallocate the instance of class GlynnPermanentCalculatorRepeated
            release_GlynnPermanentCalculatorRepeated( self->calculator.cpu_long_double_repeated );
        }
        else {
            // deallocate the instance of class GlynnPermanentCalculator
            release_GlynnPermanentCalculator( self->calculator.cpu_long_double );
        } 


    }
#ifdef __MPFR__
    else if ( self->precision == 2 ) {
        // deallocate the instance of class GlynnPermanentCalculatorInf
        release_GlynnPermanentCalculatorInf( self->calculator.cpu_inf );
    }
#endif
    else {
        printf("CPU permanent calculator uninitialized\n");
    }


    
    

    // release numpy arrays
    Py_DECREF(self->matrix);

   
    Py_TYPE(self)->tp_free((PyObject *) self);
}

#ifdef __DFE__
#define DFE_PATH_SIM "./dist/release/lib/"
#define DFE_PATH "/home/rakytap/Permanent_Project/PermanentGlynn/PermanentGlynnCPU/dist/release/lib/"
#define DFE_REP_PATH "/home/rakytap/Permanent_Project/PermanentGlynn/PermanentGlynnCPU/dist/release/lib/"
#define DFE_LIB_SIM "libPermanentGlynnSIM.so"
#define DFE_LIB "libPermanentGlynnDFE.so"
#endif

/**
@brief Method called when a python instance of the class GlynnPermanentCalculator_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class GlynnPermanentCalculator_wrapper.
*/
static PyObject *
GlynnPermanentCalculator_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {

    GlynnPermanentCalculator_wrapper *self;
    self = (GlynnPermanentCalculator_wrapper *) type->tp_alloc(type, 0);

    self->matrix = NULL;

#ifdef __DFE__
    // dynamic-loading the correct DFE permanent calculator (Simulator/DFE/single or dual) from shared libararies
    self->handle = dlopen(getenv("SLIC_CONF") ? DFE_PATH_SIM DFE_LIB_SIM /*use simulator*/ : DFE_PATH DFE_LIB /* use DFE */, RTLD_NOW); //"MAXELEROSDIR"

    if (self->handle == NULL) {
        char* pwd = getcwd(NULL, 0);
        fprintf(stderr, "%s\n'%s' (in %s mode) failed to load from working directory '%s'\n", dlerror(), getenv("SLIC_CONF") ? DFE_PATH_SIM DFE_LIB_SIM : DFE_PATH DFE_LIB, getenv("SLIC_CONF") ? "simulator" : "DFE", pwd);
        free(pwd);
        PyErr_SetString(PyExc_Exception, "Failed to load DFE libraries.");
        return NULL;
    } else {
        // in case the DFE libraries were loaded successfully the function pointers are set to initialize/releive DFE engine and run DFE calculations
        calcPermanentGlynnDFE = (CALCPERMGLYNNDFE)dlsym(self->handle, "calcPermanentGlynnDFE");
        initialize_DFE = (INITPERMGLYNNDFE)dlsym(self->handle, "initialize_DFE");
        releive_DFE = (FREEPERMGLYNNDFE)dlsym(self->handle, "releive_DFE");
    }
#endif
 
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

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"matrix", (char*)"DFE", (char*)"precision", (char*)"input_state", (char*)"output_state", NULL};

    // initiate variables for input arguments
    PyObject *matrix_arg = NULL;
    PyObject *input_state_arg = NULL;
    PyObject *output_state_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OiiOO", kwlist,
                                     &matrix_arg, &(self->DFE), &(self->precision), &input_state_arg, &output_state_arg))
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

    // store input state if given
    if ( input_state_arg && PyArray_IS_C_CONTIGUOUS(input_state_arg) ) {
        self->input_state = input_state_arg;
        Py_INCREF(self->input_state);
    }
    else if (input_state_arg) {
        self->input_state = PyArray_FROM_OTF(input_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }
    else {
        self->input_state = NULL;
    }

    // store output state if given
    if ( output_state_arg && PyArray_IS_C_CONTIGUOUS(output_state_arg) ) {
        self->output_state = output_state_arg;
        Py_INCREF(self->output_state);
    }
    else if (output_state_arg) {
        self->output_state = PyArray_FROM_OTF(output_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }
    else {
        self->output_state = NULL;
    }

    // validating input arguments
#ifdef __MPFR__
    if ( self->precision < 1 || self->precision > 2 ) {
        PyErr_SetString(PyExc_Exception, "Wrong value set for precision.");
        return -1;
    }
#else
    if ( self->precision < 1 || self->precision > 1 ) {
        PyErr_SetString(PyExc_Exception, "Wrong value set for precision.");
        return -1;
    }
#endif 


#ifdef __DFE__
    if ( self->DFE < 0 || self->DFE > 2 ) {
        PyErr_SetString(PyExc_Exception, "Wrong value set for DFE.");
        return -1;
    } 
#else
    if ( self->DFE < 0 || self->DFE > 0 ) {
        PyErr_SetString(PyExc_Exception, "DFE not implemented. Only DFE=0 (default) value is valid parameter");
        return -1;
    } 
#endif

    if ( self->precision == 1 ) {
        if ( self->input_state || self->output_state) {
            // create instance of class GlynnPermanentCalculator
            self->calculator.cpu_long_double_repeated = create_GlynnPermanentCalculatorRepeated();
        }
        else {
            // create instance of class GlynnPermanentCalculator
            self->calculator.cpu_long_double = create_GlynnPermanentCalculator();
        } 

    }
#ifdef __MPFR__
    else if ( self->precision == 2 ) {
        // create instance of class GlynnPermanentCalculator
        self->calculator.cpu_inf = create_GlynnPermanentCalculatorInf();
    }
#endif
    else {
        PyErr_SetString(PyExc_Exception, "CPU permanent calculator uninitialized.");
    }



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

    // variable to store the calculated permanent
    pic::Complex16 perm;

    if ( self->DFE==0 ) {

        if ( self->precision == 1 ) {

            if ( self->input_state || self->output_state) {

                if ( !self->input_state ) {
                    PyErr_SetString(PyExc_Exception, "Multiplicities of input state not given for class GlynnPermanentCalculatorRepeated.");
                    return NULL;
                }

                if ( !self->output_state ) {
                    PyErr_SetString(PyExc_Exception, "Multiplicities of output state not given for class GlynnPermanentCalculatorRepeated.");
                    return NULL;
                }

                // create PIC version of the input matrices
                pic::PicState_int64 input_state_mtx = numpy2PicState_int64(self->input_state);
                pic::PicState_int64 output_state_mtx = numpy2PicState_int64(self->output_state);

                try {
                    // CPU implementation of permanent calculator with long double precision and repeated rows implementation
                    perm = self->calculator.cpu_long_double_repeated->calculate(matrix_mtx, input_state_mtx, output_state_mtx);
                }
                catch (std::string err) {                    
                    PyErr_SetString(PyExc_Exception, err.c_str());
                    return NULL;
                }

            }
            else {

                try {
                    // CPU implementation of permanent calculator with long double precision
                    perm = self->calculator.cpu_long_double->calculate(matrix_mtx);
                }
                catch (std::string err) {                    
                    PyErr_SetString(PyExc_Exception, err.c_str());
                    return NULL;
                }

                
            } 
            
        }
#ifdef __MPFR__
        else if ( self->precision == 2 ) {
            // start the calculation of the permanent with infinite precision using library MPFR
            perm = self->calculator.cpu_inf->calculate(matrix_mtx);
        }
#endif
        else {
            PyErr_SetString(PyExc_Exception, "Wrong value set for precision.");
            return NULL;
        }
    }
#ifdef __DFE__
    else if (self->DFE==1 || self->DFE==2) {
        // single and dual DFE impelementation for permanent
        if (!calcPermanentGlynnDFE) {
            PyErr_SetString(PyExc_Exception, "DFE calculator handles not initialized.");
            return NULL;
        }

        // initialize DFE if needed
        if (initialize_DFE) initialize_DFE(self->DFE);

        // calculate permanent on DFE
        GlynnPermanentCalculator_DFE( matrix_mtx, perm, self->DFE);
    }
#endif

    return Py_BuildValue("D", &perm);
}



/**
@brief Method to call get attribute matrix
*/
static PyObject *
GlynnPermanentCalculator_Wrapper_getmatrix(GlynnPermanentCalculator_wrapper *self, void *closure)
{

    if ( self->matrix ) {
        Py_INCREF(self->matrix);
        return self->matrix;
    }
    else {
        Py_RETURN_NONE;
    }

}

/**
@brief Method to call set attribute matrix
*/
static int
GlynnPermanentCalculator_Wrapper_setmatrix(GlynnPermanentCalculator_wrapper *self, PyObject *matrix_arg, void *closure)
{

    // set the array on the Python side

    if ( self->matrix ) {
        Py_DECREF(self->matrix);
    }

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        self->matrix = matrix_arg;
        Py_INCREF(self->matrix);
    }
    else {
        self->matrix = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    return 0;
}





/**
@brief Method to call get the input_state
*/
static PyObject *
GlynnPermanentCalculator_Wrapper_getinput_state(GlynnPermanentCalculator_wrapper *self, void *closure)
{

    if ( self->input_state ) {
        Py_INCREF(self->input_state);
        return self->input_state;
    }
    else {
        Py_RETURN_NONE;
    }


}

/**
@brief Method to call set the input_state
*/
static int
GlynnPermanentCalculator_Wrapper_setinput_state(GlynnPermanentCalculator_wrapper *self, PyObject *input_state_arg, void *closure)
{

    // set the array on the Python side
    if ( !PyArray_Check( input_state_arg ) ) {
        PyErr_SetString(PyExc_Exception, "Input state multiplicities must be a numpy array of int64 values");
        return NULL;
    }


    if ( self->input_state ) {
        Py_DECREF(self->input_state);
        self->input_state = NULL;
    }


    if ( PyArray_IS_C_CONTIGUOUS(input_state_arg) ) {
        self->input_state = input_state_arg;
        Py_INCREF(self->input_state);
    }
    else {
        self->input_state = PyArray_FROM_OTF(input_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }

    return 0;
}




/**
@brief Method to call get the output_state
*/
static PyObject *
GlynnPermanentCalculator_Wrapper_getoutput_state(GlynnPermanentCalculator_wrapper *self, void *closure)
{

    if ( self->output_state ) {
        Py_INCREF(self->output_state);
        return self->output_state;
    }
    else {
        Py_RETURN_NONE;
    }


}

/**
@brief Method to call set matrix mean
*/
static int
GlynnPermanentCalculator_Wrapper_setoutput_state(GlynnPermanentCalculator_wrapper *self, PyObject *output_state_arg, void *closure)
{

    // set the array on the Python side

    if ( !PyArray_Check( output_state_arg ) ) {
        PyErr_SetString(PyExc_Exception, "Output state multiplicities must be a numpy array of int64 values");
        return NULL;
    }



    if ( self->output_state ) {
        Py_DECREF(self->output_state);
        self->output_state = NULL;
    }


    if ( PyArray_IS_C_CONTIGUOUS(output_state_arg) ) {
        self->output_state = output_state_arg;
        Py_INCREF(self->output_state);
    }
    else {
        self->output_state = PyArray_FROM_OTF(output_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }

    return 0;
}





/**
@brief get/set functions of class GlynnPermanentCalculator_wrapper.
*/
static PyGetSetDef GlynnPermanentCalculator_wrapper_getsetters[] = {
    {"matrix", (getter) GlynnPermanentCalculator_Wrapper_getmatrix, (setter) GlynnPermanentCalculator_Wrapper_setmatrix,
     "matrix", NULL},
    {"input_state", (getter) GlynnPermanentCalculator_Wrapper_getinput_state, (setter) GlynnPermanentCalculator_Wrapper_setinput_state,
     "input_state", NULL},
    {"output_state", (getter) GlynnPermanentCalculator_Wrapper_getoutput_state, (setter) GlynnPermanentCalculator_Wrapper_setoutput_state,
     "output_state", NULL},
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
