#ifndef GlynnPermanentCalculator_wrapper_H
#define GlynnPermanentCalculator_wrapper_H


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "GlynnPermanentCalculator.h"

#ifdef __MPFR__
#include "GlynnPermanentCalculatorInf.h"
#endif

#ifdef __DFE__
#include "GlynnPermanentCalculatorDFE.h"
#endif

#include "numpy_interface.h"


#define GlynnCPP 0
#define GlynnInf 1
#define GlynnSingleDFE 2
#define GlynnDualDFE 3
#define GlynnSingleDFEF 4
#define GlynnDualDFEF 5
#define GlynnDoubleCPU 6
#define GlynnFloatCPU 7

/// The C++ variants of class CGlynnPermanentCalculator
union CPU_glynn {
    /// long double precision calculator
    pic::GlynnPermanentCalculatorLongDouble *cpu_long_double;
    /// double precision calculator
    pic::GlynnPermanentCalculatorDouble *cpu_double;
    /// float precision calculator
    pic::GlynnPermanentCalculatorFloat *cpu_float;
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
    /// set 0 to use CPU implementation, set 1 to use infinite computational precision using the GNU MPFR library, set 2 to use single DFE implementation, set 3 to use dual DFE implementation
    int lib;
    /// CPU permanent calculator
    CPU_glynn calculator;
} GlynnPermanentCalculator_wrapper;


/**
 * @brief Creates an instance of class GlynnPermanentCalculatorLongDouble (long double precision) and returns with a pointer pointing to the class instance (C++ linking is needed)
 * @return Return with a void pointer pointing to an instance of create_GlynnPermanentCalculatorLongDouble class.
 */
pic::GlynnPermanentCalculatorLongDouble*
create_GlynnPermanentCalculatorLongDouble() {

    return new pic::GlynnPermanentCalculatorLongDouble();
}


/**
 * @brief Creates an instance of class GlynnPermanentCalculatorDouble (double precision) and returns with a pointer pointing to the class instance (C++ linking is needed)
 * @return Return with a void pointer pointing to an instance of GlynnPermanentCalculatorDouble class.
 */
pic::GlynnPermanentCalculatorDouble*
create_GlynnPermanentCalculatorDouble() {

    return new pic::GlynnPermanentCalculatorDouble();
}


/**
 * @brief Creates an instance of class GlynnPermanentCalculatorFloat (float precision) and returns with a pointer pointing to the class instance (C++ linking is needed)
 * @return Return with a void pointer pointing to an instance of GlynnPermanentCalculatorFloat class.
 */
pic::GlynnPermanentCalculatorFloat*
create_GlynnPermanentCalculatorFloat() {

    return new pic::GlynnPermanentCalculatorFloat();
}


/**
@brief Call to deallocate an instance of GlynnPermanentCalculatorLongDouble class
@param ptr A pointer pointing to an instance of GlynnPermanentCalculatorLongDouble class.
*/
void
release_GlynnPermanentCalculatorLongDouble( pic::GlynnPermanentCalculatorLongDouble* instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}


/**
@brief Call to deallocate an instance of GlynnPermanentCalculatorDouble class
@param ptr A pointer pointing to an instance of GlynnPermanentCalculatorDouble class.
*/
void
release_GlynnPermanentCalculatorDouble( pic::GlynnPermanentCalculatorDouble* instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}


/**
@brief Call to deallocate an instance of GlynnPermanentCalculatorFloat class
@param ptr A pointer pointing to an instance of GlynnPermanentCalculatorFloat class.
*/
void
release_GlynnPermanentCalculatorFloat( pic::GlynnPermanentCalculatorFloat* instance ) {
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
    if (self->lib == GlynnSingleDFE || self->lib == GlynnDualDFE || self->lib == GlynnSingleDFEF || self->lib == GlynnDualDFEF)
        dec_dfe_lib_count();
    else
#endif
    // deallocate the instance of class N_Qubit_Decomposition
    if (self->lib == GlynnCPP && self->calculator.cpu_long_double != NULL) release_GlynnPermanentCalculatorLongDouble( self->calculator.cpu_long_double );
#ifdef __MPFR__
    else if (self->lib == GlynnInf && self->calculator.cpu_inf != NULL) release_GlynnPermanentCalculatorInf( self->calculator.cpu_inf );
#endif
    else if ( self->lib == GlynnDoubleCPU && self->calculator.cpu_double != NULL ) {
        release_GlynnPermanentCalculatorDouble( self->calculator.cpu_double );
    }
    else if ( self->lib == GlynnFloatCPU && self->calculator.cpu_float != NULL ) {
        release_GlynnPermanentCalculatorFloat( self->calculator.cpu_float );
    }

    // release numpy arrays
    Py_DECREF(self->matrix);
   
    Py_TYPE(self)->tp_free((PyObject *) self);
}

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
    static char *kwlist[] = {(char*)"matrix", (char*)"lib", NULL};
    
    // initiate variables for input arguments
    PyObject *matrix_arg = NULL;    
    self->lib = GlynnCPP;
    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Oi", kwlist, &matrix_arg, 
                                     &self->lib))
        return -1;

    // convert python object array to numpy C API array
    if ( matrix_arg == NULL ) return -1;

    // establish memory contiguous arrays for C calculations
    if ( PyList_Check(matrix_arg) || PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        self->matrix = matrix_arg;
        Py_INCREF(self->matrix);
    }
    else {
        self->matrix = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }
    
    // create instance of class GlynnPermanentCalculator
    if (self->lib == GlynnCPP) {
        self->calculator.cpu_long_double = create_GlynnPermanentCalculatorLongDouble();
    }
#ifdef __MPFR__    
    else if (self->lib == GlynnInf) {
        self->calculator.cpu_inf = new pic::GlynnPermanentCalculatorInf();
    }
#endif
#ifdef __DFE__
    else if (self->lib == GlynnSingleDFE || self->lib == GlynnDualDFE || self->lib == GlynnSingleDFEF || self->lib == GlynnDualDFEF) {
        inc_dfe_lib_count();
    }
#endif
    else if (self->lib == GlynnDoubleCPU) {
        self->calculator.cpu_double = create_GlynnPermanentCalculatorDouble();
    }
    else if (self->lib == GlynnFloatCPU) {
        self->calculator.cpu_float = create_GlynnPermanentCalculatorFloat();
    }
    else {
        PyErr_SetString(PyExc_Exception, "Wrong value set for permanent library.");
        return -1;
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
GlynnPermanentCalculator_Wrapper_calculate(GlynnPermanentCalculator_wrapper *self, PyObject *args, PyObject *kwds)
{

    if (PyList_Check(self->matrix)) {
        // convert list of input numpy arrays into a vector of matices
        Py_ssize_t sz = PyList_Size(self->matrix);
        std::vector<pic::matrix> matrices;
        matrices.reserve(sz);
        for (Py_ssize_t i = 0; i < sz; i++) {
            PyObject *o = PyList_GetItem(self->matrix, i);            
            if ( PyArray_IS_C_CONTIGUOUS(o) ) {
                Py_INCREF(o);
            } else {
                o = PyArray_FROM_OTF(o, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
            }
            PyList_SetItem(self->matrix, i, o);
            matrices.push_back(numpy2matrix(o));
        }
 
        // allocate space for the resulted permanents
        pic::matrix ret(1, sz);

#ifdef __DFE__        
        if (self->lib == GlynnSingleDFE || self->lib == GlynnDualDFE || self->lib == GlynnSingleDFEF || self->lib == GlynnDualDFEF)
            GlynnPermanentCalculatorBatch_DFE( matrices, ret, self->lib == GlynnDualDFE || self->lib == GlynnDualDFEF, self->lib == GlynnSingleDFEF || self->lib == GlynnDualDFEF);
        else
#endif
        {
            for (size_t i = 0; i < matrices.size(); i++) {
                if (self->lib == GlynnCPP) {
                    try {
                        ret[i] = self->calculator.cpu_long_double->calculate(matrices[i]);
                    }
                    catch (std::string err) {
                        PyErr_SetString(PyExc_Exception, err.c_str());
                        return NULL;        
                    }
                }
#ifdef __MPFR__
                else if (self->lib == GlynnInf) {
                    ret[i] = self->calculator.cpu_inf->calculate(matrices[i]);
                }
#endif
            }
        }    
        
        PyObject* list = PyList_New(0);
        for (size_t i = 0; i < ret.size(); i++) {
            PyObject* o = Py_BuildValue("D", &ret[i]);
            PyList_Append(list, o);
            Py_DECREF(o);
        }
        return list;


    } 
    else {

        // create PIC version of the input matrices
        pic::matrix matrix_mtx = numpy2matrix(self->matrix);
    
        // start the calculation of the permanent
    
        pic::Complex16 ret;

        if (self->lib == GlynnCPP) {
            try {
                ret = self->calculator.cpu_long_double->calculate(matrix_mtx);
            }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;         
            }
        }
#ifdef __DFE__        
        else if (self->lib == GlynnSingleDFE || self->lib == GlynnDualDFE || self->lib == GlynnSingleDFEF || self->lib == GlynnDualDFEF) {
            GlynnPermanentCalculator_DFE( matrix_mtx, ret, self->lib == GlynnDualDFE || self->lib == GlynnDualDFEF, self->lib == GlynnSingleDFEF || self->lib == GlynnDualDFEF);
        }
#endif
#ifdef __MPFR__
        else if (self->lib == GlynnInf) {
            ret = self->calculator.cpu_inf->calculate(matrix_mtx);
        }
#endif
        else if (self->lib == GlynnDoubleCPU) {
            try {
                ret = self->calculator.cpu_double->calculate(matrix_mtx);
            }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
        else if (self->lib == GlynnFloatCPU) {
            try {
                ret = self->calculator.cpu_float->calculate(matrix_mtx);
            }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
    
        return Py_BuildValue("D", &ret);
    }
}


/**
@brief Method to call get attribute matrix
*/
static PyObject *
GlynnPermanentCalculator_wrapper_getmatrix(GlynnPermanentCalculator_wrapper *self, void *closure)
{
    Py_INCREF(self->matrix);
    return self->matrix;
}

/**
@brief Method to call set attribute matrix
*/
static int
GlynnPermanentCalculator_wrapper_setmatrix(GlynnPermanentCalculator_wrapper *self, PyObject *matrix_arg, void *closure)
{
    // set the array on the Python side
    Py_DECREF(self->matrix);

    // establish memory contiguous arrays for C calculations
    if ( PyList_Check(matrix_arg) || PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        self->matrix = matrix_arg;
        Py_INCREF(self->matrix);
    }
    else {
        self->matrix = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    return 0;
}

static PyGetSetDef GlynnPermanentCalculator_wrapper_getsetters[] = {
    {"matrix", (getter) GlynnPermanentCalculator_wrapper_getmatrix, (setter) GlynnPermanentCalculator_wrapper_setmatrix,
     "matrix", NULL},
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
