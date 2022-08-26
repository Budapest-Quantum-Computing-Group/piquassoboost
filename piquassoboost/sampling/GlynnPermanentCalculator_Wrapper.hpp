#ifndef GlynnPermanentCalculator_wrapper_H
#define GlynnPermanentCalculator_wrapper_H


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "GlynnPermanentCalculator.h"
#include "GlynnPermanentCalculatorSimple.h"
#include "BBFGPermanentCalculator.h"

#ifdef __DFE__
#include "GlynnPermanentCalculatorDFE.h"
#endif

#include "numpy_interface.h"


#define GlynnLongDouble 0
#define GlynnInf 1
#define GlynnSingleDFE 2
#define GlynnDualDFE 3
#define GlynnSingleDFEF 4
#define GlynnDualDFEF 5
#define GlynnDouble 6
#define BBFGPermanentCalculatorDouble 7
#define BBFGPermanentCalculatorLongDouble 8
#define GlynnSimpleDouble 10
#define GlynnSimpleLongDouble 11



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
    union {
        /// long double precision Glynn calculator
        pic::GlynnPermanentCalculatorLongDouble *cpu_long_double;
        /// double precision Glynn calculator
        pic::GlynnPermanentCalculatorDouble *cpu_double;
        /// BBFG permanent calculator
        pic::BBFGPermanentCalculator *BBFGcalculator;
        /// double precision Glynn calculator
        pic::GlynnPermanentCalculatorSimpleDouble *cpu_simple_double;
        /// double precision Glynn calculator
        pic::GlynnPermanentCalculatorSimpleLongDouble *cpu_simple_long_double;
};
} GlynnPermanentCalculator_wrapper;




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
    if (self->lib == GlynnLongDouble && self->cpu_long_double != NULL) delete self->cpu_long_double;
    else if (self->lib == BBFGPermanentCalculatorDouble && self->BBFGcalculator != NULL) delete self->BBFGcalculator;
    else if (self->lib == BBFGPermanentCalculatorLongDouble && self->BBFGcalculator != NULL) delete self->BBFGcalculator;
#ifdef __MPFR__
    else if (self->lib == GlynnInf && self->BBFGcalculator != NULL) delete self->BBFGcalculator;
#endif
    else if ( self->lib == GlynnDouble && self->cpu_double != NULL ) delete self->cpu_double;
    else if ( self->lib == GlynnSimpleDouble && self->cpu_simple_double != NULL ) delete self->cpu_simple_double;
    else if ( self->lib == GlynnSimpleLongDouble && self->cpu_simple_long_double != NULL ) delete self->cpu_simple_long_double;

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
    self->lib = BBFGPermanentCalculatorDouble;
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
    if (self->lib == GlynnLongDouble) {
        self->cpu_long_double = new pic::GlynnPermanentCalculatorLongDouble();
    }
#ifdef __MPFR__    
    else if (self->lib == GlynnInf) {
        self->BBFGcalculator = new pic::BBFGPermanentCalculator();
    }
#endif
#ifdef __DFE__
    else if (self->lib == GlynnSingleDFE || self->lib == GlynnDualDFE || self->lib == GlynnSingleDFEF || self->lib == GlynnDualDFEF) {
        inc_dfe_lib_count();
    }
#endif
    else if (self->lib == GlynnDouble)
        self->cpu_double = new pic::GlynnPermanentCalculatorDouble();
    else if (self->lib == BBFGPermanentCalculatorRepeatedDouble)
        self->BBFGcalculator = new pic::BBFGPermanentCalculator();
    else if (self->lib == BBFGPermanentCalculatorRepeatedLongDouble)
        self->BBFGcalculator = new pic::BBFGPermanentCalculator();
    else if (self->lib == GlynnSimpleDouble)
        self->cpu_simple_double = new pic::GlynnPermanentCalculatorSimpleDouble();
    else if (self->lib == GlynnSimpleLongDouble)
        self->cpu_simple_long_double = new pic::GlynnPermanentCalculatorSimpleLongDouble();
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
                if (self->lib == GlynnLongDouble) {
                    try {
                        ret[i] = self->cpu_long_double->calculate(matrices[i]);
                    }
                    catch (std::string err) {
                        PyErr_SetString(PyExc_Exception, err.c_str());
                        return NULL;        
                    }
                }
                else if (self->lib == BBFGPermanentCalculatorDouble) {
                    try {
                        ret[i] = self->BBFGcalculator->calculate(matrices[i], false);
                    }
                    catch (std::string err) {
                        PyErr_SetString(PyExc_Exception, err.c_str());
                        return NULL;
                    }
                }
                else if (self->lib == BBFGPermanentCalculatorLongDouble) {
                    try {
                        ret[i] = self->BBFGcalculator->calculate(matrices[i], true);
                    }
                    catch (std::string err) {
                        PyErr_SetString(PyExc_Exception, err.c_str());
                        return NULL;
                    }
                }
                else if (self->lib == GlynnSimpleDouble) {
                    try {
                        ret[i] = self->cpu_simple_double->calculate(matrices[i]);
                    }
                    catch (std::string err) {
                        PyErr_SetString(PyExc_Exception, err.c_str());
                        return NULL;
                    }
                }
                else if (self->lib == GlynnSimpleLongDouble) {
                    try {
                        ret[i] = self->cpu_simple_long_double->calculate(matrices[i]);
                    }
                    catch (std::string err) {
                        PyErr_SetString(PyExc_Exception, err.c_str());
                        return NULL;
                    }
                }
#ifdef __MPFR__
                else if (self->lib == GlynnInf) {
                    ret[i] = self->BBFGcalculator->calculate(matrices[i], false, true);
                }
#endif
                else {
                    PyErr_SetString(PyExc_Exception, "Wrong value set for permanent library.");
                    return NULL;
                }
            }
        }    
        
        PyObject* list = PyList_New(0);
        for (size_t i = 0; i < ret.size(); i++) {
            PyObject* o = Py_BuildValue("D", &ret[i]);
            PyList_Append(list, o);
            Py_DECREF(o);
        }

        for (int i = 0; i < ret.size(); i++){
            std::cout << i << ret[i];
        }
        std::cout << "ret.size()=0";

        return list;


    } 
    else {

        // create PIC version of the input matrices
        pic::matrix matrix_mtx = numpy2matrix(self->matrix);
    
        // start the calculation of the permanent
    
        pic::Complex16 ret;
        ret = 27.0;
        std::cout << "27 in ret\n";

        if (self->lib == GlynnLongDouble) {
            try {
                ret = self->cpu_long_double->calculate(matrix_mtx);
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
            ret = self->BBFGcalculator->calculate(matrix_mtx, false, true);
        }
#endif
        else if (self->lib == GlynnDouble) {
            try {
                ret = self->cpu_double->calculate(matrix_mtx);
            }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
        else if (self->lib == BBFGPermanentCalculatorDouble) {
            try {
                ret = self->BBFGcalculator->calculate(matrix_mtx, false);
            }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
        else if (self->lib == BBFGPermanentCalculatorLongDouble) {
            try {
                ret = self->BBFGcalculator->calculate(matrix_mtx, true);
            }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
        else if (self->lib == GlynnSimpleDouble) {
            try {
                ret = self->cpu_simple_double->calculate(matrix_mtx);
            }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
        else if (self->lib == GlynnSimpleLongDouble) {
            try {
                ret = self->cpu_simple_long_double->calculate(matrix_mtx);
            }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
        else {
            PyErr_SetString(PyExc_Exception, "Wrong value set for permanent library.");
            return NULL;
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
