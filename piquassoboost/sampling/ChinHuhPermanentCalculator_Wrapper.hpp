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

#ifndef ChinHuhPermanentCalculator_wrapper_H
#define ChinHuhPermanentCalculator_wrapper_H


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "CChinHuhPermanentCalculator.h"
#include "GlynnPermanentCalculatorRepeated.h"
#include "BBFGPermanentCalculatorRepeated.h"

#ifdef __DFE__
#include "GlynnPermanentCalculatorDFE.h"
#include "GlynnPermanentCalculatorRepeatedDFE.h"
#endif

#include "numpy_interface.h"


#define ChinHuhDouble 0
#define GlynnRep 1
#define GlynnRepSingleDFE 2
#define GlynnRepDualDFE 3
#define GlynnRepMultiSingleDFE 4
#define GlynnRepMultiDualDFE 5
#define GlynnRepCPUDouble 6
#define BBFGPermanentCalculatorRepeatedDouble 7
#define BBFGPermanentCalculatorRepeatedLongDouble 8
#define GlynnRepInf 9
#define GlynnRepSingleDFEF 10
#define GlynnRepDualDFEF 11






/**
@brief Type definition of the ChinHuhPermanentCalculator_wrapper Python class of the ChinHuhPermanentCalculator_wrapper module
*/
typedef struct ChinHuhPermanentCalculator_wrapper {
    PyObject_HEAD
    int lib;
    /// pointer to numpy matrix to keep it alive
    PyObject *matrix = NULL;
    /// pointer to numpy matrix of input states to keep it alive
    PyObject *input_state = NULL;
    /// pointer to numpy matrix of output states to keep it alive
    PyObject *output_state = NULL;
    /// The C++ variant of class CChinHuhPermanentCalculator
    union {
        pic::CChinHuhPermanentCalculator* calculator;
        pic::GlynnPermanentCalculatorRepeatedDouble* calculatorRepDouble;
        pic::GlynnPermanentCalculatorRepeatedLongDouble* calculatorRepLongDouble;
        pic::BBFGPermanentCalculatorRepeated* BBFGcalculatorRep;
    };
} ChinHuhPermanentCalculator_wrapper;








extern "C"
{




/**
@brief Method called when a python instance of the class ChinHuhPermanentCalculator_wrapper is destroyed
@param self A pointer pointing to an instance of class ChinHuhPermanentCalculator_wrapper.
*/
static void
ChinHuhPermanentCalculator_wrapper_dealloc(ChinHuhPermanentCalculator_wrapper *self)
{
#ifdef __DFE__
    if (self->lib == GlynnRepSingleDFE || self->lib == GlynnRepDualDFE || self->lib == GlynnRepMultiSingleDFE || self->lib == GlynnRepMultiDualDFE)
        dec_dfe_lib_count();
    else
#endif

    // deallocate the instance of class N_Qubit_Decomposition
    if (self->lib == ChinHuhDouble) delete self->calculator;
    else if (self->lib == GlynnRep && self->calculatorRepLongDouble != NULL) delete self->calculatorRepLongDouble;
    else if (self->lib == GlynnRepCPUDouble && self->calculatorRepDouble != NULL) delete self->calculatorRepDouble;
    else if (self->lib == BBFGPermanentCalculatorRepeatedDouble && self->BBFGcalculatorRep != NULL) delete self->BBFGcalculatorRep;
    else if (self->lib == BBFGPermanentCalculatorRepeatedLongDouble && self->BBFGcalculatorRep != NULL) delete self->BBFGcalculatorRep;
    else if (self->lib == GlynnRepInf && self->BBFGcalculatorRep != NULL) delete self->BBFGcalculatorRep;

    // release numpy arrays
    if (self->matrix) Py_DECREF(self->matrix);
    if (self->input_state) Py_DECREF(self->input_state);
    if (self->output_state) Py_DECREF(self->output_state);

    Py_TYPE(self)->tp_free((PyObject *) self);
}

/**
@brief Method called when a python instance of the class ChinHuhPermanentCalculator_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class ChinHuhPermanentCalculator_wrapper.
*/
static PyObject *
ChinHuhPermanentCalculator_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    ChinHuhPermanentCalculator_wrapper *self;
    self = (ChinHuhPermanentCalculator_wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}

    self->matrix = NULL;
    self->input_state = NULL;
    self->output_state = NULL;

    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class ChinHuhPermanentCalculator_wrapper is initialized
@param self A pointer pointing to an instance of the class ChinHuhPermanentCalculator_wrapper.
@param args A tuple of the input arguments: qbit_num (integer)
qbit_num: the number of qubits spanning the operations
@param kwds A tuple of keywords
*/
static int
ChinHuhPermanentCalculator_wrapper_init(ChinHuhPermanentCalculator_wrapper *self, PyObject *args, PyObject *kwds)
{
    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"lib", (char*)"matrix", (char*)"input_state", (char*)"output_state", NULL};

    // initiate variables for input arguments
    PyObject *matrix_arg = NULL;
    PyObject *input_state_arg = NULL;
    PyObject *output_state_arg = NULL;

    self->lib = BBFGPermanentCalculatorRepeatedDouble;
    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iOOO", kwlist,
                                     &self->lib, &matrix_arg, &input_state_arg, &output_state_arg))
        return -1;

    // convert python object array to numpy C API array
    if ( matrix_arg == NULL ) return -1;
    if ( input_state_arg == NULL ) return -1;
    if ( output_state_arg == NULL ) return -1;

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        self->matrix = matrix_arg;
        Py_INCREF(self->matrix);
    }
    else {
        self->matrix = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyList_Check(input_state_arg) || PyArray_IS_C_CONTIGUOUS(input_state_arg) ) {
        self->input_state = input_state_arg;
        Py_INCREF(self->input_state);
    }
    else {
        self->input_state = PyArray_FROM_OTF(input_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyList_Check(output_state_arg) || PyArray_IS_C_CONTIGUOUS(output_state_arg) ) {
        self->output_state = output_state_arg;
        Py_INCREF(self->output_state);
    }
    else {
        self->output_state = PyArray_FROM_OTF(output_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }

  
    // create instance of class ChinHuhPermanentCalculator
    if (self->lib == ChinHuhDouble) 
        self->calculator = new pic::CChinHuhPermanentCalculator();
    else if (self->lib == GlynnRep) {
        self->calculatorRepLongDouble = new pic::GlynnPermanentCalculatorRepeatedLongDouble();
    }
    else if (self->lib == GlynnRepInf) {
        self->BBFGcalculatorRep = new pic::BBFGPermanentCalculatorRepeated();
    }
#ifdef __DFE__
    else if (self->lib == GlynnRepSingleDFE || self->lib == GlynnRepDualDFE || self->lib == GlynnRepMultiSingleDFE || self->lib == GlynnRepMultiDualDFE)
        inc_dfe_lib_count();
#endif
    else if (self->lib == GlynnRepCPUDouble)
        self->calculatorRepDouble = new pic::GlynnPermanentCalculatorRepeatedDouble();
    else if (self->lib == BBFGPermanentCalculatorRepeatedDouble)
        self->BBFGcalculatorRep = new pic::BBFGPermanentCalculatorRepeated(); 
    else if (self->lib == BBFGPermanentCalculatorRepeatedLongDouble)
        self->BBFGcalculatorRep = new pic::BBFGPermanentCalculatorRepeated();        
    else {
        PyErr_SetString(PyExc_Exception, "Wrong value set for permanent library.");
        return -1;
    }

    return 0;
}



/**
@brief Wrapper function to call the calculate method of C++ class CChinHuhPermanentCalculator
@param self A pointer pointing to an instance of the class ChinHuhPermanentCalculator_Wrapper.
@param args A tuple of the input arguments: ??????????????
@param kwds A tuple of keywords
*/
static PyObject *
ChinHuhPermanentCalculator_Wrapper_calculate(ChinHuhPermanentCalculator_wrapper *self)
{

    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(self->matrix);

    if (PyList_Check(self->input_state) && PyList_Check(self->output_state)) {
        Py_ssize_t sz = PyList_Size(self->input_state);
        //Py_ssize_t szo = PyList_Size(self->output_state);
        std::vector<std::vector<pic::Complex16>> ret;
        ret.resize(sz);
        int multiInput = sz != 0 && PyList_Check(PyList_GetItem(self->input_state, 0));

        std::vector<pic::PicState_int64> input_states;
        std::vector<std::vector<pic::PicState_int64>> output_states(sz);
        input_states.reserve(sz);
        for (Py_ssize_t i = 0; i < sz; i++) {
            PyObject *o = PyList_GetItem(multiInput ? self->output_state : self->input_state, i);            
            if ( PyArray_IS_C_CONTIGUOUS(o) ) {
                Py_INCREF(o);
            } else {
                o = PyArray_FROM_OTF(o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
            }
            PyList_SetItem(multiInput ? self->output_state : self->input_state, i, o);
            input_states.push_back(numpy2PicState_int64(o));
            PyObject* oOut = PyList_GetItem(multiInput ? self->input_state : self->output_state, i);
            Py_INCREF(oOut);
            Py_ssize_t szOutput = PyList_Size(oOut);
            output_states[i].reserve(szOutput);
            for (Py_ssize_t j = 0; j < szOutput; j++) {
                o = PyList_GetItem(oOut, j);
                if ( PyArray_IS_C_CONTIGUOUS(o) ) {
                    Py_INCREF(o);
                } else {
                    o = PyArray_FROM_OTF(o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
                }
                PyList_SetItem(oOut, i, o);
                output_states[i].push_back(numpy2PicState_int64(o));
            }
            PyList_SetItem(multiInput ? self->input_state : self->output_state, i, oOut);
        }
        for (size_t i = 0; i < input_states.size(); i++) {
            ret[i].resize(output_states[i].size());
        }
#ifdef __DFE__        
        if (self->lib == GlynnRepSingleDFE || self->lib == GlynnRepDualDFE ||  self->lib == GlynnRepSingleDFEF || self->lib == GlynnRepDualDFEF)
            if (multiInput) GlynnPermanentCalculatorRepeatedInputBatch_DFE( matrix_mtx, output_states, input_states, ret, self->lib == GlynnRepDualDFE || self->lib == GlynnRepDualDFEF, self->lib == GlynnRepSingleDFEF || self->lib == GlynnRepDualDFEF);
            else GlynnPermanentCalculatorRepeatedOutputBatch_DFE( matrix_mtx, input_states, output_states, ret, self->lib == GlynnRepDualDFE || self->lib == GlynnRepDualDFEF, self->lib == GlynnRepSingleDFEF || self->lib == GlynnRepDualDFEF);
/*
        else if (self->lib == GlynnRepMultiSingleDFE || self->lib == GlynnRepMultiDualDFE) 
            if (multiInput) GlynnPermanentCalculatorRepeatedMultiInputBatch_DFE( matrix_mtx, output_states, input_states, ret, self->lib == GlynnRepMultiDualDFE);
            else GlynnPermanentCalculatorRepeatedMultiOutputBatch_DFE( matrix_mtx, input_states, output_states, ret, self->lib == GlynnRepMultiDualDFE);
*/
        else
#endif
        {
            for (size_t i = 0; i < input_states.size(); i++) {
                for (size_t j = 0; j < output_states[i].size(); j++) {
                    try {
                        if (self->lib == ChinHuhDouble)
                            ret[i][j] = self->calculator->calculate(matrix_mtx, multiInput ? output_states[i][j] : input_states[i], multiInput ? input_states[i] : output_states[i][j]);
                        else if (self->lib == GlynnRep)
                            ret[i][j] = self->calculatorRepLongDouble->calculate(matrix_mtx, multiInput ? output_states[i][j] : input_states[i], multiInput ? input_states[i] : output_states[i][j]);
                        else if (self->lib == GlynnRepCPUDouble)
                            ret[i][j] = self->calculatorRepDouble->calculate(matrix_mtx, multiInput ? output_states[i][j] : input_states[i], multiInput ? input_states[i] : output_states[i][j]);
                        else if (self->lib == BBFGPermanentCalculatorRepeatedDouble)
                            ret[i][j] = self->BBFGcalculatorRep->calculate(matrix_mtx, multiInput ? output_states[i][j] : input_states[i], multiInput ? input_states[i] : output_states[i][j], false);
                        else if (self->lib == BBFGPermanentCalculatorRepeatedLongDouble)
                            ret[i][j] = self->BBFGcalculatorRep->calculate(matrix_mtx, multiInput ? output_states[i][j] : input_states[i], multiInput ? input_states[i] : output_states[i][j], true);
#ifdef __MPFR__

                        else if (self->lib == GlynnRepInf)
                            ret[i][j] = self->BBFGcalculatorRep->calculate(matrix_mtx, multiInput ? output_states[i][j] : input_states[i], multiInput ? input_states[i] : output_states[i][j], false, true);

#endif
                    }
                    catch (std::string err) {
                        PyErr_SetString(PyExc_Exception, err.c_str());
                        return NULL;
                    }
                }
            }
        }
        
        PyObject* list = PyList_New(0);
        for (size_t i = 0; i < ret.size(); i++) {
            PyObject* innerList = PyList_New(0);            
            for (size_t j = 0; j < ret[i].size(); j++) {
                PyObject* o = Py_BuildValue("D", &ret[i][j]);
                PyList_Append(innerList, o);
                Py_DECREF(o);
            }
            PyList_Append(list, innerList);
            Py_DECREF(innerList);
        }
        return list;  


      
    }
    else {
        pic::PicState_int64 input_state_mtx = numpy2PicState_int64(self->input_state);
        pic::PicState_int64 output_state_mtx = numpy2PicState_int64(self->output_state);

        // start the calculation of the permanent
        pic::Complex16 ret;
    
        if (self->lib == ChinHuhDouble) {
            try {
                ret = self->calculator->calculate(matrix_mtx, input_state_mtx, output_state_mtx);
            }
            catch (std::string err) {                    
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
        else if (self->lib == GlynnRep) {
            try {
                ret = self->calculatorRepLongDouble->calculate(matrix_mtx, input_state_mtx, output_state_mtx);
            }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
#ifdef __DFE__    
        else if (self->lib == GlynnRepSingleDFE || self->lib == GlynnRepDualDFE || self->lib == GlynnRepSingleDFEF || self->lib == GlynnRepDualDFEF)
            GlynnPermanentCalculatorRepeated_DFE( matrix_mtx, input_state_mtx, output_state_mtx, ret, self->lib == GlynnRepDualDFE || self->lib == GlynnRepDualDFEF, self->lib == GlynnRepSingleDFEF || self->lib == GlynnRepDualDFEF);
        else if (self->lib == GlynnRepMultiSingleDFE || self->lib == GlynnRepMultiDualDFE) 
            GlynnPermanentCalculatorRepeatedMulti_DFE( matrix_mtx, input_state_mtx, output_state_mtx, ret, self->lib == GlynnRepMultiDualDFE);
#endif
        else if (self->lib == GlynnRepCPUDouble) {
            try {
                ret = self->calculatorRepDouble->calculate(matrix_mtx, input_state_mtx, output_state_mtx);
            }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
        else if (self->lib == BBFGPermanentCalculatorRepeatedDouble) {
            try {
                ret = self->BBFGcalculatorRep->calculate(matrix_mtx, input_state_mtx, output_state_mtx, false);
            }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
        else if (self->lib == BBFGPermanentCalculatorRepeatedLongDouble) {
            try {
                ret = self->BBFGcalculatorRep->calculate(matrix_mtx, input_state_mtx, output_state_mtx, true);
        }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
#ifdef __MPFR__
        else if (self->lib == GlynnRepInf) {
            try {
                ret = self->BBFGcalculatorRep->calculate(matrix_mtx, input_state_mtx, output_state_mtx, false, true);
        }
            catch (std::string err) {
                PyErr_SetString(PyExc_Exception, err.c_str());
                return NULL;
            }
        }
#endif
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
ChinHuhPermanentCalculator_wrapper_getmatrix(ChinHuhPermanentCalculator_wrapper *self, void *closure)
{
    Py_INCREF(self->matrix);
    return self->matrix;
}

/**
@brief Method to call set attribute matrix
*/
static int
ChinHuhPermanentCalculator_wrapper_setmatrix(ChinHuhPermanentCalculator_wrapper *self, PyObject *matrix_arg, void *closure)
{
    // set the array on the Python side
    Py_DECREF(self->matrix);

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
ChinHuhPermanentCalculator_wrapper_getinput_state(ChinHuhPermanentCalculator_wrapper *self, void *closure)
{
    Py_INCREF(self->input_state);
    return self->input_state;
}

/**
@brief Method to call set the input_state
*/
static int
ChinHuhPermanentCalculator_wrapper_setinput_state(ChinHuhPermanentCalculator_wrapper *self, PyObject *input_state_arg, void *closure)
{
    // set the array on the Python side
    Py_DECREF(self->input_state);

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
ChinHuhPermanentCalculator_wrapper_getoutput_state(ChinHuhPermanentCalculator_wrapper *self, void *closure)
{
    Py_INCREF(self->output_state);
    return self->output_state;
}

/**
@brief Method to call set matrix mean
*/
static int
ChinHuhPermanentCalculator_wrapper_setoutput_state(ChinHuhPermanentCalculator_wrapper *self, PyObject *output_state_arg, void *closure)
{

    // set the array on the Python side
    Py_DECREF(self->output_state);

    if ( PyArray_IS_C_CONTIGUOUS(output_state_arg) ) {
        self->output_state = output_state_arg;
        Py_INCREF(self->output_state);
    }
    else {
        self->output_state = PyArray_FROM_OTF(output_state_arg, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    }

    return 0;
}



static PyGetSetDef ChinHuhPermanentCalculator_wrapper_getsetters[] = {
    {"matrix", (getter) ChinHuhPermanentCalculator_wrapper_getmatrix, (setter) ChinHuhPermanentCalculator_wrapper_setmatrix,
     "matrix", NULL},
    {"input_state", (getter) ChinHuhPermanentCalculator_wrapper_getinput_state, (setter) ChinHuhPermanentCalculator_wrapper_setinput_state,
     "input_state", NULL},
    {"output_state", (getter) ChinHuhPermanentCalculator_wrapper_getoutput_state, (setter) ChinHuhPermanentCalculator_wrapper_setoutput_state,
     "output_state", NULL},
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class ChinHuhPermanentCalculator_wrapper.
*/
static PyMemberDef ChinHuhPermanentCalculator_wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef ChinHuhPermanentCalculator_wrapper_Methods[] = {
    {"calculate", (PyCFunction) ChinHuhPermanentCalculator_Wrapper_calculate, METH_NOARGS,
     "Method to calculate the permanent."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class ChinHuhPermanentCalculator_wrapper.
*/
static PyTypeObject ChinHuhPermanentCalculator_wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "ChinHuhPermanentCalculator_wrapper.ChinHuhPermanentCalculator_wrapper", /*tp_name*/
  sizeof(ChinHuhPermanentCalculator_wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) ChinHuhPermanentCalculator_wrapper_dealloc, /*tp_dealloc*/
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
  ChinHuhPermanentCalculator_wrapper_Methods, /*tp_methods*/
  ChinHuhPermanentCalculator_wrapper_Members, /*tp_members*/
  ChinHuhPermanentCalculator_wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) ChinHuhPermanentCalculator_wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  ChinHuhPermanentCalculator_wrapper_new, /*tp_new*/
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



#endif //ChinHuhPermanentCalculator_wrapper
