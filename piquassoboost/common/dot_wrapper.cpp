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
#include "dot.h"
#include "tbb/scalable_allocator.h"
#include "numpy_interface.h"





extern "C"
{




/**
@brief Wrapper function to calculate the matrix product A*B
@param self A tuple of arguments
@param args A tuple of the input arguments: ??????????????
*/
static PyObject *
dot_wrapper_dot(PyObject *self, PyObject *args)
{

    // initiate variables for input arguments
    PyObject *A_arg = NULL;
    PyObject *B_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OO",
                                     &A_arg, &B_arg))
        return Py_BuildValue("i", -1);

  
    // establish memory contiguous arrays for C calculations
    PyObject* A = NULL, *B = NULL;  

    if ( PyArray_IS_C_CONTIGUOUS(A_arg) ) {
        A = A_arg;
        Py_INCREF(A); 
    }
    else {
        A = PyArray_FROM_OTF(A_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(B_arg) ) {
        B = B_arg;
        Py_INCREF(B); 
    }
    else {
        B = PyArray_FROM_OTF(B_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix A_mtx = numpy2matrix(A);     
    pic::matrix B_mtx = numpy2matrix(B);     

    // calculate the matrix product on the C++ side
    pic::matrix C_mtx = dot(A_mtx, B_mtx);

    // release C++ matrix from the ownership of the data. (Python would handle the release of the data)
    C_mtx.set_owner( false );      

    // convert C++ matrix into numpy array
    PyObject* C = matrix_to_numpy( C_mtx );


    Py_DECREF(A);   
    Py_DECREF(B);  


    return C;
}



/**
@brief Wrapper function to calculate the matrix product A * B*
@param self A tuple of arguments
@param args A tuple of the input arguments: ??????????????
*/
static PyObject *
dot_wrapper_dot2(PyObject *self, PyObject *args)
{

    // initiate variables for input arguments
    PyObject *A_arg = NULL;
    PyObject *B_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OO",
                                     &A_arg, &B_arg))
        return Py_BuildValue("i", -1);

  
    // establish memory contiguous arrays for C calculations
    PyObject* A = NULL, *B = NULL;  

    if ( PyArray_IS_C_CONTIGUOUS(A_arg) ) {
        A = A_arg;
        Py_INCREF(A); 
    }
    else {
        A = PyArray_FROM_OTF(A_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(B_arg) ) {
        B = B_arg;
        Py_INCREF(B); 
    }
    else {
        B = PyArray_FROM_OTF(B_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix A_mtx = numpy2matrix(A);     
    pic::matrix B_mtx = numpy2matrix(B); 
    B_mtx.conjugate();    

    // calculate the matrix product on the C++ side
    pic::matrix C_mtx = dot(A_mtx, B_mtx);

    // release C++ matrix from the ownership of the data. (Python would handle the release of the data)
    C_mtx.set_owner( false );      

    // convert C++ matrix into numpy array
    PyObject* C = matrix_to_numpy( C_mtx );


    Py_DECREF(A);   
    Py_DECREF(B);  


    return C;
}






/**
@brief Wrapper function to calculate the matrix product A^T * B
@param self A tuple of arguments
@param args A tuple of the input arguments: ??????????????
*/
static PyObject *
dot_wrapper_dot3(PyObject *self, PyObject *args)
{

    // initiate variables for input arguments
    PyObject *A_arg = NULL;
    PyObject *B_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OO",
                                     &A_arg, &B_arg))
        return Py_BuildValue("i", -1);

  
    // establish memory contiguous arrays for C calculations
    PyObject* A = NULL, *B = NULL;  

    if ( PyArray_IS_C_CONTIGUOUS(A_arg) ) {
        A = A_arg;
        Py_INCREF(A); 
    }
    else {
        A = PyArray_FROM_OTF(A_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(B_arg) ) {
        B = B_arg;
        Py_INCREF(B); 
    }
    else {
        B = PyArray_FROM_OTF(B_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix A_mtx = numpy2matrix(A);     
    pic::matrix B_mtx = numpy2matrix(B); 
    A_mtx.transpose();    

    // calculate the matrix product on the C++ side
    pic::matrix C_mtx = dot(A_mtx, B_mtx);

    // release C++ matrix from the ownership of the data. (Python would handle the release of the data)
    C_mtx.set_owner( false );      

    // convert C++ matrix into numpy array
    PyObject* C = matrix_to_numpy( C_mtx );


    Py_DECREF(A);   
    Py_DECREF(B);  


    return C;
}





/**
@brief Wrapper function to calculate the matrix product A^+ * B
@param self A tuple of arguments
@param args A tuple of the input arguments: ??????????????
*/
static PyObject *
dot_wrapper_dot4(PyObject *self, PyObject *args)
{

    // initiate variables for input arguments
    PyObject *A_arg = NULL;
    PyObject *B_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OO",
                                     &A_arg, &B_arg))
        return Py_BuildValue("i", -1);

  
    // establish memory contiguous arrays for C calculations
    PyObject* A = NULL, *B = NULL;  

    if ( PyArray_IS_C_CONTIGUOUS(A_arg) ) {
        A = A_arg;
        Py_INCREF(A); 
    }
    else {
        A = PyArray_FROM_OTF(A_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(B_arg) ) {
        B = B_arg;
        Py_INCREF(B); 
    }
    else {
        B = PyArray_FROM_OTF(B_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix A_mtx = numpy2matrix(A);     
    pic::matrix B_mtx = numpy2matrix(B); 
    A_mtx.transpose();    
    A_mtx.conjugate(); 

    // calculate the matrix product on the C++ side
    pic::matrix C_mtx = dot(A_mtx, B_mtx);

    // release C++ matrix from the ownership of the data. (Python would handle the release of the data)
    C_mtx.set_owner( false );      

    // convert C++ matrix into numpy array
    PyObject* C = matrix_to_numpy( C_mtx );


    Py_DECREF(A);   
    Py_DECREF(B);  


    return C;
}





/**
@brief Wrapper function to calculate the matrix product A^* * B
@param self A tuple of arguments
@param args A tuple of the input arguments: ??????????????
*/
static PyObject *
dot_wrapper_dot5(PyObject *self, PyObject *args)
{

    // initiate variables for input arguments
    PyObject *A_arg = NULL;
    PyObject *B_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OO",
                                     &A_arg, &B_arg))
        return Py_BuildValue("i", -1);

  
    // establish memory contiguous arrays for C calculations
    PyObject* A = NULL, *B = NULL;  

    if ( PyArray_IS_C_CONTIGUOUS(A_arg) ) {
        A = A_arg;
        Py_INCREF(A); 
    }
    else {
        A = PyArray_FROM_OTF(A_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(B_arg) ) {
        B = B_arg;
        Py_INCREF(B); 
    }
    else {
        B = PyArray_FROM_OTF(B_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix A_mtx = numpy2matrix(A);     
    pic::matrix B_mtx = numpy2matrix(B); 
    A_mtx.conjugate(); 

    // calculate the matrix product on the C++ side
    pic::matrix C_mtx = dot(A_mtx, B_mtx);

    // release C++ matrix from the ownership of the data. (Python would handle the release of the data)
    C_mtx.set_owner( false );      

    // convert C++ matrix into numpy array
    PyObject* C = matrix_to_numpy( C_mtx );


    Py_DECREF(A);   
    Py_DECREF(B);  


    return C;
}




/**
@brief Wrapper function to calculate the matrix product A * B^T
@param self A tuple of arguments
@param args A tuple of the input arguments: ??????????????
*/
static PyObject *
dot_wrapper_dot6(PyObject *self, PyObject *args)
{

    // initiate variables for input arguments
    PyObject *A_arg = NULL;
    PyObject *B_arg = NULL;

    // parsing input arguments
    if (!PyArg_ParseTuple(args, "|OO",
                                     &A_arg, &B_arg))
        return Py_BuildValue("i", -1);

  
    // establish memory contiguous arrays for C calculations
    PyObject* A = NULL, *B = NULL;  

    if ( PyArray_IS_C_CONTIGUOUS(A_arg) ) {
        A = A_arg;
        Py_INCREF(A); 
    }
    else {
        A = PyArray_FROM_OTF(A_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    if ( PyArray_IS_C_CONTIGUOUS(B_arg) ) {
        B = B_arg;
        Py_INCREF(B); 
    }
    else {
        B = PyArray_FROM_OTF(B_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }


    // create PIC version of the input matrices
    pic::matrix A_mtx = numpy2matrix(A);     
    pic::matrix B_mtx = numpy2matrix(B); 
    B_mtx.transpose(); 

    // calculate the matrix product on the C++ side
    pic::matrix C_mtx = dot(A_mtx, B_mtx);

    // release C++ matrix from the ownership of the data. (Python would handle the release of the data)
    C_mtx.set_owner( false );      

    // convert C++ matrix into numpy array
    PyObject* C = matrix_to_numpy( C_mtx );


    Py_DECREF(A);   
    Py_DECREF(B);  


    return C;
}



static PyMethodDef dot_wrapper_Methods[] = {
    {"dot", (PyCFunction) dot_wrapper_dot, METH_VARARGS,
     "Method to calculate the matorx product A*B"
    },
    {"dot2", (PyCFunction) dot_wrapper_dot2, METH_VARARGS,
     "Method to calculate the matorx product A * B*"
    },
    {"dot3", (PyCFunction) dot_wrapper_dot3, METH_VARARGS,
     "Method to calculate the matorx product A^T * B"
    },
    {"dot4", (PyCFunction) dot_wrapper_dot4, METH_VARARGS,
     "Method to calculate the matorx product A^+ * B"
    },
    {"dot5", (PyCFunction) dot_wrapper_dot5, METH_VARARGS,
     "Method to calculate the matorx product A^* * B"
    },
    {"dot6", (PyCFunction) dot_wrapper_dot6, METH_VARARGS,
     "Method to calculate the matorx product A * B^T"
    },
    {NULL}  /* Sentinel */
};



/**
@brief Structure containing metadata about the module.
*/
static PyModuleDef dot_wrapper_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "dot_wrapper",
    .m_doc = "Python binding for calculating matrix product by C++ code",
    .m_size = -1,
    dot_wrapper_Methods 
};

/**
@brief Method called when the Python module is initialized
*/
PyMODINIT_FUNC
PyInit_dot_wrapper(void)
{
    // initialize Numpy API
    import_array();

    PyObject *m;

    m = PyModule_Create(&dot_wrapper_Module);
    if (m == NULL)
        return NULL;


    return m;
}



}
