#ifndef GlynnPermanentCalculator_wrapper_H
#define GlynnPermanentCalculator_wrapper_H


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "GlynnPermanentCalculator.h"
#include "numpy_interface.h"

// the maximal dimension of matrix to be ported to FPGA for permanent calculation
#define MAX_FPGA_DIM 40
#define MAX_SINGLE_FPGA_DIM 28

/**
@brief Type definition of the GlynnPermanentCalculator_wrapper Python class of the GlynnPermanentCalculator_wrapper module
*/
typedef struct GlynnPermanentCalculator_wrapper {
    PyObject_HEAD
    /// pointer to numpy matrix to keep it alive
    PyObject *matrix = NULL;
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


void calcPermanentGlynn_singleDFE(const pic::Complex16* mtx_data[4], const double* renormalize_data, const uint64_t rows, const uint64_t cols, pic::Complex16* perm);
void calcPermanentGlynnDFEDualCard(const pic::Complex16* mtx_data[8], const double* renormalize_data, const uint64_t rows, const uint64_t cols, pic::Complex16* perm);


/**
@brief Method called when a python instance of the class GlynnPermanentCalculator_wrapper is destroyed
@param self A pointer pointing to an instance of class GlynnPermanentCalculator_wrapper.
*/
static void
GlynnPermanentCalculator_wrapper_dealloc(GlynnPermanentCalculator_wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_GlynnPermanentCalculator( self->calculator );

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




pic::Complex16 calcPermanenent_DFE(pic::matrix &matrix_mtx){

    // testing purpose!!!
    /*
    if (0 && matrix_mtx.rows != matrix_mtx.cols){
        std::cout << "given matrix:\n";
	matrix_mtx.print_matrix();
    }
    */

    pic::Complex16* mtx_data = matrix_mtx.get_data();


    // calulate the maximal sum of the columns to normalize the matrix
    pic::matrix colSumMax( matrix_mtx.cols, 1);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(pic::Complex16) );
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            pic::Complex16 value1 = colSumMax[jdx] + matrix_mtx[ idx*matrix_mtx.stride + jdx];
            pic::Complex16 value2 = colSumMax[jdx] - matrix_mtx[ idx*matrix_mtx.stride + jdx];
            if ( std::abs( value1 ) < std::abs( value2 ) ) {
                colSumMax[jdx] = value2;
            }
            else {
                colSumMax[jdx] = value1;
            }

        }

    }




    // calculate the renormalization coefficients
    pic::matrix_base<double> renormalize_data(matrix_mtx.cols, 1);
    for (int jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(colSumMax[jdx]);
    }

    // renormalize the input matrix
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            matrix_mtx[ idx*matrix_mtx.stride + jdx] = matrix_mtx[ idx*matrix_mtx.stride + jdx]/renormalize_data[jdx];
        }

    }    

    // SLR and DFE split input matrices
    pic::matrix mtx_split[4];
    pic::Complex16* mtx_data_split[4];


    size_t max_fpga_rows =  MAX_SINGLE_FPGA_DIM;
    size_t max_fpga_cols =  MAX_SINGLE_FPGA_DIM/4;

    // SLR splitted data for the DFE card

    size_t rows = matrix_mtx.rows;
    size_t cols_split[4];
    cols_split[0] = max_fpga_cols < matrix_mtx.cols ? max_fpga_cols : matrix_mtx.cols;
    cols_split[1] = max_fpga_cols < (matrix_mtx.cols-cols_split[0]) ? max_fpga_cols : (matrix_mtx.cols-cols_split[0]);
    cols_split[2] = max_fpga_cols < (matrix_mtx.cols - cols_split[0] - cols_split[1]) ? max_fpga_cols : (matrix_mtx.cols - cols_split[0] - cols_split[1]);
    cols_split[3] = max_fpga_cols < (matrix_mtx.cols - cols_split[0] - cols_split[1] - cols_split[2]) ? max_fpga_cols : (matrix_mtx.cols - cols_split[0] - cols_split[1] - cols_split[2]);


    size_t col_offset = 0;
    for (int kdx=0; kdx<4; kdx++) {

        mtx_split[kdx] = pic::matrix(max_fpga_rows, max_fpga_cols);
        mtx_data_split[kdx] = mtx_split[kdx].get_data();

        pic::Complex16 padding_element(1.0,0.0);
        for (size_t idx=0; idx<rows; idx++) {
            size_t offset = idx*matrix_mtx.stride+col_offset;
            size_t offset_small = idx*mtx_split[kdx].stride;
            for (size_t jdx=0; jdx<cols_split[kdx]; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = matrix_mtx[offset+jdx];
            }

            for (size_t jdx=cols_split[kdx]; jdx<max_fpga_cols; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = padding_element;
            }
            padding_element.real(0.0);
        }
        col_offset = col_offset + cols_split[kdx];

        memset( mtx_data_split[kdx] + rows*mtx_split[kdx].stride, 0.0, (max_fpga_rows-rows)*max_fpga_cols*sizeof(pic::Complex16));
    }
/*
matrix_mtx.print_matrix();
for (int idx=0; idx<4; idx++) {
   mtx_split[idx].print_matrix();
}
*/
    // testing purpose!!!
    /*
    if (0 && matrix_mtx.rows != matrix_mtx.cols){
        std::cout << "created matrices:\n";
	
	for (int idx=0; idx<4; idx++) {
          mtx_split[idx].print_matrix();
        }

    }
    */
   pic::Complex16 perm;
    calcPermanentGlynn_singleDFE( mtx_data_split, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, &perm);

    return perm;
}


pic::matrix createMatrixMultipled(pic::matrix &mtx){
    constexpr int multipleIndex = 1;
    constexpr int multipleNumber = 3;
    pic::matrix mtxMultipled(mtx.rows + (multipleNumber - 1), mtx.cols);

    for (int i = 0; i < multipleIndex; i++){
        for (int j = 0; j < mtx.cols; j++){
            mtxMultipled[i * mtxMultipled.stride + j] =
                mtx[i * mtx.stride + j];
        }
    }
    for (int k = 0; k < multipleNumber; k++){
        for (int j = 0; j < mtx.cols; j++){
            mtxMultipled[(multipleIndex + k) * mtxMultipled.stride + j] =
                mtx[multipleIndex * mtx.stride + j];
        }
    }
    for (int i = multipleIndex + 1; i < mtx.rows; i++){
        for (int j = 0; j < mtx.cols; j++){
            mtxMultipled[(i + multipleNumber - 1) * mtxMultipled.stride + j] =
                mtx[i * mtx.stride + j];
        }
    }

    return mtxMultipled;
}
pic::matrix createMatrixFromIndices(pic::matrix &mtx, int index, int multiplicity){
    pic::matrix newMatrix(mtx.rows - 1, mtx.cols);
    for (int i = 0; i < index; i++){
        if (i == 0){
	    for (int j = 0; j < mtx.cols; j++){
	        newMatrix[j] = mtx[j] + multiplicity * mtx[index * mtx.stride + j];
	    }
	}else{
            for (int j = 0; j < mtx.cols; j++){
	        newMatrix[i * newMatrix.stride + j] = mtx[i * mtx.stride + j];
	    }
	}
    }
    for (int i = index; i < mtx.rows - 1; i++){
        for (int j = 0; j < mtx.cols; j++){
	    newMatrix[i * newMatrix.stride + j] = mtx[(i + 1) * mtx.stride + j];
	}
    }

    return newMatrix;
}
pic::matrix adaptedMatrix(pic::matrix &mtx, int rowIndex, int multiplicity){
    if ( 0 < multiplicity ){

        // Returning a new matrix with modified specific row.
        pic::matrix newMatrix = mtx.copy();

        pic::Complex16 *row = newMatrix.get_data() + rowIndex * newMatrix.stride;
        for (int i = 0; i < mtx.cols; i++){
            row[i] *= multiplicity;
        }
        return newMatrix;
    }else if ( 0 == multiplicity ){
 
        // Returning a new matrix with deleted specific row.
        pic::matrix newMatrix = pic::matrix(mtx.rows-1, mtx.cols);

        for (int i = 0; i < rowIndex; i++){
            memcpy(
                newMatrix.get_data() + i * newMatrix.stride,
                mtx.get_data() + i * mtx.stride,
                mtx.cols * sizeof(pic::Complex16)
            );
        }
        for (int i = rowIndex + 1; i < mtx.rows; i++){
            memcpy(
                newMatrix.get_data() + (i-1) * newMatrix.stride,
                mtx.get_data() + i * mtx.stride,
                mtx.cols * sizeof(pic::Complex16)
            );
        }

        return newMatrix;
    }else{
        return pic::matrix(0,0);
    }

}

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
@param self A pointer pointing to an instance of the class GlynnPermanentCalculator_Wrapper.
@param args A tuple of the input arguments: ??????????????
@param kwds A tuple of keywords
*/
static PyObject *
GlynnPermanentCalculator_Wrapper_calculateDFE(GlynnPermanentCalculator_wrapper *self)
{

    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(self->matrix);
    matrix_mtx = pic::matrix(4,7);
    matrix_mtx[0 * matrix_mtx.stride + 0] = 2;
    matrix_mtx[0 * matrix_mtx.stride + 1] = 1;
    matrix_mtx[0 * matrix_mtx.stride + 2] = 3;
    matrix_mtx[0 * matrix_mtx.stride + 3] = 7;
    matrix_mtx[0 * matrix_mtx.stride + 4] = 9;
    matrix_mtx[0 * matrix_mtx.stride + 5] = 8;
    matrix_mtx[0 * matrix_mtx.stride + 6] = 6;
    
    matrix_mtx[1 * matrix_mtx.stride + 0] = 12;
    matrix_mtx[1 * matrix_mtx.stride + 1] = 11;
    matrix_mtx[1 * matrix_mtx.stride + 2] = 13;
    matrix_mtx[1 * matrix_mtx.stride + 3] = 17;
    matrix_mtx[1 * matrix_mtx.stride + 4] = 19;
    matrix_mtx[1 * matrix_mtx.stride + 5] = 18;
    matrix_mtx[1 * matrix_mtx.stride + 6] = 16;
    
    matrix_mtx[2 * matrix_mtx.stride + 0] = 32;
    matrix_mtx[2 * matrix_mtx.stride + 1] = 31;
    matrix_mtx[2 * matrix_mtx.stride + 2] = 33;
    matrix_mtx[2 * matrix_mtx.stride + 3] = 37;
    matrix_mtx[2 * matrix_mtx.stride + 4] = 39;
    matrix_mtx[2 * matrix_mtx.stride + 5] = 38;
    matrix_mtx[2 * matrix_mtx.stride + 6] = 36;

    matrix_mtx[3 * matrix_mtx.stride + 0] = 72;
    matrix_mtx[3 * matrix_mtx.stride + 1] = 71;
    matrix_mtx[3 * matrix_mtx.stride + 2] = 73;
    matrix_mtx[3 * matrix_mtx.stride + 3] = 77;
    matrix_mtx[3 * matrix_mtx.stride + 4] = 79;
    matrix_mtx[3 * matrix_mtx.stride + 5] = 78;
    matrix_mtx[3 * matrix_mtx.stride + 6] = 76;



    pic::matrix matrixWithMultipledRows = createMatrixMultipled(matrix_mtx);
    pic::matrix matrixRow1Mult1 = adaptedMatrix(matrix_mtx, 1, 1);
    pic::matrix matrixRow1Mult3 = adaptedMatrix(matrix_mtx, 1, 3);
    pic::matrix cmatrixWithMultipledRows = createMatrixMultipled(matrix_mtx);
    pic::matrix cmatrixRow1Mult1 = adaptedMatrix(matrix_mtx, 1, 1);
    pic::matrix cmatrixRow1Mult3 = adaptedMatrix(matrix_mtx, 1, 3);

    std::cout << "matrices:"<<std::endl;
    matrixWithMultipledRows.print_matrix();
    matrixRow1Mult1.print_matrix();
    matrixRow1Mult3.print_matrix();
  
    pic::Complex16 permMatrix = calcPermanenent_DFE(matrixWithMultipledRows);
    pic::Complex16 permMatrixR1M1 = calcPermanenent_DFE(matrixRow1Mult1);
    pic::Complex16 permMatrixR1M3 = calcPermanenent_DFE(matrixRow1Mult3);
    auto &calc = self->calculator;
    pic::Complex16 cpermMatrix = calc->calculate(cmatrixWithMultipledRows);
    pic::Complex16 cpermMatrixR1M1 = calc->calculate(cmatrixRow1Mult1);
    pic::Complex16 cpermMatrixR1M3 = calc->calculate(cmatrixRow1Mult3);


    std::cout << "permMatrix      : " << permMatrix << std::endl;
    std::cout << "permMatrixR1M1  : " << permMatrixR1M1 << std::endl;
    std::cout << "permMatrixR1M3  : " << permMatrixR1M3 << std::endl;
    std::cout << "cpermMatrix     : " << cpermMatrix << std::endl;
    std::cout << "cpermMatrixR1M1 : " << cpermMatrixR1M1 << std::endl;
    std::cout << "cpermMatrixR1M3 : " << cpermMatrixR1M3 << std::endl;

    pic::Complex16 calcPermanent1 =
        (1.0 / 4.0) * (permMatrixR1M3 - 3 * permMatrixR1M1);
    pic::Complex16 calcPermanent2 =
        (1.0 / 4.0) * (cpermMatrixR1M3 - 3 * cpermMatrixR1M1);

    std::cout << "calculated permanent1  : " << calcPermanent1 << std::endl;
    std::cout << "calculated permanent2  : " << calcPermanent2 << std::endl;

    pic::matrix matrixSecondRowMultipled = createMatrixMultipled(matrix_mtx);
    pic::matrix matrixp2 = createMatrixFromIndices(matrix_mtx, 1, 2);
    pic::matrix matrixm2 = createMatrixFromIndices(matrix_mtx, 1, -2);
    pic::matrix matrix0 =  createMatrixFromIndices(matrix_mtx, 1, 0);

    auto copy_matrix_mtx = matrix_mtx.copy();
    auto copy_mSRM = matrixSecondRowMultipled.copy();
    auto copy_mp2 = matrixp2.copy();
    auto copy_mm2 = matrixm2.copy();
    auto copy_m0 = matrix0.copy();

    pic::Complex16 permOrig = calcPermanenent_DFE(matrix_mtx);
    pic::Complex16 perm0 = calcPermanenent_DFE(matrixSecondRowMultipled);
    pic::Complex16 perm1 = calcPermanenent_DFE(matrixp2);
    pic::Complex16 perm2 = calcPermanenent_DFE(matrixm2);
    pic::Complex16 perm3 = calcPermanenent_DFE(matrix0);

    //std::cout << "permOrig : " << permOrig << std::endl;
    //std::cout << "perm0    : " << perm0 << std::endl;
    //std::cout << "perm1    : " << perm1 << std::endl;
    //std::cout << "perm2    : " << perm2 << std::endl;
    //std::cout << "perm3    : " << perm3 << std::endl;

    std::cout << "calc per.: " << 1.0/4.0 * (perm1 + perm2 - 2 * perm3) << std::endl;

    {
      auto &calc = self->calculator;
 
      //std::cout << "matrix_Mtx:\n";
      //matrix_mtx.print_matrix();
      
      pic::Complex16 permOrig = calc->calculate(copy_matrix_mtx);
      pic::Complex16 perm0 = calc->calculate(copy_mSRM);
      pic::Complex16 perm1 = calc->calculate(copy_mp2);
      pic::Complex16 perm2 = calc->calculate(copy_mm2);
      pic::Complex16 perm3 = calc->calculate(copy_m0);

      std::cout << "permOrig : " << permOrig << std::endl;
      std::cout << "perm0    : " << perm0 << std::endl;
      std::cout << "perm1    : " << perm1 << std::endl;
      std::cout << "perm2    : " << perm2 << std::endl;
      std::cout << "perm3    : " << perm3 << std::endl;

      std::cout << "calc per.: " << 1.0/4.0 * (perm1 + perm2 - 2 * perm3) << std::endl;
    }


    pic::Complex16 permanent = permOrig;

    return Py_BuildValue("D", &permanent);
}

static PyObject *
GlynnPermanentCalculator_Wrapper_calculateDFE_multiple_rows(GlynnPermanentCalculator_wrapper *self)
{

    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(self->matrix);


    pic::matrix matrixSecondRowMultipled = createMatrixMultipled(matrix_mtx);
    pic::matrix matrixp2 = createMatrixFromIndices(matrix_mtx, 1, 2);
    pic::matrix matrixm2 = createMatrixFromIndices(matrix_mtx, 1, -2);
    pic::matrix matrix0 =  createMatrixFromIndices(matrix_mtx, 1, 0);

    /*
    std::cout << "Matrix original";
    matrix_mtx.print_matrix();
    std::cout << "matrix second row multipled";
    matrixSecondRowMultipled.print_matrix();
    std::cout << "matrix added 1th row with  2 multiplicity to the 0th row" ;
    matrixp2.print_matrix();
    std::cout << "matrix added 1th row with -2 multiplicity to the 0th row" ;
    matrixm2.print_matrix();
    std::cout << "matrix added 1th row with 0 multiplicity to the 0th row" ;
    matrix0.print_matrix();
    */

    auto copy_matrix_mtx = matrix_mtx.copy();
    auto copy_mSRM = matrixSecondRowMultipled.copy();
    auto copy_mp2 = matrixp2.copy();
    auto copy_mm2 = matrixm2.copy();
    auto copy_m0 = matrix0.copy();

    pic::Complex16 permOrig = calcPermanenent_DFE(matrix_mtx);
    pic::Complex16 perm0 = calcPermanenent_DFE(matrixSecondRowMultipled);
    pic::Complex16 perm1 = calcPermanenent_DFE(matrixp2);
    pic::Complex16 perm2 = calcPermanenent_DFE(matrixm2);
    pic::Complex16 perm3 = calcPermanenent_DFE(matrix0);

    std::cout << "permOrig : " << permOrig << std::endl;
    std::cout << "perm0    : " << perm0 << std::endl;
    std::cout << "perm1    : " << perm1 << std::endl;
    std::cout << "perm2    : " << perm2 << std::endl;
    std::cout << "perm3    : " << perm3 << std::endl;

    std::cout << "calc per.: " << 1.0/4.0 * (perm1 + perm2 - 2 * perm3) << std::endl;

    {
      auto &calc = self->calculator;
 
      //std::cout << "matrix_Mtx:\n";
      //matrix_mtx.print_matrix();
      
      pic::Complex16 permOrig = calc->calculate(copy_matrix_mtx);
      pic::Complex16 perm0 = calc->calculate(copy_mSRM);
      pic::Complex16 perm1 = calc->calculate(copy_mp2);
      pic::Complex16 perm2 = calc->calculate(copy_mm2);
      pic::Complex16 perm3 = calc->calculate(copy_m0);

      std::cout << "permOrig : " << permOrig << std::endl;
      std::cout << "perm0    : " << perm0 << std::endl;
      std::cout << "perm1    : " << perm1 << std::endl;
      std::cout << "perm2    : " << perm2 << std::endl;
      std::cout << "perm3    : " << perm3 << std::endl;

      std::cout << "calc per.: " << 1.0/4.0 * (perm1 + perm2 - 2 * perm3) << std::endl;
    }


    pic::Complex16 permanent = permOrig;

    return Py_BuildValue("D", &permanent);
}






/**
@brief Wrapper function to call the calculate the Permanent on a DFE
@param self A pointer pointing to an instance of the class GlynnPermanentCalculator_Wrapper.
@param args A tuple of the input arguments: ??????????????
@param kwds A tuple of keywords
*/
static PyObject *
GlynnPermanentCalculator_Wrapper_calculateDFEDualCard(GlynnPermanentCalculator_wrapper *self)
{

    // create PIC version of the input matrices
    pic::matrix matrix_mtx = numpy2matrix(self->matrix);


    pic::Complex16* mtx_data = matrix_mtx.get_data();
    

    // calulate the maximal sum of the columns to normalize the matrix
    pic::matrix colSumMax( matrix_mtx.cols, 1);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(pic::Complex16) );
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            pic::Complex16 value1 = colSumMax[jdx] + matrix_mtx[ idx*matrix_mtx.stride + jdx];
            pic::Complex16 value2 = colSumMax[jdx] - matrix_mtx[ idx*matrix_mtx.stride + jdx];
            if ( std::abs( value1 ) < std::abs( value2 ) ) {
                colSumMax[jdx] = value2;
            }
            else {
                colSumMax[jdx] = value1;
            }

        }

    }




    // calculate the renormalization coefficients
    pic::matrix_base<double> renormalize_data(matrix_mtx.cols, 1);
    for (int jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(colSumMax[jdx]);
    }

    // renormalize the input matrix
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            matrix_mtx[ idx*matrix_mtx.stride + jdx] = matrix_mtx[ idx*matrix_mtx.stride + jdx]/renormalize_data[jdx];
        }

    }    


    // SLR and DFE split input matrices
    pic::matrix mtx_split[8];
    pic::Complex16* mtx_data_split[8];


    size_t max_fpga_rows =  MAX_FPGA_DIM;
    size_t max_fpga_cols =  MAX_FPGA_DIM/8;

    // SLR splitted data for the first DFE card
    size_t cols_half1_tot = matrix_mtx.cols/2;
    size_t cols_half2_tot = matrix_mtx.cols - cols_half1_tot;

    size_t rows = matrix_mtx.rows;
    size_t cols_half1[4];
    cols_half1[0] = max_fpga_cols < cols_half1_tot ? max_fpga_cols : cols_half1_tot;
    cols_half1[1] = max_fpga_cols < (cols_half1_tot -cols_half1[0]) ? max_fpga_cols : (cols_half1_tot-cols_half1[0]);
    cols_half1[2] = max_fpga_cols < (cols_half1_tot - cols_half1[0] - cols_half1[1]) ? max_fpga_cols : (cols_half1_tot - cols_half1[0] - cols_half1[1]);
    cols_half1[3] = max_fpga_cols < (cols_half1_tot - cols_half1[0] - cols_half1[1] - cols_half1[2]) ? max_fpga_cols : (cols_half1_tot - cols_half1[0] - cols_half1[1] - cols_half1[2]);


    size_t col_offset = 0;
    for (int kdx=0; kdx<4; kdx++) {

        mtx_split[kdx] = pic::matrix(max_fpga_rows, max_fpga_cols);
        mtx_data_split[kdx] = mtx_split[kdx].get_data();

        pic::Complex16 padding_element(1.0,0.0);
        for (size_t idx=0; idx<rows; idx++) {
            size_t offset = idx*matrix_mtx.stride+col_offset;
            size_t offset_small = idx*mtx_split[kdx].stride;
            for (size_t jdx=0; jdx<cols_half1[kdx]; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = matrix_mtx[offset+jdx];
            }

            for (size_t jdx=cols_half1[kdx]; jdx<max_fpga_cols; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = padding_element;
            }
            padding_element.real(0.0);
        }
        col_offset = col_offset + cols_half1[kdx];

        memset( mtx_data_split[kdx] + rows*mtx_split[kdx].stride, 0.0, (max_fpga_rows-rows)*max_fpga_cols*sizeof(pic::Complex16));
    }


    // SLR splitted data for the second DFE card
    size_t cols_half2[4];
    cols_half2[0] = max_fpga_cols < cols_half2_tot ? max_fpga_cols : cols_half2_tot;
    cols_half2[1] = max_fpga_cols < (cols_half2_tot - cols_half2[0]) ? max_fpga_cols : (cols_half2_tot - cols_half2[0]);
    cols_half2[2] = max_fpga_cols < (cols_half2_tot - cols_half2[0] - cols_half2[1]) ? max_fpga_cols : (cols_half2_tot - cols_half2[0] - cols_half2[1]);
    cols_half2[3] = max_fpga_cols < (cols_half2_tot - cols_half2[0] - cols_half2[1] - cols_half2[2]) ? max_fpga_cols : (cols_half2_tot - cols_half2[0] - cols_half2[1] - cols_half2[2]);

    for (int kdx=0; kdx<4; kdx++) {

        mtx_split[kdx+4] = pic::matrix(max_fpga_rows, max_fpga_cols);
        mtx_data_split[kdx+4] = mtx_split[kdx+4].get_data();

        pic::Complex16 padding_element(1.0,0.0);
        for (size_t idx=0; idx<rows; idx++) {
            size_t offset = idx*matrix_mtx.stride+col_offset;
            size_t offset_small = idx*mtx_split[kdx+4].stride;
            for (size_t jdx=0; jdx<cols_half2[kdx]; jdx++) {
                mtx_data_split[kdx+4][offset_small+jdx] = matrix_mtx[offset+jdx];
            }

            for (size_t jdx=cols_half2[kdx]; jdx<max_fpga_cols; jdx++) {
                mtx_data_split[kdx+4][offset_small+jdx] = padding_element;
            }
            padding_element.real(0.0);

        }
        col_offset = col_offset + cols_half2[kdx];
        memset( mtx_data_split[kdx+4] + rows*mtx_split[kdx+4].stride, 0.0, (max_fpga_rows-rows)*max_fpga_cols*sizeof(pic::Complex16));
    }


/*
matrix_mtx.print_matrix();
for (int idx=0; idx<8; idx++) {
   mtx_split[idx].print_matrix();
}
*/

    pic::Complex16 perm;
    calcPermanentGlynnDFEDualCard( mtx_data_split, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, &perm);





    return Py_BuildValue("D", &perm);
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
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
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
    {"calculateDFE", (PyCFunction) GlynnPermanentCalculator_Wrapper_calculateDFE, METH_NOARGS,
     "Method to calculate the permanent on the DFE."
    },
    {"calculateDFE_multiple_rows", (PyCFunction) GlynnPermanentCalculator_Wrapper_calculateDFE_multiple_rows, METH_NOARGS,
     "Method to calculate the permanent on the DFE."
    },
    {"calculateDFEDualCard", (PyCFunction) GlynnPermanentCalculator_Wrapper_calculateDFEDualCard, METH_NOARGS,
     "Method to calculate the permanent on dual DFE."
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
