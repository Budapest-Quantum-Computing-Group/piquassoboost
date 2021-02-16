#include "matrix.h"
#include <cstring>
#include <iostream>
#include "tbb/tbb.h"

/// The namespace of the Picasso project
namespace pic {

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
matrix::matrix() : matrix_base<Complex16>() {

}

/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
matrix::matrix( Complex16* data_in, size_t rows_in, size_t cols_in) : matrix_base<Complex16>(data_in, rows_in, cols_in) {

}


/**
@brief Constructor of the class. Allocates data for matrix rows_in times cols_in. By default the created instance would be the owner of the stored data.
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
matrix::matrix( size_t rows_in, size_t cols_in) : matrix_base<Complex16>(rows_in, cols_in) {


}


/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input matrix. (Needed for TBB calls)
@param An instance of class matrix to be copied.
*/
matrix::matrix(const matrix &in) : matrix_base<Complex16>(in)  {

}



/**
@brief Call to create a copy of the matrix
@return Returns with the instance of the class.
*/
matrix
matrix::copy() {

  matrix ret = matrix(rows, cols);

  // logical variable indicating whether the matrix needs to be conjugated in CBLAS operations
  ret.conjugated = conjugated;
  // logical variable indicating whether the matrix needs to be transposed in CBLAS operations
  ret.transposed = transposed;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  ret.owner = true;

  memcpy( ret.data, data, rows*cols*sizeof(Complex16));

  return ret;

}



/**
@brief Call to check the array for NaN entries.
@return Returns with true if the array has at least one NaN entry.
*/
bool
matrix::isnan() {

    for (size_t idx=0; idx < rows*cols; idx++) {
        if ( std::isnan(data[idx].real()) || std::isnan(data[idx].imag()) ) {
            return true;
        }
    }

    return false;


}




} //PIC
