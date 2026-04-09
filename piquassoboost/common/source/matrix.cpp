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

#include "matrix.h"
#include <cstring>
#include <algorithm>
#include <iostream>

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
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@param stride_in The column stride of the matrix array (The array elements in one row are a_0, a_1, ... a_{cols-1}, 0, 0, 0, 0. The number of zeros is stride-cols)
@return Returns with the instance of the class.
*/
matrix::matrix( Complex16* data_in, size_t rows_in, size_t cols_in, size_t stride_in) : matrix_base<Complex16>(data_in, rows_in, cols_in, stride_in) {

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
@brief Constructor of the class. Allocates data for matrix rows_in times cols_in. By default the created instance would be the owner of the stored data.
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@param stride_in The column stride of the matrix array (The array elements in one row are a_0, a_1, ... a_{cols-1}, 0, 0, 0, 0. The number of zeros is stride-cols)
@return Returns with the instance of the class.
*/
matrix::matrix( size_t rows_in, size_t cols_in, size_t stride_in) : matrix_base<Complex16>(rows_in, cols_in, stride_in) {

}

/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input matrix. (Needed for TBB calls)
@param An instance of class matrix to be copied.
*/
matrix::matrix(const matrix &in) : matrix_base<Complex16>(in) {

}

/**
@brief Copy assignment operator of the class. Shares stored memory with reference counting.
*/
matrix&
matrix::operator=(const matrix &in) {
  matrix_base<Complex16>::operator=(in);
  return *this;
}



/**
@brief Call to create a copy of the matrix instance.
@return Returns with the instance of the class.
*/
matrix
matrix::copy() {

  matrix ret = matrix(rows, cols, stride);

  // logical variable indicating whether the matrix needs to be conjugated in CBLAS operations
  ret.conjugated = conjugated;
  // logical variable indicating whether the matrix needs to be transposed in CBLAS operations
  ret.transposed = transposed;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  ret.owner = true;

  std::copy_n(data, rows*cols, ret.data);

  return ret;

}



/**
@brief Move constructor of the class. Takes ownership of the moved matrix's data.
*/
matrix::matrix(matrix &&in) noexcept 
    : matrix_base<Complex16>() {
  // Drop default-constructed bookkeeping and steal complete storage state.
  release_data();

  data = in.data;
  rows = in.rows;
  cols = in.cols;
  stride = in.stride;
  owner = in.owner;
  conjugated = in.conjugated;
  transposed = in.transposed;
  reference_mutex = in.reference_mutex;
  references = in.references;

  in.data = NULL;
  in.rows = 0;
  in.cols = 0;
  in.stride = 0;
  in.owner = false;
  in.conjugated = false;
  in.transposed = false;
  in.reference_mutex = NULL;
  in.references = NULL;
}

/**
@brief Move assignment operator. Takes ownership of the moved matrix's data.
*/
matrix& matrix::operator=(matrix &&in) noexcept {
  if (this != &in) {
    release_data();

    data = in.data;
    rows = in.rows;
    cols = in.cols;
    stride = in.stride;
    owner = in.owner;
    conjugated = in.conjugated;
    transposed = in.transposed;
    reference_mutex = in.reference_mutex;
    references = in.references;

    in.data = NULL;
    in.rows = 0;
    in.cols = 0;
    in.stride = 0;
    in.owner = false;
    in.conjugated = false;
    in.transposed = false;
    in.reference_mutex = NULL;
    in.references = NULL;
  }
  return *this;
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
