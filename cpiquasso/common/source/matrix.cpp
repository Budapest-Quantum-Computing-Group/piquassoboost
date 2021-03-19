#include "matrix.h"
#include <cstring>
#include <iostream>

/// The namespace of the Picasso project
namespace pic {





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
