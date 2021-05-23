#ifndef calc_cholesky_decomposition_AVX_H
#define calc_cholesky_decomposition_AVX_H

#include "matrix.h"


namespace pic {

/**
@brief AVX kernel to calculate in-place Cholesky decomposition of a matrix
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@param reuse_index Labels the row and column from which the Cholesky decomposition should be continued.
@param determinant The determinant of the matrix is calculated and stored in this variable.
(if reuse_index index is greater than 0, than the contributions of the first reuse_index-1 elements of the Cholesky L matrix should be multiplied manually)
*/
void calc_cholesky_decomposition_AVX(matrix &A, const size_t reuse_index, Complex32 &determinant);


} //PIC

#endif
