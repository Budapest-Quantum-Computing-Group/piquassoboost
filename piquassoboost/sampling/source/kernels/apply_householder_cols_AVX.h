#ifndef apply_householder_cols_AVX_H
#define apply_householder_cols_AVX_H

#include "matrix.h"


namespace pic {

/**
@brief AVX kernel to apply householder transformation on a matrix A' = (1 - 2*v o v/v^2) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
*/
void
apply_householder_cols_AVX(matrix &A, matrix &v);


} //PIC

#endif
