#ifndef calc_vH_times_A_AVX_H
#define calc_vH_times_A_AVX_H

#include "matrix.h"


namespace pic {

/**
@brief AVX kernel to apply householder transformation on a matrix A' = (1 - 2*v o v A for one specific reflection vector v.
@param A matrix on which the householder transformation is applied.
@param v A matrix instance of the reflection vector
@param vH_times_A preallocated array to hold the data of vH_times_A. The result is returned via this reference.
*/
void
calc_vH_times_A_AVX(matrix &A, matrix &v, matrix &vH_times_A);


} //PIC

#endif
