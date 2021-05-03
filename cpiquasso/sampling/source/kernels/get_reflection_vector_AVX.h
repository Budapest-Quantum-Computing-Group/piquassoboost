#ifndef get_reflection_vector_AVX_H
#define get_reflection_vector_AVX_H

#include "matrix.h"


namespace pic {

/**
/@brief Determine the reflection vector for Householder transformation used in the upper Hessenberg transformation algorithm
@param input The strided input vector constructed from the k-th column of the matrix on which the Hessenberg transformation should be applied
@param norm_v_sqr The squared norm of the created reflection matrix that is returned by reference
@return Returns with the calculated reflection vector
 */
matrix
get_reflection_vector_AVX(matrix &input, double &norm_v_sqr);


} //PIC

#endif
