#ifndef calc_cholesky_decomposition_AVX_H
#define calc_cholesky_decomposition_AVX_H

#include "matrix.h"


namespace pic {

/**
*/
void
calc_cholesky_decomposition_AVX(matrix &A, const size_t reuse_index);


} //PIC

#endif