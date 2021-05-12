#ifndef TorontonianUtilities_H
#define TorontonianUtilities_H

#ifndef LAPACK_ROW_MAJOR
#define LAPACK_ROW_MAJOR 101
#endif
#ifndef LAPACK_COL_MAJOR
#define LAPACK_COL_MAJOR 102
#endif


#include "matrix.h"
#include "matrix32.h"


extern "C" {

#define LAPACK_ROW_MAJOR               101

/// Definition of the LAPACKE_zpotrf function from LAPACKE to calculate the Cholesky decomposition of a matrix
int LAPACKE_zpotrf( int matrix_layout, char UPLO, int n, pic::Complex16* mtx, int lda );
}


namespace pic {

// Cholesky decomposition
// Works for selfadjoint positive definite matrices!
// Basic version: no block matrices used.
void
calc_cholesky_decomposition(matrix& matrix, const size_t reuse_index);



void
calc_cholesky_decomposition_lapack(matrix &matrix);


// calculating determinant based on cholesky decomposition
Complex16
calc_determinant_cholesky_decomposition(matrix& mtx, const size_t reuse_index);


} // PIC

#endif
