#ifndef TorontonianUtilities_H
#define TorontonianUtilities_H

#ifndef LAPACK_ROW_MAJOR
#define LAPACK_ROW_MAJOR 101
#endif
#ifndef LAPACK_COL_MAJOR
#define LAPACK_COL_MAJOR 102
#endif


#include "matrix32.h"



extern "C" {

#define LAPACK_ROW_MAJOR               101

/// Definition of the LAPACKE_zpotrf function from LAPACKE to calculate the Cholesky decomposition of a matrix
int LAPACKE_zpotrf( int matrix_layout, char UPLO, int n, pic::Complex16* mtx, int lda );
}


namespace pic {

/**
@brief Call to calculate in-place Cholesky decomposition of a matrix. The decomposed matrix is stored in mtx.
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.
*/
void calc_cholesky_decomposition(matrix32& mtx);

/**
@brief Call to calculate in-place Cholesky decomposition of a matrix
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@param reuse_index Labels the row and column from which the Cholesky decomposition should be continued.
*/
void calc_cholesky_decomposition(matrix32& mtx, const size_t reuse_index);

/**
@brief Call to calculate in-place Cholesky decomposition of a matrix
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@param reuse_index Labels the row and column from which the Cholesky decomposition should be continued.
@param determinant The determinant of the matrix is calculated and stored in this variable.
(if reuse_index index is greater than 0, than the contributions of the first reuse_index-1 elements of the Cholesky L matrix should be multiplied manually)
*/
void calc_cholesky_decomposition(matrix32& mtx, const size_t reuse_index, Complex32 &determinant);


/**
@brief Call to calculate in-place Cholesky decomposition of a matrix using the Lapack implementation
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
*/
void calc_cholesky_decomposition_lapack(matrix32 &matrix);


/**
@brief Call to calculate the determinant of a matrix by Cholesky decomposition.
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@return Returns with the calculated determiant
*/
Complex32 calc_determinant_cholesky_decomposition(matrix32& mtx);


/**
@brief Call to calculate the determinant of a matrix by Cholesky decomposition.
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@param L A partial Cholesky decomposition of the matrix mtx. The first (reuse_index-1) diagonal elements are used to calculate the determinant.
@param reuse_index Labels the row and column from which the Cholesky decomposition should be continued.
@return Returns with the calculated determiant
*/
Complex32 calc_determinant_cholesky_decomposition(matrix32& mtx, const size_t reuse_index);


} // PIC

#endif
