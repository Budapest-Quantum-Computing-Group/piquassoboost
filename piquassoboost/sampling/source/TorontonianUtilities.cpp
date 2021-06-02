#include "TorontonianUtilities.hpp"
#include "calc_cholesky_decomposition_AVX.h"
#include <iostream>
#include "common_functionalities.h"
#include <math.h>

/*
static tbb::spin_mutex my_mutex;

double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {




/**
@brief Call to calculate in-place Cholesky decomposition of a matrix
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@param reuse_index Labels the row and column from which the Cholesky decomposition should be continued.
@param determinant The determinant of the matrix is calculated and stored in this variable.
(if reuse_index index is greater than 0, than the contributions of the first reuse_index-1 elements of the Cholesky L matrix should be multiplied manually)
*/
void
calc_cholesky_decomposition(matrix32& mtx, const size_t reuse_index, Complex32 &determinant)
{

    calc_cholesky_decomposition<matrix32, Complex32>(mtx, reuse_index, determinant);
    return;

}



/**
@brief Call to calculate in-place Cholesky decomposition of a matrix
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@param reuse_index Labels the row and column from which the Cholesky decomposition should be continued.
@param determinant The determinant of the matrix is calculated and stored in this variable.
(if reuse_index index is greater than 0, than the contributions of the first reuse_index-1 elements of the Cholesky L matrix should be multiplied manually)
*/
void
calc_cholesky_decomposition(matrix& mtx, const size_t reuse_index, Complex32 &determinant)
{

#ifdef USE_AVX

    calc_cholesky_decomposition_AVX(mtx, reuse_index, determinant);
    return;

#else

    calc_cholesky_decomposition<matrix, Complex16>(mtx, reuse_index, determinant);
    return;

#endif // USE_AVX


}


/**
@brief Call to calculate in-place Cholesky decomposition of a matrix using the Lapack implementation
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
*/
void
calc_cholesky_decomposition_lapack(matrix &matrix) {


// transform the matrix mtx into an upper Hessenberg format by calling lapack function
        char UPLO = 'L';
        int N = matrix.rows;
        int LDA = matrix.stride;

        //std::cout<<"Before lapacke call:\n";
        //mtx.print_matrix();


        LAPACKE_zpotrf(LAPACK_ROW_MAJOR, UPLO, N, matrix.get_data(), LDA);

        return;

}






} // PIC


