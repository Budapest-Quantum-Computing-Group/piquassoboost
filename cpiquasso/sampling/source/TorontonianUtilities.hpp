#include "TorontonianUtilities.h"
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
@brief Call to calculate in-place Cholesky decomposition of a matrix. The decomposed matrix is stored in mtx.
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.
*/
/*
template<class matrix_type, class complex_type>
void
calc_cholesky_decomposition(matrix_type& matrix)
{
    Complex32 determinant(0.0,0.0);
    calc_cholesky_decomposition<matrix_type, complex_type>( matrix, 0, determinant);
    return;
}
*/

/**
@brief Call to calculate in-place Cholesky decomposition of a matrix
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@param reuse_index Labels the row and column from which the Cholesky decomposition should be continued.
*/
/*
template<class matrix_type, class complex_type>
void
calc_cholesky_decomposition(matrix_type& matrix, const size_t reuse_index)
{
    Complex32 determinant(0.0,0.0);
    calc_cholesky_decomposition<matrix_type, complex_type>( matrix, reuse_index, determinant);
    return;
}
*/

/**
@brief Call to calculate in-place Cholesky decomposition of a matrix
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@param reuse_index Labels the row and column from which the Cholesky decomposition should be continued.
@param determinant The determinant of the matrix is calculated and stored in this variable.
(if reuse_index index is greater than 0, than the contributions of the first reuse_index-1 elements of the Cholesky L matrix should be multiplied manually)
*/
template<class matrix_type, class complex_type>
void
calc_cholesky_decomposition(matrix_type& mtx, const size_t &reuse_index, Complex32 &determinant)
{

    determinant = Complex32(1.0,0.0);

    // storing in the same memory the results of the algorithm
    size_t n = mtx.cols;
    // Decomposing a matrix into lower triangular matrices
    for (size_t i = reuse_index; i < n; i++) {
        complex_type* row_i = mtx.get_data()+i*mtx.stride;

        for (size_t j = reuse_index; j < i; j++) {
            {
                complex_type* row_j = mtx.get_data()+j*mtx.stride;

                complex_type sum = 0;
                // Evaluating L(i, j) using L(j, j)
                // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
                for (size_t k = 0; k < j; k++){
                    sum += mult_a_bconj( row_i[k], row_j[k]);
                }

                row_i[j] = (row_i[j] - sum) / row_j[j];
            }
        }
        complex_type sum = 0;
        // summation for diagnols
        // L_{j,j}=\sqrt{A_{j,j}-\sum_{k=0}^{j-1}L_{j,k}L_{j,k}^*}
        for (size_t k = 0; k < i; k++){
            sum += mult_a_bconj( row_i[k], row_i[k] );
        }
        row_i[i] = sqrt(row_i[i] - sum);
        determinant = determinant * row_i[i];
    }

    return;


};



/**
@brief Call to calculate the determinant of a matrix by Cholesky decomposition.
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@return Returns with the calculated determiant
*/
template<class matrix_type, class complex_type>
Complex32
calc_determinant_cholesky_decomposition(matrix_type& mtx){

 return  calc_determinant_cholesky_decomposition(mtx, 0);

}


/**
@brief Call to calculate the determinant of a matrix by Cholesky decomposition.
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@param L A partial Cholesky decomposition of the matrix mtx. The first (reuse_index-1) diagonal elements are used to calculate the determinant.
@param reuse_index Labels the row and column from which the Cholesky decomposition should be continued.
@return Returns with the calculated determiant
*/
template<class matrix_type, class complex_type>
Complex32
calc_determinant_cholesky_decomposition(matrix_type& mtx, const size_t reuse_index){

        Complex32 determinant(1.0, 0.0);

        // calculate the rest of the Cholesky decomposition and calculate the determinant
        calc_cholesky_decomposition<matrix_type, complex_type>(mtx, reuse_index, determinant);

#ifdef DEBUG
        if (reuse_index > mtx.rows ) {
            std::cout << "calc_determinant_cholesky_decomposition: reuse index should be smaller than the matrix size!" << std::endl;
            exit(-1);
        }

#endif

        // multiply the result with the remaining diagonal elements of the Cholesky matrix L, that has been reused

        for (size_t idx=0; idx < reuse_index; idx++){
                determinant *= mtx[idx * mtx.stride + idx];
        }


        return mult_a_bconj( determinant, determinant);

}





} // PIC


