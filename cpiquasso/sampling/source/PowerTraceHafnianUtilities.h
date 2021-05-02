#ifndef PowerTraceHafnianUtilities_H
#define PowerTraceHafnianUtilities_H

#include "matrix.h"
#include "matrix32.h"
#include "dot.h"


extern "C" {

#define LAPACK_ROW_MAJOR               101

/// Definition of the LAPACKE_zgehrd function from LAPACKE to calculate the upper triangle hessenberg transformation of a matrix
int LAPACKE_zgehrd( int matrix_layout, int n, int ilo, int ihi, pic::Complex16* a, int lda, pic::Complex16* tau );

}


namespace pic {


/**
/@brief Determine the reflection vector for Householder transformation used in the upper Hessenberg transformation algorithm
@param input The strided input vector constructed from the k-th column of the matrix on which the Hessenberg transformation should be applied
@param norm_v_sqr The squared norm of the created reflection matrix that is returned by reference
@return Returns with the calculated reflection vector
 */
matrix get_reflection_vector(matrix &input, double &norm_v_sqr);



/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v A for one specific reflection vector v
@param A matrix on which the householder transformation is applied.
@param v A matrix instance of the reflection vector
@param vH_times_A preallocated array to hold the data of vH_times_A. The result is returned via this reference.
*/
void calc_vH_times_A(matrix &A, matrix &v, matrix &vH_times_A);


/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
@param vH_times_A The calculated product v^H * A calculated by calc_vH_times_A.
*/
void calc_vov_times_A(matrix &A, matrix &v, matrix &vH_times_A);

/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v/v^2) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
*/
void apply_householder_rows(matrix &A, matrix &v);




/**
@brief Apply householder transformation on a matrix A' = A(1 - 2*v o v) for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
*/
void apply_householder_cols_req(matrix &A, matrix &v);





/**
@brief Reduce a general matrix to upper Hessenberg form.
@param matrix matrix to be reduced to upper Hessenberg form. The reduced matrix is returned via this input
*/
void transform_matrix_to_hessenberg(matrix &mtx);



/**
@brief Reduce a general matrix to upper Hessenberg form and applies the unitary transformation on left/right sided vectors to keep the \f$ <L|M|R> \f$ product invariant.
@param matrix matrix to be reduced to upper Hessenberg form. The reduced matrix is returned via this input
@param Lv the left sided vector
@param Rv the roght sided vector
*/
void transform_matrix_to_hessenberg(matrix &mtx, matrix& Lv, matrix& Rv );




void CalcPowerTracesAndLoopCorrections( matrix &cx_diag_elements, matrix &diag_elements, matrix& AZ, size_t pow_max, matrix32 &traces, matrix32 &loop_corrections);


/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
matrix calculate_loop_correction_2( matrix &cx_diag_elements, matrix &diag_elements, matrix& AZ, size_t num_of_modes);



/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
matrix32 CalculateLoopCorrectionWithHessenberg( matrix &cx_diag_elements, matrix& diag_elements, matrix& AZ, size_t dim_over_2);



} // PIC

#endif
