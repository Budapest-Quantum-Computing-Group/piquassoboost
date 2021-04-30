#include "PowerTraceHafnianUtilities.hpp"



namespace pic {



/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
matrix
calculate_loop_correction_2( matrix &cx_diag_elements, matrix &diag_elements, matrix& AZ, size_t num_of_modes) {


#ifdef USE_AVX
    matrix loop_correction(num_of_modes, 1);
    transform_matrix_to_hessenberg<matrix, Complex16>(AZ, diag_elements, cx_diag_elements);

#include "kernels/loop_correction_AVX.S"

    return loop_correction;

#else
    return calculate_loop_correction_2<matrix, Complex16>( cx_diag_elements, diag_elements, AZ, num_of_modes);
#endif




}



/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
matrix32
CalculateLoopCorrectionWithHessenberg( matrix &cx_diag_elements, matrix& diag_elements, matrix& AZ, size_t dim_over_2) {


/*

    if (AZ.rows < 30) {
*/
        // for smaller matrices first calculate the corerction in 16 byte precision, than convert the result to 32 byte precision
        matrix &&loop_correction = calculate_loop_correction_2(cx_diag_elements, diag_elements, AZ, dim_over_2);

        matrix32 loop_correction32(dim_over_2, 1);
        for (size_t idx=0; idx<loop_correction.size(); idx++ ) {
            loop_correction32[idx].real( loop_correction[idx].real() );
            loop_correction32[idx].imag( loop_correction[idx].imag() );
        }

        return loop_correction32;
/*
    }
    else{

        // for smaller matrices first convert the input matrices to 32 byte precision, than calculate the diag correction
        matrix_type diag_elements32( diag_elements.rows, diag_elements.cols);
        matrix_type cx_diag_elements32( cx_diag_elements.rows, cx_diag_elements.cols);
        for (size_t idx=0; idx<diag_elements32.size(); idx++) {
            diag_elements32[idx].real( diag_elements[idx].real() );
            diag_elements32[idx].imag( diag_elements[idx].imag() );

            cx_diag_elements32[idx].real( cx_diag_elements[idx].real() );
            cx_diag_elements32[idx].imag( cx_diag_elements[idx].imag() );
        }

        matrix_type AZ_32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ_32[idx].real( AZ[idx].real() );
            AZ_32[idx].imag( AZ[idx].imag() );
        }

        return calculate_loop_correction_2<matrix_type, complex_type>(cx_diag_elements32, diag_elements32, AZ_32, dim_over_2);


    }
*/

}



} // PIC

