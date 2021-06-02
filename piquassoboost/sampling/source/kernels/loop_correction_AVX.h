#ifndef loop_correction_AVX_H
#define loop_correction_AVX_H

#include "matrix.h"


namespace pic {

/**
@brief AVX kernel to calculate the loop corrections in Eq (3.26) of arXiv1805.12498 (The input matrix and vectors are Hessenberg transformed)
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
matrix calculate_loop_correction_AVX( matrix &cx_diag_elements, matrix &diag_elements, matrix& AZ, size_t num_of_modes);



} // PIC

#endif


