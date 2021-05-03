#ifndef calc_characteristic_polynomial_coeffs_AVX_H
#define calc_characteristic_polynomial_coeffs_AVX_H

#include "matrix.h"


namespace pic {

/**
@brief AVX kernel to determine the first \f$ k \f$ coefficients of the characteristic polynomial using the Algorithm 2 of LaBudde method.
 See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1) for further details.
@param mtx matrix in upper Hessenberg form.
@param highest_order the order of the highest order coefficient to be calculated (k <= n)
@return Returns with the calculated coefficients of the characteristic polynomial.
 */
matrix calc_characteristic_polynomial_coeffs_AVX(matrix &mtx, size_t highest_order);


} //PIC

#endif
