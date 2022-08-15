/**
 * Copyright 2021 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PowerTraceHafnianUtilities_H
#define PowerTraceHafnianUtilities_H

#ifndef LAPACK_ROW_MAJOR
#define LAPACK_ROW_MAJOR 101
#endif
#ifndef LAPACK_COL_MAJOR
#define LAPACK_COL_MAJOR 102
#endif


#include "matrix.h"
#include "matrix32.h"

#include "dot.h"
#include "InfinitePrecisionComplex.h"

extern "C" {

#define LAPACK_ROW_MAJOR               101

/// Definition of the LAPACKE_zgehrd function from LAPACKE to calculate the upper triangle hessenberg transformation of a matrix
int LAPACKE_zgehrd( int matrix_layout, int n, int ilo, int ihi, pic::Complex16* a, int lda, pic::Complex16* tau );

}


namespace pic {

template <class T> class complex_selector;
template <> class complex_selector<double> {
public:
    typedef Complex16 cplx_type;
};
template <> class complex_selector<long double> {
public:
    typedef Complex32 cplx_type;
};
template <> class complex_selector<FloatInf> {
public:
    typedef ComplexInf cplx_type;
};
template <> class complex_selector<RationalInf> {
public:
    typedef ComplexRationalInf cplx_type;
};
template <class T>
using cplx_select_t = typename complex_selector<T>::cplx_type;

template <class T> class matrix_selector;
template <> class matrix_selector<Complex16> {
public:
    typedef matrix mat_type;
};
template <> class matrix_selector<Complex32> {
public:
    typedef matrix32 mat_type;
};
template <> class matrix_selector<ComplexInf> {
public:
    typedef matrix_base<ComplexInf> mat_type;
}; 
template <> class matrix_selector<ComplexRationalInf> {
public:
    typedef matrix_base<ComplexRationalInf> mat_type;
}; 
template <class T>
using mtx_select_t = typename matrix_selector<T>::mat_type;



/**
/@brief Determine the reflection vector for Householder transformation used in the upper Hessenberg transformation algorithm
@param input The strided input vector constructed from the k-th column of the matrix on which the Hessenberg transformation should be applied
@param norm_v_sqr The squared norm of the created reflection matrix that is returned by reference
@return Returns with the calculated reflection vector
 */
template <typename small_scalar_type>
mtx_select_t<cplx_select_t<small_scalar_type>>
get_reflection_vector(mtx_select_t<cplx_select_t<small_scalar_type>> &input, small_scalar_type &norm_v_sqr);



/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v A for one specific reflection vector v
@param A matrix on which the householder transformation is applied.
@param v A matrix instance of the reflection vector
@param vH_times_A preallocated array to hold the data of vH_times_A. The result is returned via this reference.
*/
template<class small_scalar_type>
void
calc_vH_times_A(mtx_select_t<cplx_select_t<small_scalar_type>> &A, mtx_select_t<cplx_select_t<small_scalar_type>> &v, mtx_select_t<cplx_select_t<small_scalar_type>> &vH_times_A);


/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
@param vH_times_A The calculated product v^H * A calculated by calc_vH_times_A.
*/
template<class small_scalar_type>
void
calc_vov_times_A(mtx_select_t<cplx_select_t<small_scalar_type>> &A, mtx_select_t<cplx_select_t<small_scalar_type>> &v, mtx_select_t<cplx_select_t<small_scalar_type>> &vH_times_A);


/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v/v^2) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
*/
template <class small_scalar_type>
void
apply_householder_rows(mtx_select_t<cplx_select_t<small_scalar_type>> &A, mtx_select_t<cplx_select_t<small_scalar_type>> &v);





/**
@brief Apply householder transformation on a matrix A' = A(1 - 2*v o v) for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
*/
template<class small_scalar_type>
void
apply_householder_cols_req(mtx_select_t<cplx_select_t<small_scalar_type>> &A, mtx_select_t<cplx_select_t<small_scalar_type>> &v);





/**
@brief Reduce a general matrix to upper Hessenberg form.
@param matrix matrix to be reduced to upper Hessenberg form. The reduced matrix is returned via this input
*/
template <class small_scalar_type>
void
transform_matrix_to_hessenberg(mtx_select_t<cplx_select_t<small_scalar_type>> &mtx);



/**
@brief Reduce a general matrix to upper Hessenberg form and applies the unitary transformation on left/right sided vectors to keep the \f$ <L|M|R> \f$ product invariant.
@param matrix matrix to be reduced to upper Hessenberg form. The reduced matrix is returned via this input
@param Lv the left sided vector
@param Rv the roght sided vector
*/
template <class small_scalar_type>
void
transform_matrix_to_hessenberg(mtx_select_t<cplx_select_t<small_scalar_type>> &mtx, mtx_select_t<cplx_select_t<small_scalar_type>>& Lv, mtx_select_t<cplx_select_t<small_scalar_type>>& Rv );



/**
@brief Call to calculate the power traces \f$Tr(mtx^j)~\forall~1\leq j\leq l\f$ for a squared complex matrix \f$mtx\f$ of dimensions \f$n\times n\f$
and a loop corrections in Eq (3.26) of arXiv1805.12498
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@param pow_max
@return Returns with the calculated loop correction
*/
template <class small_scalar_type, class scalar_type>
void
CalcPowerTraces( mtx_select_t<cplx_select_t<small_scalar_type>>& AZ, size_t pow_max, mtx_select_t<cplx_select_t<scalar_type>> &traces32);


/**
@brief Call to calculate the power traces \f$Tr(mtx^j)~\forall~1\leq j\leq l\f$ for a squared complex matrix \f$mtx\f$ of dimensions \f$n\times n\f$
and a loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
template <class small_scalar_type, class scalar_type>
void
CalcPowerTracesAndLoopCorrections( mtx_select_t<cplx_select_t<small_scalar_type>> &cx_diag_elements, mtx_select_t<cplx_select_t<small_scalar_type>> &diag_elements, mtx_select_t<cplx_select_t<small_scalar_type>>& AZ, size_t pow_max, mtx_select_t<cplx_select_t<scalar_type>> &traces32, mtx_select_t<cplx_select_t<scalar_type>> &loop_corrections32);

/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
template <class small_scalar_type>
mtx_select_t<cplx_select_t<small_scalar_type>>
calculate_loop_correction_2( mtx_select_t<cplx_select_t<small_scalar_type>> &cx_diag_elements, mtx_select_t<cplx_select_t<small_scalar_type>> &diag_elements, mtx_select_t<cplx_select_t<small_scalar_type>>& AZ, size_t num_of_modes);




} // PIC

#endif
