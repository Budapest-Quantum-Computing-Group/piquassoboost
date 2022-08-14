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

#include "PowerTraceHafnianUtilities.hpp"
#include "get_reflection_vector_AVX.h"
#include "calc_vH_times_A_AVX.h"
#include "calc_vov_times_A_AVX.h"
#include "apply_householder_cols_AVX.h"
#include "loop_correction_AVX.h"




namespace pic {

/**
/@brief Determine the reflection vector for Householder transformation used in the upper Hessenberg transformation algorithm
@param input The strided input vector constructed from the k-th column of the matrix on which the Hessenberg transformation should be applied
@param norm_v_sqr The squared norm of the created reflection matrix that is returned by reference
@return Returns with the calculated reflection vector
 */
template <typename small_scalar_type>
mtx_select_t<cplx_select_t<small_scalar_type>>
get_reflection_vector(mtx_select_t<cplx_select_t<small_scalar_type>> &input, small_scalar_type &norm_v_sqr) {


#ifdef USE_AVX

    return get_reflection_vector_AVX(input, norm_v_sqr);

#else

    return get_reflection_vector<mtx_select_t<cplx_select_t<small_scalar_type>>, cplx_select_t<small_scalar_type>, small_scalar_type>(input, norm_v_sqr);

#endif

}


/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v A for one specific reflection vector v
@param A matrix on which the householder transformation is applied.
@param v A matrix instance of the reflection vector
@param vH_times_A preallocated array to hold the data of vH_times_A. The result is returned via this reference.
*/
template<class small_scalar_type>
void
calc_vH_times_A(mtx_select_t<cplx_select_t<small_scalar_type>> &A, mtx_select_t<cplx_select_t<small_scalar_type>> &v, mtx_select_t<cplx_select_t<small_scalar_type>> &vH_times_A) {


  if ( A.cols > HOUSEHOLDER_COTUFF) {

      size_t cols_mid = A.cols/2;
      mtx_select_t<cplx_select_t<small_scalar_type>> A1(A.get_data(), A.rows, cols_mid, A.stride);
      mtx_select_t<cplx_select_t<small_scalar_type>> vH_times_A_1(vH_times_A.get_data(), vH_times_A.rows, cols_mid, vH_times_A.stride);
      calc_vH_times_A<small_scalar_type>(A1, v, vH_times_A_1);

      mtx_select_t<cplx_select_t<small_scalar_type>> A2(A.get_data() + cols_mid, A.rows, A.cols - cols_mid, A.stride);
      mtx_select_t<cplx_select_t<small_scalar_type>> vH_times_A_2(vH_times_A.get_data() + cols_mid, vH_times_A.rows, vH_times_A.cols - cols_mid, vH_times_A.stride);
      calc_vH_times_A<small_scalar_type>(A2, v, vH_times_A_2);
      return;

  }
  else if ( A.rows > HOUSEHOLDER_COTUFF) {

      size_t rows_mid = A.rows/2;
      mtx_select_t<cplx_select_t<small_scalar_type>> A1(A.get_data(), rows_mid, A.cols, A.stride);
      mtx_select_t<cplx_select_t<small_scalar_type>> v1(v.get_data(), rows_mid, v.cols, v.stride);
      calc_vH_times_A<small_scalar_type>(A1, v1, vH_times_A);

      mtx_select_t<cplx_select_t<small_scalar_type>> A2(A.get_data() + rows_mid*A.stride, A.rows - rows_mid, A.cols, A.stride);
      mtx_select_t<cplx_select_t<small_scalar_type>> v2(v.get_data() + rows_mid*v.stride, v.rows - rows_mid, v.cols, v.stride);
      calc_vH_times_A<small_scalar_type>(A2, v2, vH_times_A);
      return;

  }
  else {




#ifdef USE_AVX

    calc_vH_times_A_AVX(A, v, vH_times_A);
    return;

#else

    calc_vH_times_A<mtx_select_t<cplx_select_t<small_scalar_type>>, cplx_select_t<small_scalar_type>>(A, v, vH_times_A);


    return;

#endif
  }


}


/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
@param vH_times_A The calculated product v^H * A calculated by calc_vH_times_A.
*/
template<class small_scalar_type>
void
calc_vov_times_A(mtx_select_t<cplx_select_t<small_scalar_type>> &A, mtx_select_t<cplx_select_t<small_scalar_type>> &v, mtx_select_t<cplx_select_t<small_scalar_type>> &vH_times_A) {

    if ( A.cols > HOUSEHOLDER_COTUFF) {

        size_t cols_mid = A.cols/2;
        mtx_select_t<cplx_select_t<small_scalar_type>> A1(A.get_data(), A.rows, cols_mid, A.stride);
        mtx_select_t<cplx_select_t<small_scalar_type>> vH_times_A_1(vH_times_A.get_data(), vH_times_A.rows, cols_mid, vH_times_A.stride);
        calc_vov_times_A<small_scalar_type>(A1, v, vH_times_A_1);

        mtx_select_t<cplx_select_t<small_scalar_type>> A2(A.get_data() + cols_mid, A.rows, A.cols - cols_mid, A.stride);
        mtx_select_t<cplx_select_t<small_scalar_type>> vH_times_A_2(vH_times_A.get_data() + cols_mid, vH_times_A.rows, vH_times_A.cols - cols_mid, vH_times_A.stride);
        calc_vov_times_A<small_scalar_type>(A2, v, vH_times_A_2);
        return;

    }
    else if ( A.rows > HOUSEHOLDER_COTUFF) {

        size_t rows_mid = A.rows/2;
        mtx_select_t<cplx_select_t<small_scalar_type>> A1(A.get_data(), rows_mid, A.cols, A.stride);
        mtx_select_t<cplx_select_t<small_scalar_type>> v1(v.get_data(), rows_mid, v.cols, v.stride);
        calc_vov_times_A<small_scalar_type>(A1, v1, vH_times_A);

        mtx_select_t<cplx_select_t<small_scalar_type>> A2(A.get_data() + rows_mid*A.stride, A.rows - rows_mid, A.cols, A.stride);
        mtx_select_t<cplx_select_t<small_scalar_type>> v2(v.get_data() + rows_mid*v.stride, v.rows - rows_mid, v.cols, v.stride);
        calc_vov_times_A<small_scalar_type>(A2, v2, vH_times_A);
        return;

    }
    else {

#ifdef USE_AVX

    calc_vov_times_A_AVX(A, v, vH_times_A);
    return;

#else

    calc_vov_times_A<mtx_select_t<cplx_select_t<small_scalar_type>>, cplx_select_t<small_scalar_type>>(A, v, vH_times_A);
    return;

#endif

    }



}


/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v/v^2) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
*/
template <class small_scalar_type>
void
apply_householder_rows(mtx_select_t<cplx_select_t<small_scalar_type>> &A, mtx_select_t<cplx_select_t<small_scalar_type>> &v) {


      // calculate A^~ = (1-2vov)A

      // allocate memory for the vector-matrix product v^+ A
      mtx_select_t<cplx_select_t<small_scalar_type>> vH_times_A(1, A.cols);
      memset(vH_times_A.get_data(), 0, vH_times_A.size()*sizeof(cplx_select_t<small_scalar_type>) );
      calc_vH_times_A<small_scalar_type>(A, v, vH_times_A);


      // calculate the vector-vector product v * ((v^+) * A))
      calc_vov_times_A<small_scalar_type>(A, v, vH_times_A);

      return;



}




/**
@brief Apply householder transformation on a matrix A' = A(1 - 2*v o v) for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
*/
template<class small_scalar_type>
void
apply_householder_cols_req(mtx_select_t<cplx_select_t<small_scalar_type>> &A, mtx_select_t<cplx_select_t<small_scalar_type>> &v) {

#ifdef USE_AVX

    apply_householder_cols_AVX(A, v);
    return;

#else

    apply_householder_cols_req<mtx_select_t<cplx_select_t<small_scalar_type>>, cplx_select_t<small_scalar_type>>(A, v);
    return;

#endif

}





/**
@brief Reduce a general matrix to upper Hessenberg form.
@param matrix matrix to be reduced to upper Hessenberg form. The reduced matrix is returned via this input
*/
template <class small_scalar_type>
void
transform_matrix_to_hessenberg(mtx_select_t<cplx_select_t<small_scalar_type>> &mtx) {

  // apply recursive Hauseholder transformation to eliminate the matrix elements column by column
  for (size_t idx = 1; idx < mtx.rows - 1; idx++) {

      // construct strided matrix containing data to get the reflection matrix
      mtx_select_t<cplx_select_t<small_scalar_type>> ref_vector_input(mtx.get_data() + idx*mtx.stride + idx - 1, mtx.rows-idx, 1, mtx.stride);

      // get reflection matrix and its norm
      small_scalar_type norm_v_sqr(0.0);
      mtx_select_t<cplx_select_t<small_scalar_type>> &&reflect_vector = get_reflection_vector(ref_vector_input, norm_v_sqr);

      if (norm_v_sqr == 0.0) continue;

      // construct strided submatrix in which the elements under the diagonal in the first column are transformed to zero by Householder transformation
      mtx_select_t<cplx_select_t<small_scalar_type>> mtx_strided(mtx.get_data() + idx*mtx.stride + idx - 1, mtx.rows-idx, mtx.cols-idx+1, mtx.stride);

      // apply Householder transformation from the left
      apply_householder_rows<small_scalar_type>(mtx_strided, reflect_vector);

      // construct strided submatrix on which the Householder transformation is applied from the right
      mtx_strided = mtx_select_t<cplx_select_t<small_scalar_type>>(mtx.get_data() + idx, mtx.rows, mtx.cols-idx, mtx.stride);

      // apply Householder transformation from the right
      apply_householder_cols_req<small_scalar_type>(mtx_strided, reflect_vector);

  }



}



/**
@brief Reduce a general matrix to upper Hessenberg form and applies the unitary transformation on left/right sided vectors to keep the \f$ <L|M|R> \f$ product invariant.
@param matrix matrix to be reduced to upper Hessenberg form. The reduced matrix is returned via this input
@param Lv the left sided vector
@param Rv the roght sided vector
*/
template <class small_scalar_type>
void
transform_matrix_to_hessenberg(mtx_select_t<cplx_select_t<small_scalar_type>> &mtx, mtx_select_t<cplx_select_t<small_scalar_type>>& Lv, mtx_select_t<cplx_select_t<small_scalar_type>>& Rv ) {

  // apply recursive Hauseholder transformation to eliminate the matrix elements column by column
  for (size_t idx = 1; idx < mtx.rows - 1; idx++) {

      // construct strided matrix containing data to get the reflection matrix
      mtx_select_t<cplx_select_t<small_scalar_type>> ref_vector_input(mtx.get_data() + idx*mtx.stride + idx - 1, mtx.rows-idx, 1, mtx.stride);

      // get reflection matrix and its norm
      small_scalar_type norm_v_sqr(0.0);
      mtx_select_t<cplx_select_t<small_scalar_type>> &&reflect_vector = get_reflection_vector(ref_vector_input, norm_v_sqr);

      if (norm_v_sqr == 0.0) continue;

      // apply Householder transformation on the matrix from the left
      mtx_select_t<cplx_select_t<small_scalar_type>> mtx_strided(mtx.get_data() + idx*mtx.stride + idx - 1, mtx.rows-idx, mtx.cols-idx+1, mtx.stride);
      apply_householder_rows<small_scalar_type>(mtx_strided, reflect_vector);

      // apply Householder transformation on the left vector
      mtx_select_t<cplx_select_t<small_scalar_type>> Lv_strided(Lv.get_data()+idx, Lv.rows, Lv.cols-idx, Lv.stride);
      apply_householder_cols_req<small_scalar_type>(Lv_strided, reflect_vector);

      // apply Householder transformation from the right
      mtx_strided = mtx_select_t<cplx_select_t<small_scalar_type>>(mtx.get_data() + idx, mtx.rows, mtx.cols-idx, mtx.stride);
      apply_householder_cols_req<small_scalar_type>(mtx_strided, reflect_vector);

      // apply Householder transformation on the right vector
      mtx_select_t<cplx_select_t<small_scalar_type>> Rv_strided(Rv.get_data()+Rv.stride*idx, Rv.rows-idx, Rv.cols, Rv.stride);
      apply_householder_rows<small_scalar_type>(Rv_strided, reflect_vector);


  }



}



/**
@brief Call to calculate the power traces \f$Tr(mtx^j)~\forall~1\leq j\leq l\f$ for a squared complex matrix \f$mtx\f$ of dimensions \f$n\times n\f$
and a loop corrections in Eq (3.26) of arXiv1805.12498
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
template <class small_scalar_type, class scalar_type>
void
CalcPowerTraces( mtx_select_t<cplx_select_t<small_scalar_type>>& AZ, size_t pow_max, mtx_select_t<cplx_select_t<scalar_type>> &traces32) {
    using small_matrix_type = mtx_select_t<cplx_select_t<small_scalar_type>>;
    using matrix_type = mtx_select_t<cplx_select_t<scalar_type>>;

    // for small matrices only the traces are casted into quad precision
    if (AZ.rows <= 10) {

        transform_matrix_to_hessenberg<small_scalar_type>(AZ);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        small_matrix_type&& coeffs_labudde = calc_characteristic_polynomial_coeffs<small_matrix_type, cplx_select_t<small_scalar_type>>(AZ, AZ.rows);

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
        small_matrix_type&& traces = powtrace_from_charpoly<small_matrix_type>(coeffs_labudde, pow_max);

        traces32 = matrix_type(traces.rows, traces.cols);
        for (size_t idx=0; idx<traces.size(); idx++) {
            traces32[idx].real( (scalar_type)traces[idx].real() );
            traces32[idx].imag( (scalar_type)traces[idx].imag() );
        }

        return;

    }
    // The lapack function to calculate the Hessenberg transformation is more efficient for larger matrices, but for above a given cutoff quad precision is needed
    // for these matrices of moderate size, the coefficients of the characteristic polynomials are casted into quad precision and the traces are calculated in
    // quad precision
    else if ( AZ.rows < 40 ) {


        transform_matrix_to_hessenberg<small_scalar_type>(AZ);

        matrix_type AZ32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ32[idx].real( AZ[idx].real() );
            AZ32[idx].imag( AZ[idx].imag() );
        }

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix_type&& coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix_type, cplx_select_t<scalar_type>>(AZ32, AZ.rows);

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
        traces32 = powtrace_from_charpoly<matrix_type>(coeffs_labudde, pow_max);

        return;



    }
    else{
        // above a treshold matrix size all the calculations are done in quad precision

        // matrix size for which quad precision is necessary

        matrix_type AZ32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ32[idx].real( AZ[idx].real() );
            AZ32[idx].imag( AZ[idx].imag() );
        }


        transform_matrix_to_hessenberg<matrix_type, cplx_select_t<scalar_type>, small_scalar_type>(AZ32);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix_type coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix_type, cplx_select_t<scalar_type>>(AZ32, AZ.rows);

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
        traces32 = powtrace_from_charpoly<matrix_type>(coeffs_labudde, pow_max);

        return;

    }


}

template void
CalcPowerTraces<double, double>( mtx_select_t<cplx_select_t<double>>& AZ, size_t pow_max, mtx_select_t<cplx_select_t<double>> &traces32);

template void
CalcPowerTraces<double, long double>( mtx_select_t<cplx_select_t<double>>& AZ, size_t pow_max, mtx_select_t<cplx_select_t<long double>> &traces32);

template<> void
CalcPowerTraces<long double, long double>( mtx_select_t<cplx_select_t<long double>>& AZ, size_t pow_max, mtx_select_t<cplx_select_t<long double>> &traces32)
{
    using matrix_type = mtx_select_t<cplx_select_t<long double>>;
    matrix_type AZ32( AZ.rows, AZ.cols);
    for (size_t idx=0; idx<AZ.size(); idx++) {
        AZ32[idx].real( AZ[idx].real() );
        AZ32[idx].imag( AZ[idx].imag() );
    }


    transform_matrix_to_hessenberg<matrix_type, cplx_select_t<long double>, long double>(AZ32);

    // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
    matrix_type coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix_type, cplx_select_t<long double>>(AZ32, AZ.rows);

    // calculate the power traces of the matrix AZ using LeVerrier recursion relation
    traces32 = powtrace_from_charpoly<matrix_type>(coeffs_labudde, pow_max);
}

template<>
void
transform_matrix_to_hessenberg<mtx_select_t<cplx_select_t<RationalInf>>, cplx_select_t<RationalInf>, RationalInf>(mtx_select_t<cplx_select_t<RationalInf>> &AZ) {
    using complex_type = cplx_select_t<RationalInf>;
    complex_type* AZdata = AZ.get_data();
    size_t i = 0;    
    while (i + 3 <= AZ.cols) {
        if (AZdata[(i+1)*AZ.stride+i] == 0) {
            size_t j;
            for (j = i+2; j < AZ.rows; j++) {
                if (AZdata[j*AZ.stride+i] != 0) break;
            }
            if (j == AZ.rows) break;
            for (size_t k = 0; k < AZ.rows; k++) {
                std::swap(AZdata[k*AZ.stride+i+1], AZdata[k*AZ.stride+j]);
            }
            for (size_t k = 0; k < AZ.cols; k++) {
                std::swap(AZdata[(i+1)*AZ.stride+k], AZdata[j*AZ.stride+k]);
            }
        }
        for (size_t k = i+2; k < AZ.rows; k++) {
            complex_type a = AZdata[k*AZ.stride+i] / AZdata[(i+1)*AZ.stride+i];
            for (size_t j = 0; j < AZ.cols; j++) {
                AZdata[k*AZ.stride+j] -= a * AZdata[(i+1)*AZ.stride+j];                
            }
            for (size_t j = 0; j < AZ.rows; j++) {
                AZdata[j*AZ.stride+i+1] += a * AZdata[j*AZ.stride+k];
            }
        }
        i++;
    }
}
#define USE_MATMUL_INFPREC
template<> void
CalcPowerTraces<RationalInf, RationalInf>( mtx_select_t<cplx_select_t<RationalInf>>& AZ, size_t pow_max, mtx_select_t<cplx_select_t<RationalInf>> &traces32) {
    using complex_type = cplx_select_t<RationalInf>;
    using matrix_type = mtx_select_t<complex_type>;
    matrix_type AZnew( AZ.rows, AZ.cols);
    complex_type* AZdata = AZ.get_data();
    complex_type* AZnewdata = AZnew.get_data();
    std::uninitialized_copy_n(AZdata, AZ.size(), AZnewdata);
    std::uninitialized_fill_n(traces32.get_data(), traces32.rows*traces32.cols, complex_type(0.0, 0.0));
#ifdef USE_MATMUL_INFPREC
    matrix_type AZ32( AZ.rows, AZ.cols);
    complex_type* AZ32data = AZ32.get_data();
    std::uninitialized_fill_n(AZ32data, AZ32.size(), complex_type(0.0, 0.0));
    for (size_t n = 0; n < traces32.size(); n++) {
        for (size_t i = 0; i < AZ.rows; i++) {
            traces32[n] += AZnewdata[i*AZnew.stride+i];
        }
        //traces32[n].real().num.print(); printf(" "); traces32[n].imag().num.print();
        if (n == traces32.size()-1) break;
        complex_type* swap = AZ32data;
        AZ32data = AZnewdata;
        AZnewdata = swap;
        std::fill_n(AZnewdata, AZnew.size(), complex_type(0.0, 0.0));
        for (size_t i = 0; i < AZ.rows; i++) {
            for (size_t j = 0; j < AZ.cols; j++) {
                for (size_t k = 0; k < AZ32.cols; k++) {
                    AZnewdata[i*AZnew.stride+k] += AZ32data[i*AZ32.stride+j] * AZdata[j*AZ.stride+k];
                }
            }
        }
    }
    for (size_t i = AZ32.size(); i != 0; i--) AZ32[i-1].~complex_type();
#else
    transform_matrix_to_hessenberg<matrix_type, complex_type, RationalInf>(AZnew);

    // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
    matrix_type coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix_type, complex_type>(AZnew, AZ.rows);
    //for (size_t i = coeffs_labudde.size(); i != 0; i--) coeffs_labudde[i-1].normalize();

    // calculate the power traces of the matrix AZ using LeVerrier recursion relation
    traces32 = powtrace_from_charpoly<matrix_type>(coeffs_labudde, pow_max);
    for (size_t i = coeffs_labudde.size(); i != 0; i--) coeffs_labudde[i-1].~complex_type();
#endif
    for (size_t i = AZnew.size(); i != 0; i--) AZnew[i-1].~complex_type();
}

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
CalcPowerTracesAndLoopCorrections( mtx_select_t<cplx_select_t<small_scalar_type>> &cx_diag_elements, mtx_select_t<cplx_select_t<small_scalar_type>> &diag_elements, mtx_select_t<cplx_select_t<small_scalar_type>>& AZ, size_t pow_max, mtx_select_t<cplx_select_t<scalar_type>> &traces32, mtx_select_t<cplx_select_t<scalar_type>> &loop_corrections32) {
    using small_matrix_type = mtx_select_t<cplx_select_t<small_scalar_type>>;
    using matrix_type = mtx_select_t<cplx_select_t<scalar_type>>;

    // for small matrices only the traces are casted into quad precision
    if (AZ.rows <= 10) {

        transform_matrix_to_hessenberg<small_scalar_type>(AZ, diag_elements, cx_diag_elements);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        small_matrix_type&& coeffs_labudde = calc_characteristic_polynomial_coeffs<small_matrix_type, cplx_select_t<small_scalar_type>>(AZ, AZ.rows);

        // calculate the loop correction
        small_matrix_type&& loop_corrections = calculate_loop_correction_2<small_scalar_type>( cx_diag_elements, diag_elements, AZ, pow_max);

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
        small_matrix_type&& traces = powtrace_from_charpoly<small_matrix_type>(coeffs_labudde, pow_max);

        traces32 = matrix_type(traces.rows, traces.cols);
        for (size_t idx=0; idx<traces.size(); idx++) {
            traces32[idx].real( (scalar_type)traces[idx].real() );
            traces32[idx].imag( (scalar_type)traces[idx].imag() );
        }

        loop_corrections32 = matrix_type(loop_corrections.rows, loop_corrections.cols);
        for (size_t idx=0; idx<loop_corrections.size(); idx++) {
            loop_corrections32[idx].real( loop_corrections[idx].real() );
            loop_corrections32[idx].imag( loop_corrections[idx].imag() );
        }

        return;

/*
    traces32 = matrix_type(pow_max, 1);
loop_corrections32 = matrix_type(pow_max, 1);
memset( traces32.get_data(), 0.0, traces32.size()*sizeof(Complex32));
        memset( loop_corrections32.get_data(), 0.0, loop_corrections32.size()*sizeof(Complex32));
        return;
*/

    }
    // The lapack function to calculate the Hessenberg transformation is more efficient for larger matrices, but for above a given cutoff quad precision is needed
    // for these matrices of moderate size, the coefficients of the characteristic polynomials are casted into quad precision and the traces are calculated in
    // quad precision
    else if ( AZ.rows < 40 ) {


        transform_matrix_to_hessenberg<small_scalar_type>(AZ, diag_elements, cx_diag_elements);

        matrix_type AZ32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ32[idx].real( AZ[idx].real() );
            AZ32[idx].imag( AZ[idx].imag() );
        }

        matrix_type diag_elements32(diag_elements.rows, diag_elements.cols);
        for (size_t idx=0; idx<diag_elements.size(); idx++) {
            diag_elements32[idx].real( diag_elements[idx].real() );
            diag_elements32[idx].imag( diag_elements[idx].imag() );
        }

        matrix_type cx_diag_elements32(cx_diag_elements.rows, cx_diag_elements.cols);
        for (size_t idx=0; idx<diag_elements.size(); idx++) {
            cx_diag_elements32[idx].real( cx_diag_elements[idx].real() );
            cx_diag_elements32[idx].imag( cx_diag_elements[idx].imag() );
        }

        // calculate the loop correction
        loop_corrections32 = calculate_loop_correction_2<matrix_type, cplx_select_t<scalar_type>>( cx_diag_elements32, diag_elements32, AZ32, pow_max);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix_type&& coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix_type, cplx_select_t<scalar_type>>(AZ32, AZ.rows);

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
        traces32 = powtrace_from_charpoly<matrix_type>(coeffs_labudde, pow_max);

        return;

    }
    else{
        // above a treshold matrix size all the calculations are done in quad precision

        // matrix size for which quad precision is necessary

        matrix_type AZ32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ32[idx].real( AZ[idx].real() );
            AZ32[idx].imag( AZ[idx].imag() );
        }

        matrix_type diag_elements32(diag_elements.rows, diag_elements.cols);
        for (size_t idx=0; idx<diag_elements.size(); idx++) {
            diag_elements32[idx].real( diag_elements[idx].real() );
            diag_elements32[idx].imag( diag_elements[idx].imag() );
        }

        matrix_type cx_diag_elements32(cx_diag_elements.rows, cx_diag_elements.cols);
        for (size_t idx=0; idx<diag_elements.size(); idx++) {
            cx_diag_elements32[idx].real( cx_diag_elements[idx].real() );
            cx_diag_elements32[idx].imag( cx_diag_elements[idx].imag() );
        }


        transform_matrix_to_hessenberg<matrix_type, cplx_select_t<scalar_type>, small_scalar_type>(AZ32, diag_elements32, cx_diag_elements32);

        // calculate the loop correction
        loop_corrections32 = calculate_loop_correction_2<matrix_type, cplx_select_t<scalar_type>>( cx_diag_elements32, diag_elements32, AZ32, pow_max);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix_type coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix_type, cplx_select_t<scalar_type>>(AZ32, AZ.rows);

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
        traces32 = powtrace_from_charpoly<matrix_type>(coeffs_labudde, pow_max);

        return;

    }




}

template void
CalcPowerTracesAndLoopCorrections<double, long double>( mtx_select_t<cplx_select_t<double>> &cx_diag_elements, mtx_select_t<cplx_select_t<double>> &diag_elements, mtx_select_t<cplx_select_t<double>>& AZ, size_t pow_max, mtx_select_t<cplx_select_t<long double>> &traces32, mtx_select_t<cplx_select_t<long double>> &loop_corrections32);

/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
template <class small_scalar_type>
mtx_select_t<cplx_select_t<small_scalar_type>>
calculate_loop_correction_2( mtx_select_t<cplx_select_t<small_scalar_type>> &cx_diag_elements, mtx_select_t<cplx_select_t<small_scalar_type>> &diag_elements, mtx_select_t<cplx_select_t<small_scalar_type>>& AZ, size_t num_of_modes) {


#ifdef USE_AVX
    return calculate_loop_correction_AVX( cx_diag_elements, diag_elements, AZ, num_of_modes);
#else
    return calculate_loop_correction_2<mtx_select_t<cplx_select_t<small_scalar_type>>, cplx_select_t<small_scalar_type>>( cx_diag_elements, diag_elements, AZ, num_of_modes);
#endif




}



} // PIC

