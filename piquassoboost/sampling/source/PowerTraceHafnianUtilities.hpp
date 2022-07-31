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

#ifndef PowerTraceHafnianUtilities_TEMPLATE_HPP
#define PowerTraceHafnianUtilities_TEMPLATE_HPP



#ifndef LONG_DOUBLE_CUTOFF
#define LONG_DOUBLE_CUTOFF 40
#endif // LONG_DOUBLE_CUTOFF

#include "PowerTraceHafnianUtilities.h"
#include <iostream>
#include "common_functionalities.h"
#include <math.h>
#include "tbb/tick_count.h"

#define HOUSEHOLDER_COTUFF 40

/*
static tbb::spin_mutex my_mutex;

static double time_szamlalo = 0.0;
static double time_nevezo = 0.0;
*/

namespace pic {



/**
/@brief Determine the reflection vector for Householder transformation used in the upper Hessenberg transformation algorithm
@param input The strided input vector constructed from the k-th column of the matrix on which the Hessenberg transformation should be applied
@param norm_v_sqr The squared norm of the created reflection matrix that is returned by reference
@return Returns with the calculated reflection vector
 */
template<class matrix_type, class complex_type, class small_scalar_type>
matrix_type
get_reflection_vector(matrix_type &input, small_scalar_type &norm_v_sqr) {

  small_scalar_type sigma(0.0);
  norm_v_sqr = 0.0;
  matrix_type reflect_vector(input.rows,1);
  for (size_t idx = 0; idx < reflect_vector.size(); idx++) {
      complex_type &element = input[idx*input.stride];
      reflect_vector[idx] =  element;//mtx[(idx + offset) * mtx_size + offset - 1];
      norm_v_sqr = norm_v_sqr + element.real()*element.real() + element.imag()*element.imag(); //adding the squared magnitude
  }
  sigma = sqrt(norm_v_sqr);


  small_scalar_type abs_val = std::sqrt( reflect_vector[0].real()*reflect_vector[0].real() + reflect_vector[0].imag()*reflect_vector[0].imag() );
  norm_v_sqr = 2*(norm_v_sqr + abs_val*sigma);
  if (abs_val != 0.0){
      //small_scalar_type angle = std::arg(reflect_vector[0]); // sigma *= (reflect_vector[0] / std::abs(reflect_vector[0]));
      auto addend = reflect_vector[0]/abs_val*sigma;
      reflect_vector[0].real( reflect_vector[0].real() + addend.real());
      reflect_vector[0].imag( reflect_vector[0].imag() + addend.imag());
  }
  else {
      reflect_vector[0].real( reflect_vector[0].real() + sigma );
  }

  if (norm_v_sqr == 0.0)
      return reflect_vector;

  // normalize the reflection matrix
  small_scalar_type norm_v = std::sqrt(norm_v_sqr);
  for (size_t idx=0; idx<reflect_vector.size(); idx++) {
      reflect_vector[idx] = reflect_vector[idx]/norm_v;
  }

  norm_v_sqr = 1.0;

  return reflect_vector;
}



/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v A for one specific reflection vector v
@param A matrix on which the householder transformation is applied.
@param v A matrix instance of the reflection vector
@param vH_times_A preallocated array to hold the data of vH_times_A. The result is returned via this reference.
*/
template<class matrix_type, class complex_type>
void
calc_vH_times_A(matrix_type &A, matrix_type &v, matrix_type &vH_times_A) {


    if ( A.cols > HOUSEHOLDER_COTUFF) {

        size_t cols_mid = A.cols/2;
        matrix_type A1(A.get_data(), A.rows, cols_mid, A.stride);
        matrix_type vH_times_A_1(vH_times_A.get_data(), vH_times_A.rows, cols_mid, vH_times_A.stride);
        calc_vH_times_A<matrix_type, complex_type>(A1, v, vH_times_A_1);

        matrix_type A2(A.get_data() + cols_mid, A.rows, A.cols - cols_mid, A.stride);
        matrix_type vH_times_A_2(vH_times_A.get_data() + cols_mid, vH_times_A.rows, vH_times_A.cols - cols_mid, vH_times_A.stride);
        calc_vH_times_A<matrix_type, complex_type>(A2, v, vH_times_A_2);
        return;

    }
    else if ( A.rows > HOUSEHOLDER_COTUFF) {

        size_t rows_mid = A.rows/2;
         matrix_type A1(A.get_data(), rows_mid, A.cols, A.stride);
        matrix_type v1(v.get_data(), rows_mid, v.cols, v.stride);
        calc_vH_times_A<matrix_type, complex_type>(A1, v1, vH_times_A);

        matrix_type A2(A.get_data() + rows_mid*A.stride, A.rows - rows_mid, A.cols, A.stride);
        matrix_type v2(v.get_data() + rows_mid*v.stride, v.rows - rows_mid, v.cols, v.stride);
        calc_vH_times_A<matrix_type, complex_type>(A2, v2, vH_times_A);
        return;

    }
    else {

        size_t sizeH = v.size();


  // calculate the vector-matrix product (v^+) * A
        for (size_t row_idx = 0; row_idx < sizeH-1; row_idx=row_idx+2) {

            size_t offset_A_data =  row_idx * A.stride;
            complex_type* data_A = A.get_data() + offset_A_data;
            complex_type* data_A2 = data_A + A.stride;

            for (size_t j = 0; j < A.cols-1; j = j + 2) {
                vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);
                vH_times_A[j+1] = vH_times_A[j+1] + mult_a_bconj(data_A[j+1], v[row_idx]);
                vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A2[j], v[row_idx+1]);
                vH_times_A[j+1] = vH_times_A[j+1] + mult_a_bconj(data_A2[j+1], v[row_idx+1]);
            }


            if (A.cols % 2 == 1) {
                size_t j = A.cols-1;
                vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);
                vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A2[j], v[row_idx+1]);

            }


        }

        if (sizeH % 2 == 1) {

            size_t row_idx = sizeH-1;

            size_t offset_A_data =  row_idx * A.stride;
            complex_type* data_A = A.get_data() + offset_A_data;


            for (size_t j = 0; j < A.cols-1; j = j + 2) {
                vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);
                vH_times_A[j+1] = vH_times_A[j+1] + mult_a_bconj(data_A[j+1], v[row_idx]);
            }


            if (A.cols % 2 == 1) {
                size_t j = A.cols-1;
                vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);

            }



        }

/*
        size_t sizeH = v.size();

        // calculate the vector-matrix product (v^+) * A
        for (size_t row_idx = 0; row_idx < sizeH; row_idx++) {

            size_t offset_A_data =  row_idx * A.stride;
            complex_type* data_A = A.get_data() + offset_A_data;

            for (size_t j = 0; j < A.cols; j++) {
                vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);
            }


        }
*/

        return;

    }


}


/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
@param vH_times_A The calculated product v^H * A calculated by calc_vH_times_A.
*/
template<class matrix_type, class complex_type>
void
calc_vov_times_A(matrix_type &A, matrix_type &v, matrix_type &vH_times_A) {

    if ( A.cols > HOUSEHOLDER_COTUFF) {

        size_t cols_mid = A.cols/2;
        matrix_type A1(A.get_data(), A.rows, cols_mid, A.stride);
        matrix_type vH_times_A_1(vH_times_A.get_data(), vH_times_A.rows, cols_mid, vH_times_A.stride);
        calc_vov_times_A<matrix_type, complex_type>(A1, v, vH_times_A_1);

        matrix_type A2(A.get_data() + cols_mid, A.rows, A.cols - cols_mid, A.stride);
        matrix_type vH_times_A_2(vH_times_A.get_data() + cols_mid, vH_times_A.rows, vH_times_A.cols - cols_mid, vH_times_A.stride);
        calc_vov_times_A<matrix_type, complex_type>(A2, v, vH_times_A_2);
        return;

    }
    else if ( A.rows > HOUSEHOLDER_COTUFF) {

        size_t rows_mid = A.rows/2;
        matrix_type A1(A.get_data(), rows_mid, A.cols, A.stride);
        matrix_type v1(v.get_data(), rows_mid, v.cols, v.stride);
        calc_vov_times_A<matrix_type, complex_type>(A1, v1, vH_times_A);

        matrix_type A2(A.get_data() + rows_mid*A.stride, A.rows - rows_mid, A.cols, A.stride);
        matrix_type v2(v.get_data() + rows_mid*v.stride, v.rows - rows_mid, v.cols, v.stride);
        calc_vov_times_A<matrix_type, complex_type>(A2, v2, vH_times_A);
        return;

    }
    else {

        size_t size_v = v.size();

        for (size_t row_idx = 0; row_idx < size_v-1; row_idx = row_idx+2) {

            size_t offset_data_A =  row_idx * A.stride;
            complex_type* data_A = A.get_data() + offset_data_A;
            complex_type* data_A2 = data_A + A.stride;

            complex_type factor = v[row_idx]*2.0;
            complex_type factor2 = v[row_idx+1]*2.0;

            for (size_t kdx = 0; kdx < A.cols-1; kdx=kdx+2) {
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx];
                data_A2[kdx] = data_A2[kdx] - factor2 * vH_times_A[kdx];
                data_A[kdx+1] = data_A[kdx+1] - factor * vH_times_A[kdx+1];
                data_A2[kdx+1] = data_A2[kdx+1] - factor2 * vH_times_A[kdx+1];
            }


            if ( A.cols % 2 == 1) {
                size_t kdx = A.cols-1;
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx];
                data_A2[kdx] = data_A2[kdx] - factor2 * vH_times_A[kdx];
            }


        }



        if (size_v % 2 == 1 ) {

            size_t row_idx = v.rows-1;
            complex_type* data_A = A.get_data() + row_idx * A.stride;

            complex_type factor = v[row_idx]*2.0;

            for (size_t kdx = 0; kdx < A.cols-1; kdx=kdx+2) {
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx];
                data_A[kdx+1] = data_A[kdx+1] - factor * vH_times_A[kdx+1];
            }


            if ( A.cols % 2 == 1) {
                size_t kdx = A.cols-1;
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx];
            }

        }
/*

        // calculate the vector-vector product v * ((v^+) * A))
        for (size_t row_idx = 0; row_idx < v.rows; row_idx++) {

            size_t offset_data_A =  row_idx * A.stride;
            complex_type* data_A = A.get_data() + offset_data_A;

            complex_type factor = v[row_idx]*2.0;
            for (size_t j = 0; j < A.cols; j++) {
                data_A[j] = data_A[j] - factor * vH_times_A[j];
            }
        }
*/

        return;

    }

}


/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v/v^2) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
*/
template<class matrix_type, class complex_type>
void
apply_householder_rows(matrix_type &A, matrix_type &v) {


      // calculate A^~ = (1-2vov)A

      // allocate memory for the vector-matrix product v^+ A
      matrix_type vH_times_A(1, A.cols);
      memset(vH_times_A.get_data(), 0, vH_times_A.size()*sizeof(complex_type) );
      calc_vH_times_A<matrix_type, complex_type>(A, v, vH_times_A);


      // calculate the vector-vector product v * ((v^+) * A))
      calc_vov_times_A<matrix_type, complex_type>(A, v, vH_times_A);

      return;



}




/**
@brief Apply householder transformation on a matrix A' = A(1 - 2*v o v) for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
*/
template<class matrix_type, class complex_type>
void
apply_householder_cols_req(matrix_type &A, matrix_type &v) {

    size_t sizeH = v.size();

    for (size_t idx = 0; idx < A.rows-1; idx=idx+2) {

        complex_type* data_A = A.get_data() + idx*A.stride;
        complex_type* data_A2 = data_A + A.stride;

        complex_type factor(0.0,0.0);
        complex_type factor2(0.0,0.0);
        for (size_t v_idx = 0; v_idx < sizeH; v_idx++) {
            factor  = factor  + data_A[v_idx] * v[v_idx];
            factor2 = factor2 + data_A2[v_idx] * v[v_idx];
        }


        factor  = factor*2.0;
        factor2 = factor2*2.0;
        for (size_t jdx=0; jdx<sizeH; jdx++) {
            data_A[jdx] = data_A[jdx] - mult_a_bconj(factor, v[jdx]);
            data_A2[jdx] = data_A2[jdx] - mult_a_bconj(factor2, v[jdx]);
        }


    }


    if (A.rows % 2 == 1 ) {

        complex_type* data_A = A.get_data() + (A.rows-1)*A.stride;

        complex_type factor(0.0,0.0);
        for (size_t v_idx = 0; v_idx < sizeH; v_idx++) {
            factor = factor + data_A[v_idx] * v[v_idx];
        }

        factor = factor*2.0;
        for (size_t jdx=0; jdx<sizeH; jdx++) {
            data_A[jdx] = data_A[jdx] - mult_a_bconj(factor, v[jdx]);
        }



    }

/*
    size_t sizeH = v.size();

    // calculate A^~(1-2vov)
    for (size_t idx = 0; idx < A.rows; idx++) {
        size_t offset_data_A = idx*A.stride;
        complex_type* data_A = A.get_data() + offset_data_A;

        complex_type factor(0.0,0.0);
        for (size_t v_idx = 0; v_idx < sizeH; v_idx++) {
            factor = factor + data_A[v_idx] * v[v_idx];
        }

        factor = factor*2.0;
        for (int jdx=0; jdx<sizeH; jdx++) {
            data_A[jdx] = data_A[jdx] - mult_a_bconj(factor, v[jdx]);
        }

    }

*/
    return;


}





/**
@brief Reduce a general matrix to upper Hessenberg form.
@param matrix matrix to be reduced to upper Hessenberg form. The reduced matrix is returned via this input
*/
template<class matrix_type, class complex_type, class small_scalar_type>
void
transform_matrix_to_hessenberg(matrix_type &mtx) {

  // apply recursive Hauseholder transformation to eliminate the matrix elements column by column
  for (size_t idx = 1; idx < mtx.rows - 1; idx++) {

      // construct strided matrix containing data to get the reflection matrix
      matrix_type ref_vector_input(mtx.get_data() + idx*mtx.stride + idx - 1, mtx.rows-idx, 1, mtx.stride);

      // get reflection matrix and its norm
      small_scalar_type norm_v_sqr(0.0);
      matrix_type &&reflect_vector = get_reflection_vector<matrix_type, complex_type, small_scalar_type>(ref_vector_input, norm_v_sqr);

      if (norm_v_sqr == 0.0) continue;

      // construct strided submatrix in which the elements under the diagonal in the first column are transformed to zero by Householder transformation
      matrix_type mtx_strided(mtx.get_data() + idx*mtx.stride + idx - 1, mtx.rows-idx, mtx.cols-idx+1, mtx.stride);

      // apply Householder transformation from the left
      apply_householder_rows<matrix_type, complex_type>(mtx_strided, reflect_vector);

      // construct strided submatrix on which the Householder transformation is applied from the right
      mtx_strided = matrix_type(mtx.get_data() + idx, mtx.rows, mtx.cols-idx, mtx.stride);

      // apply Householder transformation from the right
      apply_householder_cols_req<matrix_type, complex_type>(mtx_strided, reflect_vector);

  }



}



/**
@brief Reduce a general matrix to upper Hessenberg form and applies the unitary transformation on left/right sided vectors to keep the \f$ <L|M|R> \f$ product invariant.
@param matrix matrix to be reduced to upper Hessenberg form. The reduced matrix is returned via this input
@param Lv the left sided vector
@param Rv the roght sided vector
*/
template<class matrix_type, class complex_type, class small_scalar_type>
void
transform_matrix_to_hessenberg(matrix_type &mtx, matrix_type Lv, matrix_type Rv ) {

  // apply recursive Hauseholder transformation to eliminate the matrix elements column by column
  for (size_t idx = 1; idx < mtx.rows - 1; idx++) {

      // construct strided matrix containing data to get the reflection matrix
      matrix_type ref_vector_input(mtx.get_data() + idx*mtx.stride + idx - 1, mtx.rows-idx, 1, mtx.stride);

      // get reflection matrix and its norm
      small_scalar_type norm_v_sqr(0.0);
      matrix_type &&reflect_vector = get_reflection_vector<matrix_type, complex_type, small_scalar_type>(ref_vector_input, norm_v_sqr);

      if (norm_v_sqr == 0.0) continue;

      // apply Householder transformation on the matrix from the left
      matrix_type mtx_strided(mtx.get_data() + idx*mtx.stride + idx - 1, mtx.rows-idx, mtx.cols-idx+1, mtx.stride);
      apply_householder_rows<matrix_type, complex_type>(mtx_strided, reflect_vector);

      // apply Householder transformation on the left vector
      matrix_type Lv_strided(Lv.get_data()+idx, Lv.rows, Lv.cols-idx, Lv.stride);
      apply_householder_cols_req<matrix_type, complex_type>(Lv_strided, reflect_vector);

      // apply Householder transformation from the right
      mtx_strided = matrix_type(mtx.get_data() + idx, mtx.rows, mtx.cols-idx, mtx.stride);
      apply_householder_cols_req<matrix_type, complex_type>(mtx_strided, reflect_vector);

      // apply Householder transformation on the right vector
      matrix_type Rv_strided(Rv.get_data()+Rv.stride*idx, Rv.rows-idx, Rv.cols, Rv.stride);
      apply_householder_rows<matrix_type, complex_type>(Rv_strided, reflect_vector);


  }



}




/**
@brief Call to determine the first \f$ k \f$ coefficients of the characteristic polynomial using the Algorithm 2 of LaBudde method.
 See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1) for further details.
@param mtx matrix in upper Hessenberg form.
@param highest_order the order of the highest order coefficient to be calculated (k <= n)
@return Returns with the calculated coefficients of the characteristic polynomial.
 *
 */
template<class matrix_type, class complex_type>
matrix_type
calc_characteristic_polynomial_coeffs(matrix_type &mtx, size_t highest_order)
{
 // the matrix c holds not just the polynomial coefficients, but also auxiliary
 // data. To retrieve the characteristic polynomial coeffients from the matrix c, use
 // this map for characteristic polynomial coefficient c_j:
 // if j = 0, c_0 -> 1
 // if j > 0, c_j -> c[(n - 1) * n + j - 1]


    // check the dimensions of the matrix in debug mode
    assert( mtx.rows == mtx.cols);

    //dimension of the matrix
    size_t dim = mtx.rows;


    // allocate memory for the coefficients c_k of p(\lambda)
    matrix_type coeffs(dim, dim);
    memset(coeffs.get_data(), 0, dim*dim*sizeof(complex_type));


    // c^(1)_1 = -\alpha_1
    coeffs[0] = -mtx[0];

    // c^(2)_1 = c^(1)_1 - \alpha_2
    coeffs[dim] = coeffs[0] - mtx[dim+1];

    // c^(2)_2 = \alpha_1\alpha_2 - h_{12}\beta_2
    coeffs[dim+1] =  mtx[0]*mtx[dim+1] - mtx[1]*mtx[dim];

    // for (i=3:k do)
    for (size_t idx=2; idx<=highest_order-1; idx++) {
        // i = idx + 1

        // calculate the products of matrix elements \beta_i
        // the n-th (0<=n<=idx-2) element of the arary stands for:
        // beta_prods[n] = \beta_i * \beta_i-1 * ... * \beta_{i-n}
        matrix_type beta_prods(idx,1);
        beta_prods[0] = mtx[idx*dim + idx-1];
        for (size_t prod_idx=1; prod_idx<=idx-1; prod_idx++) {
            beta_prods[prod_idx] = beta_prods[prod_idx-1] * mtx[(idx-prod_idx)*dim + (idx-prod_idx-1)];
        }

        // c^(i)_1 = c^(i-1)_1 - \alpha_i
        coeffs[idx*dim] = coeffs[(idx-1)*dim] - mtx[idx*dim + idx];

        // for j=2 : i-1 do
        for (size_t jdx=1; jdx<=idx-1; jdx++) {
            // j = jdx + 1

            // sum = \sum_^{j-2}{m=1} h_{i-m,i} \beta_i*...*\beta_{i-m+1} c^{(i-m-1)}_{j-m-1}  - h_{i-j+1,i}* beta_i*...*beta_{i-j+2}
            complex_type sum(0.0,0.0);

            // for m=j-2 : 1 do
            for ( size_t mdx=1; mdx<=jdx-1; mdx++) {
                // m = mdx

                // sum = sum + h_{i-m, i} * beta_prod * c^{(i-m-1)}_{j-m-1}
                sum = sum + mtx[(idx-mdx)*dim+idx] * beta_prods[mdx-1] * coeffs[(idx-mdx-1)*dim + jdx-mdx-1];

            }

            // sum = sum + h_{i-j+1,i} * \beta_prod
            sum = sum + mtx[(idx-jdx)*dim + idx] * beta_prods[jdx-1];

            // c^(i)_j = c^(i-1)_j - \alpha_i*c^(i-1)_{j-1} - sum
            coeffs[idx*dim+jdx] = coeffs[(idx-1)*dim+jdx] - mtx[idx*dim+idx] * coeffs[(idx-1)*dim + jdx-1] - sum;
        }

        // sum = \sum_^{i-2}{m=1} h_{i-m,i} \beta_i*...*\beta_{i-m+1} c^{(i-m-1)}_{i-m-1}  - h_{1,i}* beta_i*...*beta_{2}
        complex_type sum(0.0,0.0);

        // for m=j-2 : 1 do
        for ( size_t mdx=1; mdx<=idx-1; mdx++) {
            // m = mdx

            // sum = sum + h_{i-m, i} * beta_prod * c^{(i-m-1)}_{j-m-1}
            sum = sum + mtx[(idx-mdx)*dim+idx] * beta_prods[mdx-1] * coeffs[(idx-mdx-1)*dim + idx-mdx-1];
        }

        // c^(i)_i = -\alpha_i c^{(i-1)}_{i-1} - sum
        coeffs[idx*dim+idx] = -mtx[idx*dim+idx]*coeffs[(idx-1)*dim+idx-1] - sum - mtx[idx]*beta_prods[beta_prods.size()-1];

    }

    // for i=k+1 : n do
    for (size_t idx = highest_order; idx<dim; idx++ ) {
        // i = idx + 1

        // c^(i)_1 = c^(i-1)_1 - \alpha_i
        coeffs[idx*dim] = coeffs[(idx-1)*dim] - mtx[idx*dim + idx];

        // calculate the products of matrix elements \beta_i
        // the n-th (0<=n<=idx-2) element of the arary stands for:
        // beta_prods[n] = \beta_i * \beta_i-1 * ... * \beta_{i-n}

        if (highest_order >= 2) {
            matrix_type beta_prods(idx,1);
            beta_prods[0] = mtx[idx*dim + idx-1];
            for (size_t prod_idx=1; prod_idx<=idx-1; prod_idx++) {
                beta_prods[prod_idx] = beta_prods[prod_idx-1] * mtx[(idx-prod_idx)*dim + (idx-prod_idx-1)];
            }

            // for j = 2 : k do
            for (size_t jdx=1; jdx<=highest_order-1; jdx++) {
                // j = jdx + 1

                // sum = \sum_^{j-2}{m=1} h_{i-m,i} \beta_i*...*\beta_{i-m+1} c^{(i-m-1)}_{j-m-1}  - h_{i-j+1,i}* beta_i*...*beta_{i-j+2}
                complex_type sum(0.0,0.0);

                // for m=j-2 : 1 do
                for ( size_t mdx=1; mdx<=jdx-1; mdx++) {
                    // m = mdx

                    // sum = sum + h_{i-m, i} * beta_prod * c^{(i-m-1)}_{j-m-1}
                    sum = sum + mtx[(idx-mdx)*dim+idx] * beta_prods[mdx-1] * coeffs[(idx-mdx-1)*dim + jdx-mdx-1];

                }

                // sum = sum + h_{i-j+1,i} * \beta_prod
                sum = sum + mtx[(idx-jdx)*dim + idx] * beta_prods[jdx-1];

                // c^(i)_j = c^(i-1)_j - \alpha_i*c^(i-1)_{j-1} - sum
                coeffs[idx*dim+jdx] = coeffs[(idx-1)*dim+jdx] - mtx[idx*dim+idx] * coeffs[(idx-1)*dim + jdx-1] - sum;
            }



         }
    }


    return coeffs;
}



/**
@brief Call to calculate the traces of \f$ A^{p}\f$, where 1<=p<=pow is an integer and A is a square matrix.
The trace is calculated from the coefficients of its characteristic polynomial.
In the case that the power p is above the size of the matrix we can use an optimization described in Appendix B of [arxiv:1805.12498](https://arxiv.org/pdf/1104.3769v1.pdf)
@param c matrix containing the characteristic polynomial coefficients
@param pow the maximal exponent p
@return Returns with the calculated power traces
 */
template<class matrix_type>
matrix_type
powtrace_from_charpoly(matrix_type &coeffs, size_t pow) {

    size_t dim = coeffs.rows;


    if (pow == 0) {
        matrix_type ret(1,1);
        ret[0].real( (double) dim );
        ret[0].imag( 0.0 );
        return ret;
    }

    // allocate memory for the power traces
    matrix_type traces(pow,1);


    // Tr(A) = -c1
    size_t element_offset = (dim - 1) * dim;
    traces[0] = -coeffs[element_offset];

  // Calculate power traces using the LeVerrier recursion relation
  size_t kdx_max = pow < dim ? pow : dim;
  for (size_t idx = 2; idx <= kdx_max; idx++) {

    // Tr(A^idx)
    size_t element_offset2 = (dim - 1) * dim + idx - 1;
    traces[idx - 1] = coeffs[element_offset2] * (-(double)idx);

    for (size_t j = idx - 1; j >= 1; j--) {
      traces[idx - 1] -= coeffs[element_offset2 - j] * traces[j - 1];
    }

  }


  // Appendix B optimization
  if (pow > dim) {
    for (size_t idx = 1; idx <= pow - dim; idx++) {

      size_t element_offset = dim + idx - 1;
      size_t element_offset_coeffs = (dim - 1) * dim - 1;
      traces[element_offset] = 0.0;

      for (size_t jdx = 1; jdx <= dim; jdx++) {
        traces[element_offset] -= traces[element_offset - jdx] * coeffs[element_offset_coeffs + jdx];
      }

    }

  } // if


  return traces;

}





/**
@brief Call to calculate the power traces \f$Tr(mtx^j)~\forall~1\leq j\leq l\f$ for a squared complex matrix \f$mtx\f$ of dimensions \f$n\times n\f$.
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@param pow_max maximum matrix power when calculating the power trace.
@return a vector containing the power traces of matrix `z` to power \f$1\leq j \leq l\f$.
*/
template<class matrix_type, class complex_type, class scalar_type, class small_matrix_type, class small_complex_type, class small_scalar_type>
matrix_type
calc_power_traces(matrix &AZ, size_t pow_max) {

    // for small matrices only the traces are casted into quad precision
    if (AZ.rows <= 10) {

        transform_matrix_to_hessenberg<small_matrix_type, small_complex_type, small_scalar_type>(AZ);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        small_matrix_type&& coeffs_labudde = calc_characteristic_polynomial_coeffs<small_matrix_type, small_complex_type>(AZ, AZ.rows);

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
        small_matrix_type&& traces = powtrace_from_charpoly<small_matrix_type>(coeffs_labudde, pow_max);

        matrix_type traces32(traces.rows, traces.cols);
        for (size_t idx=0; idx<traces.size(); idx++) {
            traces32[idx].real( (scalar_type)traces[idx].real() );
            traces32[idx].imag( (scalar_type)traces[idx].imag() );
        }

        return traces32;

    }
    // The lapack function to calculate the Hessenberg transformation is more efficient for larger matrices, but for above a given cutoff quad precision is needed
    // for these matrices of moderate size, the coefficients of the characteristic polynomials are casted into quad precision and the traces are calculated in
    // quad precision
    else if ( (AZ.rows < 40) || (sizeof(complex_type) == sizeof(Complex16)) ) {

        // transform the matrix mtx into an upper Hessenberg format by calling lapack function
        int N = AZ.rows;
        int ILO = 1;
        int IHI = N;
        int LDA = N;
        matrix tau(N-1,1);
        LAPACKE_zgehrd(LAPACK_ROW_MAJOR, N, ILO, IHI, AZ.get_data(), LDA, tau.get_data() );

        matrix_type AZ32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ32[idx].real( AZ[idx].real() );
            AZ32[idx].imag( AZ[idx].imag() );
        }

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix_type&& coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix_type, complex_type>(AZ32, AZ.rows);

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
        return powtrace_from_charpoly<matrix_type>(coeffs_labudde, pow_max);



    }
    else{
        // above a treshold matrix size all the calculations are done in quad precision

        // matrix size for which quad precision is necessary

        matrix_type AZ32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ32[idx].real( AZ[idx].real() );
            AZ32[idx].imag( AZ[idx].imag() );
        }

        transform_matrix_to_hessenberg<matrix_type, complex_type, small_scalar_type>(AZ32);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix_type coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix_type, complex_type>(AZ32, AZ.rows);

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
        return powtrace_from_charpoly<matrix_type>(coeffs_labudde, pow_max);

    }



/*
{
            tbb::spin_mutex::scoped_lock my_lock{my_mutex};
            time_szamlalo = time_szamlalo  + (t1-t0).seconds();
            time_nevezo = time_nevezo  + (t3-t2).seconds();
            std::cout << time_szamlalo/time_nevezo << std::endl;
        }
*/





}






/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
template<class matrix_type, class complex_type>
matrix_type
calculate_loop_correction( matrix_type &cx_diag_elements, matrix_type& diag_elements, matrix_type& AZ, size_t num_of_modes) {

    matrix_type loop_correction(num_of_modes, 1);
    matrix_type tmp_vec(1, diag_elements.size());

    for (size_t idx=0; idx<num_of_modes; idx++) {

        complex_type tmp(0.0,0.0);
        for (size_t jdx=0; jdx<diag_elements.size(); jdx++) {
            tmp = tmp + diag_elements[jdx] * cx_diag_elements[jdx];
        }

        loop_correction[idx] = tmp;


        memset(tmp_vec.get_data(), 0, tmp_vec.size()*sizeof(complex_type));

        if (sizeof(complex_type) == 16) {

            Complex16 alpha(1.0,0.0);
            Complex16 beta(0.0,0.0);

            cblas_zgemv(CblasRowMajor, CblasNoTrans, AZ.rows, AZ.cols, (void*)&alpha, (void*)AZ.get_data(), AZ.stride,
            (void*)cx_diag_elements.get_data(), 1, (void*)&beta, (void*)tmp_vec.get_data(), 1);
        }
        else {

            for (size_t jdx=0; jdx<cx_diag_elements.size(); jdx++) {
                tmp = complex_type(0.0,0.0);
                complex_type* data = AZ.get_data() + jdx*AZ.stride;
                for (size_t kdx=0; kdx<cx_diag_elements.size(); kdx++) {
                    tmp += data[kdx] * cx_diag_elements[kdx];
                }
                tmp_vec[jdx] = tmp;
            }
        }


        memcpy(cx_diag_elements.get_data(), tmp_vec.get_data(), tmp_vec.size()*sizeof(complex_type));

    }



/*
    for (size_t idx=0; idx<num_of_modes; idx++) {

        complex_type tmp(0.0,0.0);
        for (size_t jdx=0; jdx<diag_elements.size(); jdx++) {
            tmp = tmp + cx_diag_elements[jdx] * diag_elements[jdx];
        }

        loop_correction[idx] = tmp;


         memset(tmp_vec.get_data(), 0, tmp_vec.size()*sizeof(complex_type));

         for (size_t kdx=0; kdx<cx_diag_elements.size(); kdx++) {
             for (size_t jdx=0; jdx<cx_diag_elements.size(); jdx++) {
                  tmp_vec[jdx] = tmp_vec[jdx] + cx_diag_elements[kdx] * AZ[kdx * AZ.stride + jdx];
             }
         }

         memcpy(cx_diag_elements.get_data(), tmp_vec.get_data(), tmp_vec.size()*sizeof(complex_type));

    }
*/

    return loop_correction;

}






/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
template<class matrix_type, class complex_type>
matrix_type
calculate_loop_correction_2( matrix_type &cx_diag_elements, matrix_type& diag_elements, matrix_type& AZ, size_t num_of_modes) {

    matrix_type loop_correction(num_of_modes, 1);
    //transform_matrix_to_hessenberg<matrix_type, complex_type, small_scalar_type>(AZ, diag_elements, cx_diag_elements);

    size_t max_idx = cx_diag_elements.size();
    matrix_type tmp_vec(1, max_idx);
    complex_type* cx_data = cx_diag_elements.get_data();
    complex_type* diag_data = diag_elements.get_data();


    for (size_t idx=0; idx<num_of_modes; idx++) {


        complex_type tmp(0.0,0.0);
        for (size_t jdx=0; jdx<max_idx; jdx++) {
            tmp = tmp + diag_data[jdx] * cx_data[jdx];
        }

        loop_correction[idx] = tmp;


        complex_type* data = AZ.get_data();


        tmp = complex_type(0.0,0.0);
        for (size_t kdx=0; kdx<max_idx; kdx++) {
            tmp += data[kdx] * cx_data[kdx];
        }
        tmp_vec[0] = tmp;


        for (size_t jdx=1; jdx<max_idx; jdx++) {
            data = data + AZ.stride;
            tmp = complex_type(0.0,0.0);
            for (size_t kdx=jdx-1; kdx<max_idx; kdx++) {
                tmp += data[kdx] * cx_data[kdx];
            }
            tmp_vec[jdx] = tmp;

        }


        memcpy(cx_diag_elements.get_data(), tmp_vec.get_data(), tmp_vec.size()*sizeof(complex_type));

    }


    return loop_correction;

}




} // PIC

#endif // PowerTraceHafnianUtilities_TEMPLATE_HPP
