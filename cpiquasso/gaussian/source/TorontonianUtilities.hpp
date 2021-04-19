#include "PowerTraceHafnianUtilities.h"
#include <iostream>
#include "common_functionalities.h"
#include <math.h>


static tbb::spin_mutex my_mutex;
/*
double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {


/**
/@brief Determine the reflection vector for Householder transformation used in the upper Hessenberg transformation algorithm
@param mtx The matrix instance on which the Hessenberg transformation should be applied
@param offset Starting index (i.e. offset index) of rows/columns from which the householder transformation should be applied
@return Returns with the calculated reflection vector
 */
template<class matrix_type, class complex_type>
matrix_type
get_reflection_vector(matrix_type &vector_strided) {

  size_t element_num = vector_strided.size();
  double sigma(0.0);
  double abs_value = std::sqrt( vector_strided[0].real()*vector_strided[0].real() + vector_strided[0].imag()*vector_strided[0].imag() );

  matrix_type reflect_vector(element_num,1);
  for (size_t idx = 0; idx < element_num; idx++) {
    complex_type &element = vector_strided[idx*vector_strided.stride];
    reflect_vector[idx] = element;
    sigma = sigma + element.real()*element.real() + element.imag()*element.imag(); //adding the squared magnitude
  }
  sigma = sqrt(sigma);


  if (abs_value != 0.0){
    //double angle = std::arg(reflect_vector[0]); // sigma *= (reflect_vector[0] / std::abs(reflect_vector[0]));
    auto addend = reflect_vector[0]/abs_value*sigma;
    reflect_vector[0].real( reflect_vector[0].real() + addend.real());
    reflect_vector[0].imag( reflect_vector[0].imag() + addend.imag());
  }
  else {
    reflect_vector[0].real( reflect_vector[0].real() + sigma );
  }


  return reflect_vector;

}


/**
/@brief Determine the reflection vector for Householder transformation used in the upper Hessenberg transformation algorithm
@param mtx The matrix instance on which the Hessenberg transformation should be applied
@param offset Starting index (i.e. offset index) of rows/columns from which the householder transformation should be applied
@return Returns with the calculated reflection vector
 */
template<class matrix_type, class complex_type>
matrix_type
get_reflection_vector(matrix_type &mtx, size_t offset) {


  size_t mtx_size = mtx.rows;

  size_t sizeH = mtx_size - offset;


  double sigma(0.0);
  matrix_type reflect_vector(sizeH,1);
  for (size_t idx = 0; idx < sizeH; idx++) {
    reflect_vector[idx] = mtx[(idx + offset) * mtx_size + offset - 1];
    sigma = sigma + reflect_vector[idx].real()*reflect_vector[idx].real() + reflect_vector[idx].imag()*reflect_vector[idx].imag(); //adding the squared magnitude
  }
  sigma = sqrt(sigma);

  double abs_value = std::sqrt( reflect_vector[0].real()*reflect_vector[0].real() + reflect_vector[0].imag()*reflect_vector[0].imag() );
  if (abs_value != 0.0){
    //double angle = std::arg(reflect_vector[0]); // sigma *= (reflect_vector[0] / std::abs(reflect_vector[0]));
    auto addend = reflect_vector[0]/abs_value*sigma;
    reflect_vector[0].real( reflect_vector[0].real() + addend.real());
    reflect_vector[0].imag( reflect_vector[0].imag() + addend.imag());
  }
  else {
    reflect_vector[0].real( reflect_vector[0].real() + sigma );
  }


  return reflect_vector;
}


/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v/v^2) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
@param offset Starting index (i.e. offset index) of rows/columns from which the householder transformation should be applied
*/
template<class matrix_type, class complex_type>
void
apply_householder_col(matrix_type &A, matrix_type &v) {

  double norm_v_sqr = 0.0;
  for (size_t idx=0; idx<v.size(); idx++) {
      norm_v_sqr = norm_v_sqr + v[idx].real()*v[idx].real() + v[idx].imag()*v[idx].imag();
  }


  if (norm_v_sqr == 0.0)
    return;


  size_t element_num = v.size();

  // calculate A^~ = (1-2vov)A

  // allocate memory for the vector-matrix product v^+ A
  matrix_type vH_times_A(1, A.cols);
  memset(vH_times_A.get_data(), 0, vH_times_A.size()*sizeof(complex_type) );

  // calculate the vector-matrix product (v^+) * A
  for (size_t row_idx = 0; row_idx < element_num; row_idx++) {

    size_t offset_A_data =  row_idx*A.stride;
    complex_type* data_A = A.get_data() + offset_A_data;

    for (size_t jdx = 0; jdx < A.cols; jdx++) {
      vH_times_A[jdx] = vH_times_A[jdx] + mult_a_bconj(data_A[jdx], v[row_idx]);
    }


  }


  // calculate the vector-vector product v * ((v^+) * A))
  for (size_t row_idx = 0; row_idx < element_num; row_idx++) {

    size_t offset_data_A =  row_idx*A.stride;
    complex_type* data_A = A.get_data() + offset_data_A;

    complex_type factor = (v[row_idx]/norm_v_sqr)*2.0;
    for (size_t jdx = 0; jdx < A.cols; jdx++) {
      data_A[jdx] = data_A[jdx] - factor * vH_times_A[jdx];
    }
  }

  return;

}




/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v/v^2) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
@param offset Starting index (i.e. offset index) of rows/columns from which the householder transformation should be applied
*/
template<class matrix_type, class complex_type>
void
apply_householder_row(matrix_type &A, matrix_type &v) {

  double norm_v_sqr = 0.0;
  for (size_t idx=0; idx<v.size(); idx++) {
      norm_v_sqr = norm_v_sqr + v[idx].real()*v[idx].real() + v[idx].imag()*v[idx].imag();
  }


  if (norm_v_sqr == 0.0)
    return;


  size_t element_num = v.size();


  // calculate A^~ = A^~(1-2vov)
  for (size_t idx = 0; idx < A.rows; idx++) {
    size_t offset_data_A = idx*A.stride;
    complex_type* data_A = A.get_data() + offset_data_A;

    complex_type factor(0.0,0.0);
    for (size_t v_idx = 0; v_idx < element_num; v_idx++) {
        //factor = factor + data_A[v_idx] * v[v_idx];
        factor = factor + mult_a_bconj(data_A[v_idx], v[v_idx]);
    }

    factor = (factor/norm_v_sqr)*2.0;
    for (int jdx=0; jdx<element_num; jdx++) {
        //data_A[jdx] = data_A[jdx] - mult_a_bconj(factor, v[jdx]);
        data_A[jdx] = data_A[jdx] - factor * v[jdx];
    }

  }

  return;


}




/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v/v^2) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
@param offset Starting index (i.e. offset index) of rows/columns from which the householder transformation should be applied
*/
template<class matrix_type, class complex_type>
void
apply_householder(matrix_type &A, matrix_type &v, size_t offset) {



  double norm_v_sqr = 0.0;
  for (size_t idx=0; idx<v.size(); idx++) {
      norm_v_sqr = norm_v_sqr + v[idx].real()*v[idx].real() + v[idx].imag()*v[idx].imag();
  }


  if (norm_v_sqr == 0.0)
    return;


  size_t sizeH = v.size();
  size_t size_A = A.rows;

  // calculate A^~ = (1-2vov)A

  // allocate memory for the vector-matrix product v^+ A
  matrix_type vH_times_A(size_A - offset + 1, 1);
  memset(vH_times_A.get_data(), 0, vH_times_A.size()*sizeof(complex_type) );

  // calculate the vector-matrix product (v^+) * A
  for (size_t row_idx = 0; row_idx < sizeH; row_idx++) {

    size_t offset_A_data =  (offset + row_idx) * size_A + offset - 1;
    complex_type* data_A = A.get_data() + offset_A_data;

    for (size_t j = 0; j < size_A - offset + 1; j++) {
      vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);
    }


  }


  // calculate the vector-vector product v * ((v^+) * A))
  for (size_t row_idx = 0; row_idx < sizeH; row_idx++) {

    size_t offset_data_A =  (offset + row_idx) * size_A + offset - 1;
    complex_type* data_A = A.get_data() + offset_data_A;

    complex_type factor = (v[row_idx]/norm_v_sqr)*2.0;
    for (size_t j = 0; j < size_A - offset + 1; j++) {
      data_A[j] = data_A[j] - factor * vH_times_A[j];
    }
  }


  // calculate A^~ = A^~(1-2vov)
  for (size_t idx = 0; idx < size_A; idx++) {
    size_t offset_data_A = (idx)*size_A + offset;
    complex_type* data_A = A.get_data() + offset_data_A;

    complex_type factor(0.0,0.0);
    for (size_t v_idx = 0; v_idx < sizeH; v_idx++) {
        factor = factor + data_A[v_idx] * v[v_idx];
    }

    factor = (factor/norm_v_sqr)*2.0;
    for (int jdx=sizeH-1; jdx>=0; jdx--) {
        data_A[jdx] = data_A[jdx] - mult_a_bconj(factor, v[jdx]);
    }

  }

  return;


}


/**
@brief Calculate the determinant of a symmetric or selfadjoint matrix int Hessenberg form. 
See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1) for further details.
@param mtx matrix in upper Hessenberg form. It has to be symmetric or selfadjoint.
@return determinant of the matrix.
*/
template<class matrix_type, class complex_type>
complex_type
calc_determinant_of_selfadjoint_hessenberg_matrix(matrix_type& mtx) {
    // dimension of the matrix
    size_t n = mtx.rows;

    // allocate memory for the recursive calculation
    complex_type* subdeterminants = (complex_type*)scalable_aligned_malloc( n*sizeof(complex_type), CACHELINE);
    
    // the first elements are calculated previously
    if (n <= 2){
        if (n == 1){
            // determinant is the single element of the matrix
            return mtx[0];
        }else{
            // det : (0,0)*(1,1)-(0,1)*(1,0)
            return mtx[0]*mtx[3] - mtx[1]*mtx[2];
        }
    }

    // first subdeterminant (0,0) element
    subdeterminants[0] = mtx[0];
    // second subdeterminant: (0,0)*(1,1)-(0,1)*(1,0)
    subdeterminants[1] = mtx[0]*mtx[n + 1] - mtx[1]*mtx[n];

    // determinant of the matrix in hessenberg form can be calculated based on
    for (size_t idx = 2; idx < n; idx++){
        subdeterminants[idx] = 
            subdeterminants[idx-1]* mtx[n*idx + idx]
          - subdeterminants[idx-2]*mtx[n*idx + idx - 1]*mtx[n*(idx-1) + idx];
    }

    // the determinant is the last subdeterminant
    return subdeterminants[n-1];
}


/**
@brief Reduce a general matrix to upper Hessenberg form.
@param matrix matrix to be reduced to upper Hessenberg form. The reduced matrix is returned via this input
*/
template<class matrix_type, class complex_type>
void
transform_matrix_to_hessenberg_TU(matrix_type &mtx) {

//matrix_type mtx_cp = mtx.copy();

/*
  // apply recursive Hauseholder transformation to eliminate the matrix elements column by column
  for (size_t idx = 1; idx < mtx.rows - 1; idx++) {
    matrix_type &&reflect_vector = get_reflection_vector<matrix_type, complex_type>(mtx, idx);
    apply_householder<matrix_type, complex_type>(mtx, reflect_vector, idx);
  }
  */

//{
//    tbb::spin_mutex::scoped_lock my_lock{my_mutex};

  // apply recursive Hauseholder transformation to eliminate the matrix elements column by column
  for (size_t idx = 1; idx < mtx.rows - 1; idx++) {

    // apply Hauseholder transformation (1-2vov)A to transform the elements of the idx-th column into zeros

    // get the reflection vector
    complex_type *data = mtx.get_data() + idx*mtx.stride + idx-1;
    matrix_type vector_strided(data, mtx.rows-idx, 1, mtx.stride);
    matrix_type &&reflect_vector = get_reflection_vector<matrix_type, complex_type>(vector_strided);


    // apply Householder transformation on a matrix block to zero out the elements in the idx-th column
    data = mtx.get_data() + idx*mtx.stride + idx-1;
    matrix_type mtx_block(data, mtx.rows-idx, mtx.cols-idx+1, mtx.stride );
    apply_householder_col<matrix_type, complex_type>(mtx_block, reflect_vector );

    // apply Hauseholder transformation A(1-2vov) to transform the elements of the idx-th row into zeros

    // get the reflection vector
    data = mtx.get_data() + (idx-1)*mtx.stride + idx;
    vector_strided = matrix_type(data, mtx.cols-idx, 1, 1);
    reflect_vector = get_reflection_vector<matrix_type, complex_type>(vector_strided);

    // apply Householder transformation on a matrix block to zero out the elements in the idx-th row
    data = mtx.get_data() + (idx-1)*mtx.stride + idx;
    mtx_block = matrix_type(data, mtx.rows-idx+1, mtx.cols-idx, mtx.stride );
    apply_householder_row<matrix_type, complex_type>(mtx_block, reflect_vector );
/*
    mtx.print_matrix();
    reflect_vector.print_matrix();
*/
 // }


}
/*
if (mtx.rows == 6) {
        mtx.print_matrix();
        //mtx_cp.print_matrix();
        exit(-1);

  }
*/
}




template<class matrix_type, class complex_type>
complex_type
calc_determinant_hessenberg_labudde_symmetric(matrix& mtx){
    // for small matrices nothing has to be casted into quad precision
    if (mtx.rows <= 10) {
        transform_matrix_to_hessenberg_TU<matrix, Complex16>(mtx);

        return calc_determinant_of_selfadjoint_hessenberg_matrix<matrix, Complex16>(mtx);
    }
    // The lapack function to calculate the Hessenberg transformation is more efficient for larger matrices, but for above a given cutoff quad precision is needed
    // for these matrices of moderate size, the coefficients of the characteristic polynomials are casted into quad precision and the traces are calculated in
    // quad precision
    else if ( (mtx.rows < 30 && (sizeof(complex_type) > sizeof(Complex16))) || (sizeof(complex_type) == sizeof(Complex16)) ) {


        // transform the matrix mtx into an upper Hessenberg format by calling lapack function
        int N = mtx.rows;
        int ILO = 1;
        int IHI = N;
        int LDA = N;
        matrix tau(N-1,1);
        LAPACKE_zgehrd(LAPACK_ROW_MAJOR, N, ILO, IHI, mtx.get_data(), LDA, tau.get_data() );

        matrix_type mtx32( mtx.rows, mtx.cols);
        for (size_t idx=0; idx<mtx.size(); idx++) {
            mtx32[idx].real( mtx[idx].real() );
            mtx32[idx].imag( mtx[idx].imag() );
        }

        return calc_determinant_of_selfadjoint_hessenberg_matrix<matrix_type, complex_type>(mtx);
    }
    else{
        // above a treshold matrix size all the calculations are done in quad precision
        // matrix size for which quad precision is necessary

        matrix_type mtx32( mtx.rows, mtx.cols);
        for (size_t idx=0; idx<mtx.size(); idx++) {
            mtx32[idx].real( mtx[idx].real() );
            mtx32[idx].imag( mtx[idx].imag() );
        }

        transform_matrix_to_hessenberg_TU<matrix_type, complex_type>(mtx32);

        return calc_determinant_of_selfadjoint_hessenberg_matrix<matrix_type, complex_type>(mtx32);
    }

}


// Cholesky decomposition
// Works for selfadjoint positive definite matrices!
template<class matrix_type, class complex_type>
matrix_type
calc_cholesky_decomposition(matrix_type& matrix)
{
    // storing in the same memory the results of the algorithm
    size_t n = matrix.cols;
    // Decomposing a matrix into lower triangular matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            complex_type sum = 0;
 
            if (j == i) 
            {
                // summation for diagnols
                // L_{j,j}=\sqrt{A_{j,j}-\sum_{k=0}^{j-1}L_{j,k}L_{j,k}^*}
                for (int k = 0; k < j; k++){
                    sum += matrix[j*matrix.stride+k] * conjugate(matrix[j*matrix.stride+k]);
                }
                matrix[j*matrix.stride+j] = sqrt(matrix[j*matrix.stride+j] -
                                        sum);
            } else {
                // Evaluating L(i, j) using L(j, j)
                // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
                for (int k = 0; k < j; k++){
                    sum += (matrix[i*matrix.stride+k] * conjugate(matrix[j*matrix.stride+k]));
                }
                matrix[i*matrix.stride+j] = (matrix[i*matrix.stride+j] - sum) /
                                      matrix[j*matrix.stride+j];
            }
        }
    }
    return matrix;
    // creating another matrix to store the result of the algorithm
/*
    size_t n = matrix.cols;
    matrix_type lower_tridiag(n, n);
    memset(lower_tridiag.data, 0, n * n * sizeof(complex_type));
 
    // Decomposing a matrix into lower triangular matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            complex_type sum = 0;
 
            if (j == i) 
            {
                // summation for diagnols
                // L_{j,j}=\sqrt{A_{j,j}-\sum_{k=0}^{j-1}L_{j,k}L_{j,k}^*}
                for (int k = 0; k < j; k++){
                    sum += lower_tridiag[j*lower_tridiag.stride+k] * conjugate(lower_tridiag[j*lower_tridiag.stride+k]);
                }
                lower_tridiag[j*lower_tridiag.stride+j] = sqrt(matrix[j*matrix.stride+j] -
                                        sum);
                //std::cout << "("<<i<<","<<j<<"): "<<lower_tridiag[i*lower_tridiag.stride+j]<<std::endl;
            } else {
                // Evaluating L(i, j) using L(j, j)
                // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
                for (int k = 0; k < j; k++){
                    sum += (lower_tridiag[i*lower_tridiag.stride+k] * conjugate(lower_tridiag[j*lower_tridiag.stride+k]));
                }
                lower_tridiag[i*lower_tridiag.stride+j] = (matrix[i*matrix.stride+j] - sum) /
                                      lower_tridiag[j*lower_tridiag.stride+j];
                //std::cout << "("<<i<<","<<j<<"): "<<lower_tridiag[i*lower_tridiag.stride+j]<<std::endl;
            }
        }
    }
    return lower_tridiag;
*/
}

// calculating determinant based on cholesky decomposition
template<class matrix_type, class complex_type>
complex_type
calc_determinant_cholesky_decomposition(matrix& mtx){
    // for small matrices nothing has to be casted into quad precision
    if (mtx.rows <= 10) {
        calc_cholesky_decomposition<matrix, Complex16>(mtx);

        Complex16 determinant(1, 0);

        for (size_t idx; idx < mtx.cols; idx++){
            Complex16 elem = mtx[idx * mtx.stride + idx];
            determinant *= elem;
        }
        
        return determinant * conjugate(determinant);
    }
    // The lapack function to calculate the Cholesky decomposition is more efficient for larger matrices, but for above a given cutoff quad precision is needed
    // for these matrices of moderate size, the coefficients of the characteristic polynomials are casted into quad precision and the traces are calculated in
    // quad precision
    else if ( (mtx.rows < 30 && (sizeof(complex_type) > sizeof(Complex16))) || (sizeof(complex_type) == sizeof(Complex16)) ) {


        // transform the matrix mtx into an upper Hessenberg format by calling lapack function
        char UPLO = 'L';
        int N = mtx.rows;
        int LDA = N;
        int INFO = 0;

        //std::cout<<"Before lapacke call:\n";
        //mtx.print_matrix();


        LAPACKE_zpotrf(LAPACK_ROW_MAJOR, UPLO, N, mtx.get_data(), LDA);

        //std::cout<<"After lapacke call:\n";
        //mtx.print_matrix();


        matrix_type diag( 1, N);
        for (size_t idx=0; idx<N; idx++) {
            diag[idx].real( mtx[idx*N + idx].real() );
            diag[idx].imag( mtx[idx*N + idx].imag() );
        }
        complex_type det = complex_type(1,0);

        for (size_t idx=0; idx<N; idx++) {
            det *= diag[idx];
        }
        return det * conjugate(det);
    }
    else{
        // above a treshold matrix size all the calculations are done in quad precision
        // matrix size for which quad precision is necessary
        return complex_type(1,0);
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
    memset(coeffs.get_data(), 0, dim*dim*sizeof(Complex16));


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


  // Tr(A)
  traces[0] = -coeffs[(dim - 1) * dim];

  // Calculate power traces using the LeVerrier recursion relation
  size_t kdx_max = pow < dim ? pow : dim;
  for (size_t idx = 2; idx <= kdx_max; idx++) {

    // Tr(A^idx)
    size_t element_offset = (dim - 1) * dim + idx - 1;
    traces[idx - 1] = coeffs[element_offset] * (-(double)idx);

    for (size_t j = idx - 1; j >= 1; j--) {
      traces[idx - 1] -= coeffs[element_offset - j] * traces[j - 1];
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
template<class matrix_type, class complex_type>
matrix_type
calc_power_traces(matrix &AZ, size_t pow_max) {

    // for small matrices only the traces are casted into quad precision
    if (AZ.rows <= 10) {

        transform_matrix_to_hessenberg_TU<matrix, Complex16>(AZ);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix&& coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix, Complex16>(AZ, AZ.rows);

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
        matrix&& traces = powtrace_from_charpoly<matrix>(coeffs_labudde, pow_max);

        matrix_type traces32(traces.rows, traces.cols);
        for (size_t idx=0; idx<traces.size(); idx++) {
            traces32[idx].real( (long double)traces[idx].real() );
            traces32[idx].imag( (long double)traces[idx].imag() );
        }

        return traces32;

    }
    // The lapack function to calculate the Hessenberg transformation is more efficient for larger matrices, but for above a given cutoff quad precision is needed
    // for these matrices of moderate size, the coefficients of the characteristic polynomials are casted into quad precision and the traces are calculated in
    // quad precision
    else if ( (AZ.rows < 30 && (sizeof(complex_type) > sizeof(Complex16))) || (sizeof(complex_type) == sizeof(Complex16)) ) {


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
        matrix32&& coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix32, Complex32>(AZ32, AZ.rows);

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

        transform_matrix_to_hessenberg_TU<matrix_type, complex_type>(AZ32);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix_type coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix_type, complex_type>(AZ32, AZ.rows);

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
        return powtrace_from_charpoly<matrix_type>(coeffs_labudde, pow_max);

    }



/*
{
            tbb::spin_mutex::scoped_lock my_lock{my_mutex};
            time_nominator = time_nominator  + (t1-t0).seconds();
            time_nevezo = time_nevezo  + (t3-t2).seconds();
            std::cout << time_nominator/time_nevezo << std::endl;
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
calculate_loop_correction( matrix_type &diag_elements, matrix_type& cx_diag_elements, matrix_type& AZ, size_t num_of_modes) {

    matrix_type loop_correction(num_of_modes, 1);
    matrix_type tmp_vec(1, diag_elements.size());


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


    return loop_correction;

}



} // PIC

