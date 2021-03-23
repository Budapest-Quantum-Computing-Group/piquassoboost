#include <iostream>
#include "PowerTraceHafnianUtilities.h"
#include "common_functionalities.h"
#include <math.h>

extern "C" {

#define LAPACK_ROW_MAJOR               101

/// Definition of the LAPACKE_zgehrd function from LAPACKE to calculate the upper triangle hessenberg transformation of a matrix
int LAPACKE_zgehrd( int matrix_layout, int n, int ilo, int ihi, pic::Complex16* a, int lda, pic::Complex16* tau );

}

/*
tbb::spin_mutex my_mutex;

double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {


/**
@brief Call to calculate the power traces \f$Tr(mtx^j)~\forall~1\leq j\leq l\f$ for a squared complex matrix \f$mtx\f$ of dimensions \f$n\times n\f$.
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@param pow_max maximum matrix power when calculating the power trace.
@return a vector containing the power traces of matrix `z` to power \f$1\leq j \leq l\f$.
*/
matrix
calc_power_traces(matrix &AZ, size_t pow_max) {

    // The lapack function is more efficient for larger matrices
    if (AZ.rows > 10) {

        // transform the matrix mtx into an upper hessenberg format by calling lapack function
        int N = AZ.rows;
        int ILO = 1;
        int IHI = N;
        int LDA = N;
        matrix tau(N-1,1);
        LAPACKE_zgehrd(LAPACK_ROW_MAJOR, N, ILO, IHI, AZ.get_data(), LDA, tau.get_data() );

    }
    else {
        transform_matrix_to_hessenberg(AZ);
    }

/*
{
            tbb::spin_mutex::scoped_lock my_lock{my_mutex};
            time_nominator = time_nominator  + (t1-t0).seconds();
            time_nevezo = time_nevezo  + (t3-t2).seconds();
            std::cout << time_nominator/time_nevezo << std::endl;
        }
*/

    // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
    matrix coeffs_labudde = calc_characteristic_polynomial_coeffs(AZ, AZ.rows);

    // calculate the power traces of the matrix AZ using LeVerrier recursion relation
    return powtrace_from_charpoly(coeffs_labudde, pow_max);
}




/**
@brief Call to determine the first \f$ k \f$ coefficients of the characteristic polynomial using the Algorithm 2 of LaBudde method.
 See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1) for further details.
@param mtx matrix in upper Hessenberg form.
@param highest_order the order of the highest order coefficient to be calculated (k <= n)
@return Returns with the calculated coefficients of the characteristic polynomial.
 *
 */
matrix
calc_characteristic_polynomial_coeffs(matrix &mtx, size_t highest_order)
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
    matrix coeffs(dim, dim);
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
        matrix beta_prods(idx,1);
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
            Complex16 sum(0.0,0.0);

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
        Complex16 sum(0.0,0.0);

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
            matrix beta_prods(idx,1);
            beta_prods[0] = mtx[idx*dim + idx-1];
            for (size_t prod_idx=1; prod_idx<=idx-1; prod_idx++) {
                beta_prods[prod_idx] = beta_prods[prod_idx-1] * mtx[(idx-prod_idx)*dim + (idx-prod_idx-1)];
            }

            // for j = 2 : k do
            for (size_t jdx=1; jdx<=highest_order-1; jdx++) {
                // j = jdx + 1

                // sum = \sum_^{j-2}{m=1} h_{i-m,i} \beta_i*...*\beta_{i-m+1} c^{(i-m-1)}_{j-m-1}  - h_{i-j+1,i}* beta_i*...*beta_{i-j+2}
                Complex16 sum(0.0,0.0);

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
matrix
powtrace_from_charpoly(matrix &coeffs, size_t pow) {

  size_t dim = coeffs.rows;


  if (pow == 0) {
    matrix ret(1,1);
    ret[0].real( (double) dim );
    ret[0].imag( 0.0 );
    return ret;
  }

  // allocate memory for the power traces
  matrix traces(pow,1);


  // Tr(A)
  traces[0] = -coeffs[(dim - 1) * dim];

  // Calculate power traces using the LeVerrier recursion relation
  size_t kdx_max = pow < dim ? pow : dim;
  for (size_t idx = 2; idx <= kdx_max; idx++) {

    // Tr(A^idx)
    size_t element_offset = (dim - 1) * dim + idx - 1;
    traces[idx - 1] = -(double)idx * coeffs[element_offset];

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
/@brief Determine the reflection vector for Householder transformation used in the upper Hessenberg transformation algorithm
@param mtx The matrix instance on which the Hessenberg transformation should be applied
@param offset Starting index (i.e. offset index) of rows/columns from which the householder transformation should be applied
@return Returns with the calculated reflection vector
 */
matrix
get_reflection_vector(matrix &mtx, size_t offset) {

  size_t mtx_size = mtx.rows;

  size_t sizeH = mtx_size - offset;


  double sigma(0.0);
  matrix reflect_vector(sizeH,1);
  for (size_t idx = 0; idx < sizeH; idx++) {
    reflect_vector[idx] = mtx[(idx + offset) * mtx_size + offset - 1];
    sigma = sigma + std::norm(reflect_vector[idx]); //adding the squared magnitude
  }
  sigma = sqrt(sigma);

  if (reflect_vector[0] != Complex16(0.0,0.0)){
    double angle = std::arg(reflect_vector[0]); // sigma *= (reflect_vector[0] / std::abs(reflect_vector[0]));
    reflect_vector[0] = reflect_vector[0] + sigma*std::polar(1.0, angle);
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
void
apply_householder(matrix &A, matrix &v, size_t offset) {


  double norm_v_sqr = 0.0;
  for (size_t idx=0; idx<v.size(); idx++) {
      norm_v_sqr = norm_v_sqr + std::norm(v[idx]);
  }


  if (norm_v_sqr == 0.0)
    return;


  size_t sizeH = v.size();
  size_t size_A = A.rows;

  // calculate A^~ = (1-2vov)A

  // allocate memory for the vector-matrix product v^+ A
  matrix vH_times_A(size_A - offset + 1, 1);
  memset(vH_times_A.get_data(), 0, vH_times_A.size()*sizeof(Complex16) );

  // calculate the vector-matrix product (v^+) * A
  for (size_t row_idx = 0; row_idx < sizeH; row_idx++) {

    size_t offset_A_data =  (offset + row_idx) * size_A + offset - 1;
    Complex16* data_A = A.get_data() + offset_A_data;

    for (size_t j = 0; j < size_A - offset + 1; j++) {
      vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);
    }


  }




  // calculate the vector-vector product v * ((v^+) * A))
  for (size_t row_idx = 0; row_idx < sizeH; row_idx++) {

    size_t offset_data_A =  (offset + row_idx) * size_A + offset - 1;
    Complex16* data_A = A.get_data() + offset_data_A;

    Complex16 factor = (v[row_idx]/norm_v_sqr)*2.0;
    for (size_t j = 0; j < size_A - offset + 1; j++) {
      data_A[j] = data_A[j] - factor * vH_times_A[j];
    }
  }


  // calculate A^~(1-2vov)
  for (size_t idx = 0; idx < size_A; idx++) {
    size_t offset_data_A = (idx)*size_A + offset;
    Complex16* data_A = A.get_data() + offset_data_A;

    Complex16 factor(0.0,0.0);
    for (size_t v_idx = 0; v_idx < sizeH; v_idx++) {
        factor = factor + data_A[v_idx] * v[v_idx];
    }

    factor = 2.0*factor/norm_v_sqr;
    for (int jdx=sizeH-1; jdx>=0; jdx--) {
        data_A[jdx] = data_A[jdx] - mult_a_bconj(factor, v[jdx]);
    }

  }


  return;


}

/**
@brief Reduce a general matrix to upper Hessenberg form.
@param matrix matrix to be reduced to upper Hessenberg form. The reduced matrix is returned via this input
*/
void
transform_matrix_to_hessenberg(matrix &mtx) {

  // apply recursive Hauseholder transformation to eliminate the matrix elements column by column
  for (size_t idx = 1; idx < mtx.rows - 1; idx++) {
    matrix reflect_vector = get_reflection_vector(mtx, idx);
    apply_householder(mtx, reflect_vector, idx);
  }
}




} // PIC
