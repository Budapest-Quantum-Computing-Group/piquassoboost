#ifndef TORONTONIAN_UTILITIES_HPP_INCLUDED
#define TORONTONIAN_UTILITIES_HPP_INCLUDED

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


// Cholesky decomposition
// Works for selfadjoint positive definite matrices!
// Basic version: no block matrices used.
template<class matrix_type, class complex_type>
void
calc_cholesky_decomposition(matrix_type& matrix)
{
    // storing in the same memory the results of the algorithm
    size_t n = matrix.cols;
    // Decomposing a matrix into lower triangular matrices
    for (int i = 0; i < n; i++) {
        complex_type* row_i = matrix.get_data()+i*matrix.stride;
        for (int j = 0; j < i; j++) {
            {
                complex_type* row_j = matrix.get_data()+j*matrix.stride;

                complex_type sum = 0;                
                // Evaluating L(i, j) using L(j, j)
                // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
                for (int k = 0; k < j; k++){
                    sum += mult_a_bconj( row_i[k], row_j[k]);
                }
                //std::cout << "L_("<<i<<","<<j<<") : sum: " << sum<<std::endl;
                //std::cout << "L_("<<i<<","<<j<<") : mult: " << mult_a_bconj( row_i[j-1], row_j[j-1])<<std::endl;
                
                row_i[j] = (row_i[j] - sum) / row_j[j];
                //std::cout << "L_("<<i<<","<<j<<") : value: " << row_i[j]<<std::endl;
            }
        }
        complex_type sum = 0;
        // summation for diagnols
        // L_{j,j}=\sqrt{A_{j,j}-\sum_{k=0}^{j-1}L_{j,k}L_{j,k}^*}
        for (int k = 0; k < i; k++){
            sum += mult_a_bconj( row_i[k], row_i[k] );
        }
        row_i[i] = sqrt(row_i[i] - sum);
    }


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
                    sum += mult_a_bconj(lower_tridiag[j*lower_tridiag.stride+k], lower_tridiag[j*lower_tridiag.stride+k]);
                }
                lower_tridiag[j*lower_tridiag.stride+j] = sqrt(matrix[j*matrix.stride+j] -
                                        sum);
                //std::cout << "("<<i<<","<<j<<"): "<<lower_tridiag[i*lower_tridiag.stride+j]<<std::endl;
            } else {
                // Evaluating L(i, j) using L(j, j)
                // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
                for (int k = 0; k < j; k++){
                    sum += mult_a_bconj(lower_tridiag[i*lower_tridiag.stride+k], lower_tridiag[j*lower_tridiag.stride+k]);
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


template<class matrix_type, class complex_type>
matrix_type
calc_inverse_of_lower_triangular_matrix_adjoint(matrix_type &L11){
    // assuming L11.rows == L11.cols
    size_t dim = L11.rows;

    // inverse of L11^*
    // columns and rows are changed into each other
    matrix_type inverse_of_L11(dim, dim);
    for (size_t row_idx = 0; row_idx < dim; row_idx++){
        complex_type *inverse_row = inverse_of_L11.get_data() + row_idx * inverse_of_L11.stride;

        complex_type &diag_elem = L11[row_idx * L11.stride + row_idx];
        double norm_diag = diag_elem.real() * diag_elem.real() + diag_elem.imag() * diag_elem.imag();
        
        //inverse_of_L11[row_idx * inverse_of_L11.stride + row_idx] = diag_elem / norm_diag;
        inverse_row[row_idx] = diag_elem / norm_diag;
        
        /*
        complex_type one(1.0, 0.0);
        complex_type div = complex_type(1.0D) / L11[row_idx * L11.stride + row_idx];
        inverse_of_L11[row_idx * inverse_of_L11.stride + row_idx] = mult_a_bconj( one, div);
        */

        for (size_t col_idx = 0; col_idx < row_idx; col_idx++){
            inverse_row[col_idx] = complex_type(0.0, 0.0);
        }

        for (size_t col_idx = row_idx + 1; col_idx < inverse_of_L11.cols; col_idx++){
            complex_type summand(0.0, 0.0);
            complex_type *row_L11 = L11.get_data() + col_idx * L11.stride;
            for (size_t k = row_idx; k < col_idx; k++){
                // L11 cols and rows the other way around and conjugate
                //summand += mult_a_bconj(inverse_row[k], L11[col_idx * L11.stride + k]);
                summand += mult_a_bconj(inverse_row[k], row_L11[k]);
            }
            // L11 cols and rows the other way around            
            complex_type &b = row_L11[col_idx];
            double norm = b.real() * b.real() + b.imag() * b.imag();

            inverse_row[col_idx] = summand * b / -norm;
            /*
            complex_type divisor = complex_type(-1.0D) / L11[col_idx * L11.stride + col_idx];
            inverse_of_L11[row_idx * inverse_of_L11.stride + col_idx] =
                mult_a_bconj( summand, divisor );
            */
        }
    }

    return inverse_of_L11;
}

// Calculating A21 * L11^*^-1
template<class matrix_type, class complex_type>
void
update_first_block(matrix_type &A21, matrix_type &L11){

    // inverse of L11^*
    // columns and rows are changed into each other
    matrix_type inverse_of_L11 = calc_inverse_of_lower_triangular_matrix_adjoint<matrix_type, complex_type>(L11);
    //std::cout << "inverse: "<<std::endl;
    //inverse_of_L11.print_matrix();

    //std::cout << "Inverse matrix we update with:" << std::endl;
    //inverse_of_L11.print_matrix();

    // updating with L11 inverse
    matrix_type product = dot(A21, inverse_of_L11);

    //std::cout << "Copying data from product ("<< product.rows<<","<<product.cols<<")"<<std::endl;
    //std::cout << "to A21 ("<< A21.rows<<","<<A21.cols<<")"<<std::endl;
    for (size_t row_idx = 0; row_idx < product.rows; row_idx++){
        std::memcpy(A21.get_data() + row_idx * A21.stride, product.get_data() + row_idx * product.stride, product.cols * sizeof(complex_type));
    }
}

/*
                   DO 360 k = 1,n
                       IF (nounit) THEN
                           temp = one/a(k,k)
                           DO 320 i = 1,m
                               b(i,k) = temp*b(i,k)
   320                     CONTINUE
                       END IF
                       DO 340 j = k + 1,n
                           IF (a(j,k).NE.zero) THEN
                               temp = a(j,k)
                               DO 330 i = 1,m
                                   b(i,j) = b(i,j) - temp*b(i,k)
   330                         CONTINUE
                           END IF
   340                 CONTINUE
                       IF (alpha.NE.one) THEN
                           DO 350 i = 1,m
                               b(i,k) = alpha*b(i,k)
   350                     CONTINUE
                       END IF
   360             CONTINUE
*/
// Calculating A21 * L11^*^-1
template<class matrix_type, class complex_type>
void
//                                  b                  a
update_first_block_based_on_LAPACKE(matrix_type &A21, matrix_type &L11){
    size_t n = A21.cols;
    size_t m = A21.rows;
    //std::cout << "A21: "<< m << " x "<<n<<std::endl;
    //std::cout << "L11: "<< L11.rows << " x "<<L11.cols<<std::endl;

    // k = 1,n
    for (size_t k = 0; k < n; k++){
        // temp = one/a(k,k)
        complex_type temp = complex_type(1.0) / L11[k*L11.stride + k];
        // i = 1,m
        for (size_t i = 0; i < m; i++){
            // b(i,k) = temp*b(i,k)
            A21[i*A21.stride+k] *= temp;
        }
        for (size_t j = k+1; j < n; j++){
            // temp = a(j,k)
            complex_type &temp = L11[j*L11.stride+k];
            for (size_t i = 0; i < m; i++){
                // b(i,j) = b(i,j) - temp*b(i,k)
                A21[i * A21.stride + j] -= mult_a_bconj( A21[i*A21.stride+k], temp );
            }
        }
    }
}




// Calculating A21 * L11^*^-1
// B = X * L
// Algorithm calculates matrix X from B and L matrices, where L is upper triangular
// b(0,0) = sum(k=0,t) x(0,k)*l(k,0) = x(0,0) * l(0,0)
// b(0,1) = sum(k=0,t) x(0,k)*l(k,1) = x(0,0) * l(0,1) + x(0,1) * l(1,1)
// b(0,j) = sum(k=0,t) x(0,k)*l(k,j)
// b(i,0) = sum(k=0,t) x(i,k)*l(k,0) = x(i,0) * l(0,0)
// b(i,j) = sum(k=0,t) x(i,k)*l(k,j)
// from this we get:
// x(0,0) = b(0,0) / l(0,0)
// x(0,1) = b(0,1) - x(0,0) * l(0,1)
// x(0,j) = ( b(0,j) - sum(k=0,j-1) x(0,k)*l(k,j) ) / l(j,j)
// x(i,0) = b(i,0) / l(0,0)
// x(i,1) = b(i,1) - x(i,0) * l(0,1)
// x(i,j) = ( b(i,j) - sum(k=0,j-1) x(i,k)*l(k,j) ) / l(j,j)
template<class matrix_type, class complex_type>
void
// x = new A21
// b = old A21
// l = L11 conjugated transpose
update_first_block_rowwise(matrix_type &A21, matrix_type &L11){
    size_t cols = A21.cols;
    size_t rows = A21.rows;
    
    // i = 0, i < rows
    for (size_t i = 0; i < rows; i++){
        // x(i,0) = b(i,0) / l(0,0)
        complex_type *row_A21_i = A21.get_data() + i * A21.stride;
        row_A21_i[0] /= L11[0];
        // j = 1, j < cols
        for (size_t j = 1; j < cols; j++){
            complex_type *row_L11_j = L11.get_data() + j * L11.stride;
            // A) sum = b(i,j)
            //complex_type sum = A21[i*A21.stride+j];

            complex_type &elem = row_A21_i[j];
            // k = 0, k < j
            for (size_t k = 0; k < j; k++){
                // L(k,j) = L^*(j,k) conjugated
                // A) sum = sum - x(i,k) * l(k,j)
                // B) b(i,j) = b(i,j) - x(i,k) * l(k,j) 
                elem -= mult_a_bconj( row_A21_i[k], row_L11_j[k] );
            }
            // A) b(i,j) = sum / l(j,j)
            // B) b(i,j) = b(i,j) / l(j,j)
            elem /= row_L11_j[j];
        }
    }
}



// A22' = A22 - L21 * L21^*
template<class matrix_type, class complex_type>
void
update_second_block(matrix_type &A22, matrix_type &L21){
    size_t rows = A22.rows;
    size_t cols = A22.cols;
    for (size_t row_idx = 0; row_idx < rows; row_idx++){
        complex_type *row_A22 = A22.get_data() + row_idx * A22.stride;
        for (size_t col_idx = 0; col_idx < cols; col_idx++){
            complex_type *row_L21_row_idx = L21.get_data() + row_idx * L21.stride;
            complex_type *row_L21_col_idx = L21.get_data() + col_idx * L21.stride;

            complex_type &elem = row_A22[col_idx];
            for (size_t k = 0; k < L21.cols; k++){
                elem -= mult_a_bconj( row_L21_row_idx[k], row_L21_col_idx[k] );
            }
        }    
    }
}


// Cholesky decomposition
// Works only for selfadjoint positive definite matrices!
template<class matrix_type, class complex_type>
void
calc_cholesky_decomposition_block_based(matrix_type &matrix, size_t size_of_first_block)
{
    // storing in the same memory the results of the algorithm
    //const size_t n = matrix.cols;

    // Algorithm based on http://www.netlib.org/utk/papers/factor/node9.html
    // For any k :
    // Calculate the L_kk from A_kk:
    //   L11 = \sqrt(A11)
    //   L21 = A21 * L11^-1
    //   A22' = A22 - L21 * L21^*
    // call recursive function on L21

    // First block size has to be at most equal to the dimension of the matrix
    size_of_first_block = std::min(matrix.cols, size_of_first_block);

    // Second block size has to be at least zero
    size_t size_of_second_block;
    if (matrix.cols > size_of_first_block){
        size_of_second_block = matrix.cols - size_of_first_block;
    }else{
        size_of_second_block = 0;
    }
    //std::cout << "1st b: " << size_of_first_block << " 2nd b: " << size_of_second_block << std::endl;
    matrix_type A11(
        matrix.get_data(),
        size_of_first_block,
        size_of_first_block,
        matrix.stride);

    //std::cout << "Original matrix:" << std::endl;
    //matrix.print_matrix();

    // L11 calculated based on standard Cholesky decomposition
    calc_cholesky_decomposition<matrix_type, complex_type>(A11);
    
    if (size_of_second_block == 0){
        return;
    }

    matrix_type A21(
        matrix.get_data() + size_of_first_block*matrix.stride,
        size_of_second_block,
        size_of_first_block,
        matrix.stride);
    matrix_type A22(
        matrix.get_data() + size_of_first_block*matrix.stride + size_of_first_block,
        size_of_second_block,
        size_of_second_block,
        matrix.stride);
    

    //std::cout << "======================================"<<std::endl;
    
    //std::cout << "1st b: " << size_of_first_block << " 2nd b: " << size_of_second_block << std::endl;
    //std::cout << "After decomposing A11:" << std::endl;
    //matrix.print_matrix();

    //   L21 = A21 * L11^*^-1
    update_first_block_rowwise<matrix_type, complex_type>(A21, A11);

    //std::cout << "After updating A21:" << std::endl;
    //matrix.print_matrix();

    //   A22' = A22 - L21 * L21^*
    update_second_block<matrix_type, complex_type>(A22, A21);

    //std::cout << "After updating A22:" << std::endl;
    //matrix.print_matrix();

    // call recursive function on L21
    calc_cholesky_decomposition_block_based<matrix_type, complex_type>(A22, size_of_first_block);
}


// calculating determinant based on cholesky decomposition
template<class matrix_type, class complex_type>
complex_type
calc_determinant_cholesky_decomposition(matrix& mtx){
    // for small matrices nothing has to be casted into quad precision
    if (mtx.rows <= 10) {
        calc_cholesky_decomposition<matrix, Complex16>(mtx);

        Complex16 determinant(1.0, 0.0);

        for (size_t idx; idx < mtx.cols; idx++){
            Complex16 &elem = mtx[idx * mtx.stride + idx];
            determinant *= elem;
        }
        
        return mult_a_bconj( determinant, determinant);
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

        complex_type det = complex_type(1,0);

        for (size_t idx=0; idx<N; idx++) {
            det *= mtx[idx * mtx.stride + idx];
        }
        return mult_a_bconj( det, det);
    }
    else{
        // above a treshold matrix size all the calculations are done in quad precision
        // matrix size for which quad precision is necessary
        return complex_type(1.0, 0.0);
    }

}






} // PIC


#endif // TORONTONIAN_UTILITIES_HPP_INCLUDED