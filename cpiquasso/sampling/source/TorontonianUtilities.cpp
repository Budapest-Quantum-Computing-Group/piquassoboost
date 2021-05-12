#include "TorontonianUtilities.h"
#include "calc_cholesky_decomposition_AVX.h"
#include <iostream>
#include "common_functionalities.h"
#include <math.h>


static tbb::spin_mutex my_mutex;
/*
double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {

/*
// Cholesky decomposition
// Works for selfadjoint positive definite matrices!
// Basic version: no block matrices used.
template<class matrix, class Complex16>
void
calc_cholesky_decomposition(matrix& matrix)
{
    // storing in the same memory the results of the algorithm
    int n = matrix.cols;


    Complex16* row_i = matrix.get_data();
    row_i[0] = sqrt(row_i[0]);

    // Decomposing a matrix into lower triangular matrices
    for (int i = 1; i < n; i++) {

        row_i = row_i + matrix.stride;

        Complex16* row_j = matrix.get_data();
        Complex16* row_j2 = row_j + matrix.stride;

        for (int j = 0; j < i-1; j=j+2) {

            Complex16 sum = 0;
            Complex16 sum2 = 0;
            // Evaluating L(i, j) using L(j, j)
            // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
            for (int k = 0; k < j-1; k=k+2){
                int k2 = k+1;
                sum += mult_a_bconj( row_i[k], row_j[k]) + mult_a_bconj( row_i[k2], row_j[k2]);
                sum2 += mult_a_bconj( row_i[k], row_j2[k]) + mult_a_bconj( row_i[k2], row_j2[k2]);
            }

            if (j%2 == 1) {

                int k = j-1;

                sum += mult_a_bconj( row_i[k], row_j[k]);
                sum2 += mult_a_bconj( row_i[k], row_j2[k]);

            }

            row_i[j] = (row_i[j] - sum) / row_j[j];

            sum2 += mult_a_bconj( row_i[j], row_j2[j]);
            row_i[j+1] = (row_i[j+1] - sum2) / row_j2[j+1];

            row_j = row_j + 2*matrix.stride;
            row_j2 = row_j2 + 2*matrix.stride;

        }



        if ( i%2 == 1) {
            int j = i-1;

            row_j = matrix.get_data() + j * matrix.stride;

            Complex16 sum = 0;
            // Evaluating L(i, j) using L(j, j)
            // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
            for (int k = 0; k < j; k++){
                sum += mult_a_bconj( row_i[k], row_j[k]);
            }

            row_i[j] = (row_i[j] - sum) / row_j[j];

#ifdef DEBUG
            if (matrix.isnan()) {

                std::cout << "matrix is NAN" << std::endl;
                matrix.print_matrix();
                exit(-1);
             }
#endif
        }



        Complex16 sum = 0;
        // summation for diagonals
        // L_{j,j}=\sqrt{A_{j,j}-\sum_{k=0}^{j-1}L_{j,k}L_{j,k}^*}
        for (int k = 0; k < i-1; k=k+2){
            sum += mult_a_bconj( row_i[k], row_i[k] ) + mult_a_bconj( row_i[k+1], row_i[k+1] );
        }

        if ( i%2 == 1) {
            sum += mult_a_bconj( row_i[i-1], row_i[i-1] );
        }


        row_i[i] = sqrt(row_i[i] - sum);


    }

}
*/

// Cholesky decomposition
// Works for selfadjoint positive definite matrices!
// Basic version: no block matrices used.
void
calc_cholesky_decomposition(matrix& matrix)
{

#ifdef USE_AVX

    calc_cholesky_decomposition_AVX( matrix );
    return;

#else

    // storing in the same memory the results of the algorithm
    size_t n = matrix.cols;
    // Decomposing a matrix into lower triangular matrices
    for (int i = 0; i < n; i++) {
        Complex16* row_i = matrix.get_data()+i*matrix.stride;
        for (int j = 0; j < i; j++) {
            {
                Complex16* row_j = matrix.get_data()+j*matrix.stride;

                Complex16 sum = 0;
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
        Complex16 sum = 0;
        // summation for diagnols
        // L_{j,j}=\sqrt{A_{j,j}-\sum_{k=0}^{j-1}L_{j,k}L_{j,k}^*}
        for (int k = 0; k < i; k++){
            sum += mult_a_bconj( row_i[k], row_i[k] );
        }
        row_i[i] = sqrt(row_i[i] - sum);
    }

    return;


#endif // USE_AVX

}



void
calc_cholesky_decomposition_lapack(matrix &matrix) {


// transform the matrix mtx into an upper Hessenberg format by calling lapack function
        char UPLO = 'L';
        int N = matrix.rows;
        int LDA = matrix.stride;
        int INFO = 0;

        //std::cout<<"Before lapacke call:\n";
        //mtx.print_matrix();


        LAPACKE_zpotrf(LAPACK_ROW_MAJOR, UPLO, N, matrix.get_data(), LDA);

        return;

}


// calculating determinant based on cholesky decomposition
Complex16
calc_determinant_cholesky_decomposition(matrix& mtx){
    // for small matrices nothing has to be casted into quad precision

//    if (mtx.rows <= 10) {
        calc_cholesky_decomposition(mtx);

        Complex16 determinant(1.0, 0.0);

        for (size_t idx; idx < mtx.cols; idx++){
            Complex16 &elem = mtx[idx * mtx.stride + idx];
            determinant *= elem;
        }

        return mult_a_bconj( determinant, determinant);
/*    }
    // The lapack function to calculate the Cholesky decomposition is more efficient for larger matrices, but for above a given cutoff quad precision is needed
    // for these matrices of moderate size, the coefficients of the characteristic polynomials are casted into quad precision and the traces are calculated in
    // quad precision
    else {//if ( mtx.rows < 30 ) {


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

        Complex16 det = Complex16(1,0);

        for (size_t idx=0; idx<N; idx++) {
            det *= mtx[idx * mtx.stride + idx];
        }
        return mult_a_bconj( det, det);
  //  }
  /*
    else{
        // above a treshold matrix size all the calculations are done in quad precision
        // matrix size for which quad precision is necessary
        return Complex16(1.0, 0.0);
    }
*/
}






} // PIC


