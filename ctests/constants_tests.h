#ifndef CONSTANTS_TESTS_H
#define CONSTANTS_TESTS_H


#include "dot.h"

namespace pic {


constexpr double epsilon = 1E-10;



enum RandomMatrixType
{
    RANDOM,
    SYMMETRIC,
    SELFADJOINT,
    POSITIVE_DEFINIT
};


template<class matrix_type>
matrix_type
matrix_conjugate_traspose(matrix_type& matrix)
{
    matrix_type transposed(matrix.cols, matrix.rows);

    for (size_t i = 0; i < matrix.rows; i++){
        for (size_t j = 0; j < matrix.cols; j++){
            transposed[j * transposed.stride + i].real(matrix[i * matrix.stride + j].real());
            transposed[j * transposed.stride + i].imag(-matrix[i * matrix.stride + j].imag());
        }
    }
    return transposed;
}


// returns a random matrix of the given type:
// RANDOM : fully random
// SYMMETRIC : random complex symmetric matrix
// SELFADJOINT : random selfadjoint (hermitian) matrix
template<class matrix_type, class complex_type>
matrix_type 
getRandomMatrix(size_t n, pic::RandomMatrixType type){
    matrix_type mtx(n, n);

    // initialize random generator as a standard normal distribution generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<long double> distribution(0.0, 1.0);

    if (type == pic::RANDOM){
        // fill up matrix with fully random elements
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = 0; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                long double randnum2 = distribution(generator);
                mtx[row_idx * n + col_idx] = complex_type(randnum1, randnum2);
            }
        }
    }else if (type == pic::SYMMETRIC){
        // fill up matrix with fully random elements symmetrically
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                long double randnum2 = distribution(generator);
                mtx[row_idx * n + col_idx] = complex_type(randnum1, randnum2);
                mtx[col_idx * n + row_idx] = complex_type(randnum1, randnum2);
            }
        }
    }else if (type == pic::SELFADJOINT){
        // hermitian case, selfadjoint matrix
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                if (row_idx == col_idx){
                    mtx[row_idx * n + col_idx] = complex_type(randnum1, 0);
                }else{
                    long double randnum2 = distribution(generator);
                    mtx[row_idx * n + col_idx] = complex_type(randnum1, randnum2);
                    mtx[col_idx * n + row_idx] = complex_type(randnum1, -randnum2);
                }
            }
        }
    }else if (type == pic::POSITIVE_DEFINIT){
        // hermitian case, selfadjoint, positive definite matrix
        // if you have a random matrix M then M * M^* gives you a positive definite hermitian matrix
        matrix_type mtx1 = getRandomMatrix<matrix_type, complex_type>(n, pic::RANDOM);
        matrix_type mtx2 = matrix_conjugate_traspose<matrix_type>(mtx1);

        mtx = pic::dot(mtx1, mtx2);

        for (size_t i = 0; i < n; i++){
            for (size_t j = 0; j < n; j++){
                mtx[i * n + j] /= n;
            }    
            mtx[i * n + i] += complex_type(1.0,0.0);
        }
        
    }
    return mtx;
}

} // PIC

#endif
