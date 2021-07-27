#ifndef CONSTANTS_TESTS_H
#define CONSTANTS_TESTS_H

// enabling asserts
#ifdef NDEBUG
#undef NDEBUG
#endif

#ifndef DEBUG
#define DEBUG
#endif


#include "dot.h"

namespace pic {


constexpr double epsilon = 1E-10;



enum RandomMatrixType
{
    // fully random elements
    RANDOM,
    // fully random elements symmetrically
    SYMMETRIC,
    // selfadjoint matrix
    SELFADJOINT,
    // selfadjoint, positive definite matrix
    POSITIVE_DEFINIT,
    // lower triangular with fully random elements
    LOWER_TRIANGULAR,
    // upper triangular with fully random elements
    UPPER_TRIANGULAR
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

template<class matrix_type, class complex_type>
bool
is_identity_matrix(matrix_type& matrix)
{
    for (size_t idx = 0; idx < matrix.rows; idx++){
        for (size_t jdx = 0; jdx < matrix.cols; jdx++){
            if (idx == jdx){
                if ( std::abs(matrix[idx * matrix.stride + jdx] - complex_type(1.0) ) > epsilon ){
                    return false;
                }
                
            }else{
                if ( std::abs(matrix[idx * matrix.stride + jdx]) > epsilon ){
                    return false;
                }
            }
        }
    }
    return true;
}

} // PIC

#endif
