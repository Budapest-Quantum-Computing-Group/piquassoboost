#include "dot.h"
#include <cstring>
#include <iostream>
#include "tbb/tbb.h"
#include <tbb/scalable_allocator.h>

/// The namespace of the Picasso project
namespace pic {

/**
@brief Call to calculate the product of two complex matrices by calling method zgemm3m from the CBLAS library.
@param A The first matrix.
@param B The second matrix
@param C The Pointer to the resulted matrix. The calculated matrix is returned via this pointer.
@param matrix_size The number rows in the matrices
@return Returns with zero on success.
*/
int
dot( matrix &A, matrix &B, matrix &C ) {
// openblas_set_num_threads(1)

    // parameters alpha and beta for the cblas_zgemm3m function (the input matrices are not scaled)
    Complex16 alpha(1.0, 0.0);
    Complex16 beta(0.0, 0.0);

    // check the matrix shapes in DEBUG mode
    assert( check_matrices( A, B, C ) );

    // remove memory trash from the allocated memory of the result matrix
    //memset( C.get_data(), 0, C.rows*C.cols*sizeof(Complex16) );


    // setting CBLAS transpose operations
    CBLAS_TRANSPOSE Atranspose, Btranspose;
    get_cblas_transpose( A, Atranspose );
    get_cblas_transpose( B, Btranspose );

    CBLAS_ORDER order = CblasRowMajor;

    // split the matrix product into N subblocks
    int N = (int)A.rows/150 + 1;



#if BLAS==1 // MKL
    CBLAS_TRANSPOSE Btranspose_loc;
    Complex16* B_data = B.get_data();

    // MKL does not support option CblasConjNoTrans
    if (Btranspose == CblasConjNoTrans) {
        Btranspose_loc = CblasNoTrans;
        Complex16* B_data_tmp = B.get_data();
        B_data = (Complex16*)scalable_aligned_malloc( B.cols*B.rows*sizeof(Complex16), CACHELINE);
        assert(B_data);
        vzConj( B.cols*B.rows, B_data_tmp, B_data );  // TODO: parallelize this call
    }
    else {
        B_data = B.get_data();
        Btranspose_loc = Btranspose;
    }
#else
    CBLAS_TRANSPOSE Btranspose_loc = Btranspose;
    Complex16* B_data = B.get_data();
#endif


    // call the zgemm3m function to calculate the product of the complex matrices
    tbb::parallel_for(0, N, 1, [&order, &Atranspose, &Btranspose_loc, &A, &B, &C, &alpha, &beta, B_data, N](int i) {
        int rows_in_block = A.rows/N;
        int A_offset = i*A.cols*rows_in_block;
        int C_offset = i*B.cols*rows_in_block;

        int rows_in_zgemm3m;
        if ( i < N-1 ) {
            rows_in_zgemm3m = rows_in_block;
        }
        else {
            rows_in_zgemm3m = A.rows - i*rows_in_block;

        }

#if BLAS==1 // MKL
        CBLAS_TRANSPOSE Atranspose_loc;
        Complex16* A_row;

        // MKL does not support option CblasConjNoTrans
        if (Atranspose == CblasConjNoTrans) {
            Atranspose_loc = CblasNoTrans;
            Complex16* A_row_tmp = A.get_data() + A_offset;
            A_row = (Complex16*)scalable_aligned_malloc( A.cols*rows_in_zgemm3m*sizeof(Complex16), CACHELINE);
            assert(A_row);
            vzConj( A.cols*rows_in_zgemm3m, A_row_tmp, A_row );
        }
        else {
            A_row = A.get_data() + A_offset;
            Atranspose_loc = Atranspose;
        }

        Complex16* C_row = C.get_data() + C_offset;

        cblas_zgemm3m(order, Atranspose_loc, Btranspose_loc, rows_in_zgemm3m, B.cols, A.cols, (double*)&alpha, (double*)A_row, A.cols, (double*)B_data, B.cols, (double*)&beta, (double*)C_row, B.cols);

        if (Atranspose == CblasConjNoTrans) {
            scalable_aligned_free(A_row);
            A_row = NULL;
        }

#elif BLAS==2 //OpenBLAS
        Complex16* A_row = A.get_data() + A_offset;
        Complex16* C_row = C.get_data() + C_offset;
        cblas_zgemm3m(order, Atranspose, Btranspose_loc, rows_in_zgemm3m, B.cols, A.cols, (double*)&alpha, (double*)A_row, A.cols, (double*)B_data, B.cols, (double*)&beta, (double*)C_row, B.cols);
#endif
    }); // TBB



#if BLAS==1 // MKL
        // MKL does not support option CblasConjNoTrans, so the allocated extra array for the conjugation must be released
        if (Btranspose == CblasConjNoTrans) {
            scalable_aligned_free(B_data);
            B_data = NULL;
        }

#endif

    return 0;


}



/**
@brief Call to check the shape of the matrices for method dot. (Called in DEBUG mode)
@param A The first matrix in the product of type matrix.
@param B The second matrix in the product of type matrix
@param C The resulted (preallocated) matrix of the product of type matrix
@return Returns with true if the test passed, false otherwise.
*/
bool
check_matrices( matrix &A, matrix &B, matrix &C ) {


    if (!A.is_transposed() & !B.is_transposed())  {
        if ( A.cols != B.rows ) {
            std::cout << "pic::dot:: Cols of matrix A does not match rows of matrix B!" << std::endl;
            return false;
        }
        if ( A.rows != C.rows ) {
            std::cout << "pic::dot:: Rows of matrix A does not match rows of matrix C!" << std::endl;
            return false;
        }
        if ( B.cols != C.cols ) {
            std::cout << "pic::dot:: Cols of matrix B does not match cols of matrix C!" << std::endl;
            return false;
        }
    }
    else if ( A.is_transposed() & !B.is_transposed() )  {
        if ( A.rows != B.rows ) {
            std::cout << "pic::dot:: Cols of matrix A.transpose does not match rows of matrix B!" << std::endl;
            return false;
        }
        if ( A.cols != C.rows ) {
            std::cout << "pic::dot:: Rows of matrix A.tanspose does not match rows of matrix C!" << std::endl;
            exit(-1);
        }
        if ( B.cols != C.cols ) {
            std::cout << "pic::dot:: Cols of matrix B does not match cols of matrix C!" << std::endl;
            return false;
        }
    }
    else if ( A.is_transposed() & B.is_transposed() )  {
        if ( A.rows != B.cols ) {
            std::cout << "pic::dot:: Cols of matrix A.transpose does not match rows of matrix B.transpose!" << std::endl;
            return false;
        }
        if ( A.cols != C.rows ) {
            std::cout << "pic::dot:: Rows of matrix A.transpose does not match rows of matrix C!" << std::endl;
            return false;
        }
        if ( B.rows != C.cols ) {
            std::cout << "pic::dot:: Cols of matrix B.transpose does not match cols of matrix C!" << std::endl;
            return false;
        }
    }
    else if ( !A.is_transposed() & B.is_transposed() )  {
        if ( A.cols != B.cols ) {
            std::cout << "pic::dot:: Cols of matrix A does not match rows of matrix B.transpose!" << std::endl;
            return false;
        }
        if ( A.rows != C.rows ) {
            std::cout << "pic::dot:: Rows of matrix A does not match rows of matrix C!" << std::endl;
            return false;
        }
        if ( B.rows != C.cols ) {
            std::cout << "pic::dot:: Cols of matrix B.transpose does not match cols of matrix C!" << std::endl;
            return false;
        }
    }


    // check the pointer of the matrices
    if ( A.get_data() == NULL ) {
        std::cout << "pic::dot:: No preallocated data in matrix A!" << std::endl;
        return false;
    }
    if ( B.get_data() == NULL ) {
        std::cout << "pic::dot:: No preallocated data in matrix B!" << std::endl;
        return false;
    }
    if ( C.get_data() == NULL ) {
        std::cout << "pic::dot:: No preallocated data in matrix C!" << std::endl;
        return false;
    }

    return true;

}


/**
@brief Call to get the transpose properties of the input matrix for CBLAS calculations
@param A The matrix of type matrix.
@param transpose The returned vale of CBLAS_TRANSPOSE.
*/
void
get_cblas_transpose( matrix &A, CBLAS_TRANSPOSE &transpose ) {

    if ( A.is_conjugated() & A.is_transposed() ) {
        transpose = CblasConjTrans;
    }
    else if ( A.is_conjugated() & !A.is_transposed() ) {
        transpose = CblasConjNoTrans; // not present in MKL
    }
    else if ( !A.is_conjugated() & A.is_transposed() ) {
        transpose = CblasTrans;
    }
    else {
        transpose = CblasNoTrans;
    }

}



} //PIC
