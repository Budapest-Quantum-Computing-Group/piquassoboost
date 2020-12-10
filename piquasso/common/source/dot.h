#ifndef Dot_H
#define Dot_H


#include "matrix.h"

#ifdef __cplusplus
extern "C"
{
#endif
typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;

/// Definition of the zgemm3m function from CBLAS
void cblas_zgemm3m(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const void *alpha, const void *A, const int lda, const void *B, const int ldb, const void *beta, void *C, const int ldc);

#if BLAS==1 // MKL
/// Calculate the element-wise complex conjugate of a vector
void vzConj(int num, pic::Complex16* input, pic::Complex16* output);
#endif

#ifdef __cplusplus
}
#endif



/// The namespace of the Picasso project
namespace pic {


/**
@brief Call to calculate the product of two complex matrices by calling method zgemm3m from the CBLAS library.
@param A The first matrix in the product of type matrix.
@param B The second matrix in the product of type matrix
@param C The resulted (preallocated) matrix of the product of type matrix
@return Returns with zero on success.
*/
int dot( matrix &A, matrix &B, matrix &C );



/**
@brief Call to check the shape of the matrices for method dot. (Called in DEBUG mode)
@param A The first matrix in the product of type matrix.
@param B The second matrix in the product of type matrix
@param C The resulted (preallocated) matrix of the product of type matrix
@return Returns with true if the test passed, false otherwise.
*/
bool check_matrices( matrix &A, matrix &B, matrix &C );



/**
@brief Call to get the transpose properties of the input matrix for CBLAS calculations
@param A The matrix of type matrix.
@param transpose The returned vale of CBLAS_TRANSPOSE.
*/
void get_cblas_transpose( matrix &A, CBLAS_TRANSPOSE &transpose );

}

#endif
