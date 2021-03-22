#ifndef Dot_H
#define Dot_H


#include "matrix.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;

/// Definition of the zgemm function from CBLAS
void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
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
@return Returns with the resulted matrix.
*/
matrix dot( matrix &A, matrix &B );



/**
@brief Call to check the shape of the matrices for method dot. (Called in DEBUG mode)
@param A The first matrix in the product of type matrix.
@param B The second matrix in the product of type matrix
@return Returns with true if the test passed, false otherwise.
*/
bool check_matrices( matrix &A, matrix &B );



/**
@brief Call to get the transpose properties of the input matrix for CBLAS calculations
@param A The matrix of type matrix.
@param transpose The returned vale of CBLAS_TRANSPOSE.
*/
void get_cblas_transpose( matrix &A, CBLAS_TRANSPOSE &transpose );

// relieve Python extension from TBB functionalities
#ifndef CPYTHON

/**
@brief Structure containing row limits for the partitioning of the matrix product calculations.
(the partitioning follows the divide-and-conquer strategy)
*/
struct row_indices {

    /// The firs row in matrix A participating in the multiplication sub-problem.
    size_t Arows_start;
    /// The last row in matrix A participating in the multiplication sub-problem. (The rows are picked from a closed-open range [Arows_start, Arows_end) )
    size_t Arows_end;
    /// The number of rows in matrix A participating in the multiplication sub-problem.
    size_t Arows;
    /// The firs row in matrix B participating in the multiplication sub-problem.
    size_t Brows_start;
    /// The last row in matrix B participating in the multiplication sub-problem. (The rows are picked from a closed-open range [Brows_start, Brows_end) )
    size_t Brows_end;
    /// The number of rows in matrix B participating in the multiplication sub-problem.
    size_t Brows;
    /// The firs row in matrix C participating in the multiplication sub-problem.
    size_t Crows_start;
    /// The last row in matrix C participating in the multiplication sub-problem. (The rows are picked from a closed-open range [Crows_start, Crows_end) )
    size_t Crows_end;
    /// The number of rows in matrix C participating in the multiplication sub-problem.
    size_t Crows;
};

/**
@brief Structure containing column limits for the partitioning of the matrix product calculations.
(the partitioning follows the divide-and-conquer strategy)
*/
struct col_indices {
    /// The firs col in matrix A participating in the multiplication sub-problem.
    size_t Acols_start;
    /// The last col in matrix A participating in the multiplication sub-problem. (The cols are picked from a closed-open range [Acols_start, Acols_end) )
    size_t Acols_end;
    /// The number of cols in matrix A participating in the multiplication sub-problem.
    size_t Acols;
    /// The firs col in matrix B participating in the multiplication sub-problem.
    size_t Bcols_start;
    /// The last col in matrix B participating in the multiplication sub-problem. (The cols are picked from a closed-open range [Bcols_start, Bcols_end) )
    size_t Bcols_end;
    /// The number of cols in matrix B participating in the multiplication sub-problem.
    size_t Bcols;
    /// The firs col in matrix C participating in the multiplication sub-problem.
    size_t Ccols_start;
    /// The last col in matrix C participating in the multiplication sub-problem. (The col are picked from a closed-open range [Ccols_start, Ccols_end) )
    size_t Ccols_end;
    /// The number of cols in matrix C participating in the multiplication sub-problem.
    size_t Ccols;
};




/**
@brief Class to calculate a matrix product C=A*B in serial. This class is used to divide the multiplication into chunks for parallel calculations.
*/
class zgemm_Task_serial {

public:
    /// The matrix A
    matrix A;
    /// The matrix B
    matrix B;
    /// The matrix C
    matrix C;
    /// Structure containing row limits for the partitioning of the matrix product calculations.
    row_indices rows;
    /// Structure containing column limits for the partitioning of the matrix product calculations.
    col_indices cols;
    /// CBLAS storage order
    CBLAS_ORDER order;




public:


/**
@brief Constructor of the class. (In this case the row/col limits are extracted from matrices A,B,C).
@param A_in The object representing matrix A.
@param B_in The object representing matrix B.
@param C_in The object representing matrix C.
*/
zgemm_Task_serial( matrix &A_in, matrix &B_in, matrix &C_in);

/**
@brief Constructor of the class.
@param A_in The object representing matrix A.
@param B_in The object representing matrix B.
@param C_in The object representing matrix C.
@param rows_in Structure containing row limits for the partitioning of the matrix product calculations.
@param cols_in Structure containing column limits for the partitioning of the matrix product calculations.
*/
zgemm_Task_serial( matrix &A_in, matrix &B_in, matrix &C_in, row_indices& rows_in, col_indices& cols_in);

/**
@brief Call to calculate the product of matrix chunks defined by attributes rows, cols. The result is stored in the corresponding chunk of matrix C.
*/
void zgemm_chunk();

}; // zgemm_Task_serial



/**
@brief Class to calculate a matrix product C=A*B in parallel. The parallelism follow the strategy of divide-and-conquer.
*/
class zgemm_Task: public zgemm_Task_serial {



public:


/**
@brief Constructor of the class. (In this case the row/col limits are extracted from matrices A,B,C).
@param A_in The object representing matrix A.
@param B_in The object representing matrix B.
@param C_in The object representing matrix C.
*/
zgemm_Task( matrix &A_in, matrix &B_in, matrix &C_in);

/**
@brief Constructor of the class.
@param A_in The object representing matrix A.
@param B_in The object representing matrix B.
@param C_in The object representing matrix C.
@param rows_in Structure containing row limits for the partitioning of the matrix product calculations.
@param cols_in Structure containing column limits for the partitioning of the matrix product calculations.
*/
zgemm_Task( matrix &A_in, matrix &B_in, matrix &C_in, row_indices& rows_in, col_indices& cols_in);



/**
@brief This function is called when a task is spawned. It divides the work into chunks following the strategy of divide-and-conquer until the problem size meets a predefined treshold.
@return Returns with a pointer to a tbb::task instance or with a null pointer.
*/
void execute(tbb::task_group &g);



}; // zgemm_Task


#endif // CPYTHON

}; // PIC

#endif //Dot_H
