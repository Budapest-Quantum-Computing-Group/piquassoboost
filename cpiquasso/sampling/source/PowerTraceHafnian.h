#ifndef PowerTraceHafnian_H
#define PowerTraceHafnian_H

#include "matrix.h"


namespace pic {



/**
@brief Class to calculate the hafnian of a complex matrix by the power trace method
*/
class PowerTraceHafnian {

protected:
    /// The covariance matrix of the Gaussian state.
    matrix mtx;


public:


/**
@brief Default constructor of the class.
@param mtx_in The covariance matrix of the Gaussian state.
@return Returns with the instance of the class.
*/
PowerTraceHafnian( matrix &mtx_in );

/**
@brief Default destructor of the class.
*/
virtual ~PowerTraceHafnian();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate();


/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void Update_mtx( matrix &mtx_in);


protected:

/**
@brief Call to calculate the power traces \f$Tr(mtx^j)~\forall~1\leq j\leq l\f$ for a squared complex matrix \f$mtx\f$ of dimensions \f$n\times n\f$.
@param mtx an instance of class matrix.
@param pow_max maximum matrix power when calculating the power trace.
@return a vector containing the power traces of matrix `z` to power \f$1\leq j \leq l\f$.
*/
matrix calc_power_traces(matrix &mtx, size_t pow_max);


/**
@brief Call to determine the first \f$ k \f$ coefficients of the characteristic polynomial using the Algorithm 2 of LaBudde method.
 See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1) for further details.
@param mtx matrix in upper Hessenberg form.
@param highest_order the order of the highest order coefficient to be calculated (k <= n)
@return Returns with the calculated coefficients of the characteristic polynomial.
 *
 */
matrix calc_characteristic_polynomial_coeffs(matrix &mtx, size_t highest_order);


/**
@brief Call to calculate the traces of \f$ A^{p}\f$, where 1<=p<=pow is an integer and A is a square matrix.
The trace is calculated from the coefficients of its characteristic polynomial.
In the case that the power p is above the size of the matrix we can use an optimization described in Appendix B of [arxiv:1805.12498](https://arxiv.org/pdf/1104.3769v1.pdf)
@param coeffs matrix containing the characteristic polynomial coefficients
@param pow the maximal exponent p
@return Returns with the calculated power traces
 */
matrix powtrace_from_charpoly(matrix &coeffs, size_t pow);

/**
@brief Reduce a general matrix to upper Hessenberg form.
@param matrix matrix to be reduced to upper Hessenberg form. The reduced matrix is returned via this input
*/
void transform_matrix_to_hessenberg(matrix &mtx);


/**
@brief Apply householder transformation on a matrix A' = (1 - 2*v o v/v^2) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
@param offset Starting index (i.e. offset index) of rows/columns from which the householder transformation should be applied
*/
void apply_householder(matrix &A, matrix &v, size_t offset);

/**
/@brief Determine the reflection vector for Householder transformation used in the upper Hessenberg transformation algorithm
@param mtx The matrix instance on which the Hessenberg transformation should be applied
@param offset Starting index (i.e. offset index) of rows/columns from which the householder transformation should be applied
@return Returns with the calculated reflection vector
 */
matrix get_reflection_vector(matrix &mtx, size_t offset);

}; //PowerTraceHafnian


} // PIC

#endif
