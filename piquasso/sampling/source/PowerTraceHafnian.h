#ifndef PowerTraceHafnian_H
#define PowerTraceHafnian_H

#include "matrix.h"


namespace pic {



/**
@brief Class representing a matrix Chin-Huh permanent calculator
*/
class PowerTraceHafnian {

protected:
    /// The effective scattering matrix of a boson sampling instance
    matrix mtx;


public:


/**
@brief Default constructor of the class.
@param mtx_in The effective scattering matrix of a boson sampling instance
@return Returns with the instance of the class.
*/
PowerTraceHafnian( matrix &mtx_in );

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
Complex16 calculate();


/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void Update_mtx( matrix &mtx_in);


private:

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
@param highest_orde the order of the highest order coefficient to be calculated (k <= n)
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



}; //PowerTraceHafnian


} // PIC

#endif
