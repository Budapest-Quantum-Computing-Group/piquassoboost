#ifndef TorontonianRecursive_H
#define TorontonianRecursive_H

#include "Torontonian.h"
#include "PicState.h"
#include "PicVector.hpp"
#include "matrix_real.h"
#include "matrix_real16.h"



namespace pic {


class TorontonianRecursive  {


protected:

    /// The input matrix. Must be selfadjoint positive definite matrix with eigenvalues between 0 and 1.
    matrix_real mtx_orig;
    /** The scaled input matrix for which the calculations are performed.
    If the mean magnitude of the matrix elements is one, the treshold of quad precision can be set to higher values.
    */
    matrix_real mtx;

public:

/**
@brief Constructor of the class.
@param mtx_in A selfadjoint matrix for which the torontonian is calculated. This matrix has to be positive definite matrix with eigenvalues between 0 and 1 (for example a covariance matrix of the Gaussian state.) matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state)
@return Returns with the instance of the class.
*/
TorontonianRecursive( matrix_real &mtx_in );


/**
@brief Default destructor of the class.
*/
virtual ~TorontonianRecursive();

/**
@brief Call to calculate the hafnian of a complex matrix
@param use_extended Logical variable to indicate whether use extended precision for cholesky decomposition (default), or not.
@return Returns with the calculated hafnian
*/
virtual double calculate(bool use_extended);

/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
virtual void Update_mtx( matrix_real &mtx_in);


}; //TorontonianRecursive




} // PIC

#endif
