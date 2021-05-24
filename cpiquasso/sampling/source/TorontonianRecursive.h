#ifndef TorontonianRecursive_H
#define TorontonianRecursive_H

#include "Torontonian.h"
#include "PicState.h"
#include "PicVector.hpp"



namespace pic {

/**
@brief Wrapper class to calculate the hafnian of a complex matrix by the recursive power trace method, which also accounts for the repeated occupancy in the covariance matrix.
This class is an interface class betwwen the Python extension and the C++ implementation to relieve python extensions from TBB functionalities.
(CPython does not support static objects with constructors/destructors)
*/
class TorontonianRecursive  {


protected:

    /// The input matrix. Must be selfadjoint positive definite matrix with eigenvalues between 0 and 1.
    matrix mtx_orig;
    /** The scaled input matrix for which the calculations are performed.
    If the mean magnitude of the matrix elements is one, the treshold of quad precision can be set to higher values.
    */
    matrix mtx;

public:

/**
@brief Constructor of the class.
@param mtx_in A selfadjoint matrix for which the torontonian is calculated. This matrix has to be positive definite matrix with eigenvalues between 0 and 1 (for example a covariance matrix of the Gaussian state.) matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state)
@return Returns with the instance of the class.
*/
TorontonianRecursive( matrix &mtx_in );


/**
@brief Default destructor of the class.
*/
virtual ~TorontonianRecursive();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual double calculate();

/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
virtual void Update_mtx( matrix &mtx_in);


}; //PowerTraceHafnianRecursive





} // PIC

#endif
