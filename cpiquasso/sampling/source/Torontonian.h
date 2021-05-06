#ifndef Torontonian_H
#define Torontonian_H

#include "matrix.h"


namespace pic {



/**
@brief Class to calculate the torontonian of a complex matrix by the power trace method
*/
class Torontonian {

protected:
    /// The input matrix. Must be positive definite matrix with eigenvalues between 0 and 1.
    matrix mtx_orig;
    /** The scaled input matrix for which the calculations are performed.
    If the mean magnitude of the matrix elements is one, the treshold of quad precision can be set to higher values.
    */
    matrix mtx;
    /// The scale factor of the input matric
    double scale_factor;


public:


/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
Torontonian();

/**
@brief Constructor of the class.
@param mtx_in A selfadjoint matrix for which the torontonian is calculated. This matrix has to be positive definite matrix with eigenvalues between 0 and 1 (for example a covariance matrix of the Gaussian state.)
@return Returns with the instance of the class.
*/
Torontonian( matrix &mtx_in );

/**
@brief Default destructor of the class.
*/
virtual ~Torontonian();

/**
@brief Method to calculate the torontonian of a complex selfadjoint positive definite matrix with eigenvalues between 0 and 1.
@return Returns with the calculated hafnian
*/
//virtual Complex16 calculate();
virtual double calculate();


/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void Update_mtx( matrix &mtx_in);


protected:

/**
@brief Call to scale the input matrix according to according to Eq (2.11) of in arXiv 1805.12498
@param mtx_in Input matrix defined by
*/
virtual void ScaleMatrix();



}; //Torontonian


} // PIC

#endif // Torontonian_H
