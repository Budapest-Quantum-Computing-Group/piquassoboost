#ifndef GaussianState_Cov_H
#define GaussianState_Cov_H

#include "matrix.h"
#include "PicState.h"
#include <vector>



namespace pic {

/**
@brief Class representing a Gaussian State. The state is stored by the covariance matrix and the displpacements
*/
class GaussianState_Cov {

protected:

    /// The displacement of the Gaussian state
    matrix m;
    /// The covariance matrix
    matrix covariance_matrix;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GaussianState_Cov();

/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix
@param m_in The displacement of the Gaussian state
@return Returns with the instance of the class.
*/
GaussianState_Cov( matrix &covariance_matrix_in, matrix &m_in);


/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix (The displacements are set to zeros)
@return Returns with the instance of the class.
*/
GaussianState_Cov( matrix &covariance_matrix_in );




/**
@brief Call to update the memory address of the vector containing the displacements
@param m_in The new displacement vector
*/
void Update_m(matrix &m_in);


/**
@brief Call to update the memory address of the matrix stored in covariance_matrix
@param covariance_matrix_in The covariance matrix describing the gaussian state
*/
void Update_covariance_matrix( matrix &covariance_matrix_in );


/**
@brief Call to get a reduced Gaussian state (i.e. the gaussian state represented by a subset of modes of the original gaussian state)
@param modes An instance of PicState_int64 containing the modes to be extracted from the original gaussian state
@return Returns with the reduced Gaussian state
*/
GaussianState_Cov getReducedGaussianState( PicState_int64 &modes );

}; //GaussianState_Cov


} // PIC

#endif
