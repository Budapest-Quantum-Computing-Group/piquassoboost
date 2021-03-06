#ifndef GaussianState_Cov_H
#define GaussianState_Cov_H

#include "matrix.h"
#include "PicState.h"
#include <vector>



namespace pic {

/// enumeration labeling the representation of a gaussian state
enum representation { qudratures /* i.e. q_1, q_2, q_3, ... p_1, p_2 ... p_N */ , complex_amplitudes /*i.e. a_1, a_2, ... , a^+_1, ... a^+_N */ };


/**
@brief Class representing a Gaussian State. The state is stored by the covariance matrix and the displpacements
*/
class GaussianState_Cov {

protected:

    /// The displacement of the Gaussian state
    matrix m;
    /// The covariance matrix \f$ M_{ij (xp)} = \langle Y_i Y_j + Y_j Y_i \rangle_\rho \f$, where \f$ Y = (\overline{q}, \overline{p}) \f$,
    matrix covariance_matrix;
    /// representation basis of the gaussian state (with quadratures or with the complex_amplitudes)
    representation repr;

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
@param repr The representation type (see enumeration representation)
@return Returns with the instance of the class.
*/
GaussianState_Cov( matrix &covariance_matrix_in, matrix &m_in, representation repr_in);


/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix (The displacements are set to zeros)
@param repr The representation type (see enumeration representation)
@return Returns with the instance of the class.
*/
GaussianState_Cov( matrix &covariance_matrix_in, representation repr_in);




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



/**
@brief Call to convert the representation of the Gaussian state into complex amplitude representation, so the
displacement would be the expectation value \f$ m = \langle \hat{\xi}_i \rangle_{\rho} \f$ and the covariance matrix
\f$ covariance_matrix = \langle \hat{\xi}_i\hat{\xi}_j \rangle_{\rho}  - m_im_j\f$ .
*/
void ConvertToComplexAmplitudes();


/**
@brief Call to get the density matrix element $\f \langle i | \rho | j \rangle $\f
@param i_state PicState_int64 instance representin Fock state i
@param j_state PicState_int64 instance representin Fock state j
@return Returns with the density matrix element
*/
Complex16 getDensityMatrixElements( PicState_int64& i_state, PicState_int64& j_state );

}; //GaussianState_Cov


} // PIC

#endif
