#ifndef GaussianSimulationStrategyFast_H
#define GaussianSimulationStrategyFast_H

#include "GaussianSimulationStrategy.h"


namespace pic {




/**
@brief Call to calculate sum of integers stored in a container
@param vec a container if integers
@return Returns with the sum of the elements of the container
*/
static int64_t sum( PicState_int64 &vec);




/**
@brief Class representing a Gaussian boson sampling simulation strategy utilizing the recursive power trace hafnian algorithm.
*/
class GaussianSimulationStrategyFast : public GaussianSimulationStrategy {


public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategyFast();

/**
@brief Constructor of the class. (The displacement is set to zero by this constructor)
@param covariance_matrix_in The covariance matrix describing the gaussian state
@param cutoff the Fock basis truncation.
@param max_photons specifies the maximum number of photons that can be counted in the output samples.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategyFast( matrix &covariance_matrix_in, const size_t& cutoff, const size_t& max_photons );


/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix describing the gaussian state
@param displacement The mean (displacement) of the Gaussian state
@param cutoff the Fock basis truncation.
@param max_photons specifies the maximum number of photons that can be counted in the output samples.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategyFast( matrix &covariance_matrix_in, matrix& displacement_in, const size_t& cutoff, const size_t& max_photons );



/**
@brief Destructor of the class
*/
~GaussianSimulationStrategyFast();



protected:





/**
@brief Call to calculate the probability associated with observing output state given by current_output
@param Qinv An instace of matrix class conatining the inverse of matrix Q calculated by method get_Qinv.
@param Qdet The determinant of matrix Q.
@param A Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
@param m The displacement \f$ \alpha \f$ defined by Eq (8) of Ref. arXiv 2010.15595
@param current_output The fock representation of the current output for which the probability is calculated
@return Returns with the calculated probability
*/
double calc_probability( matrix& Qinv, const double& Qdet, matrix& A, matrix& m, PicState_int64& current_output );



/**
@brief Call to extract selected modes from the covariance matrix in \$f a_1, a_1^*, a_2, a_2^* ... \f$ ordering.
@param A Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
@param selected_modes An array of labels containing the selected modes
@return Returns with the matrix containing the selected modes
*/
matrix ExtractModes( matrix& A, PicState_int64& selected_modes );


/**
@brief Call to add correction coming from the displacement to the diagonal elements of A_S (see Eq. (11) in arXiv 2010.15595)
@param A Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
(The output is returned via this variable)
@param gamma The diagonal correction according to see Eq. (11) in arXiv 2010.15595. Here gamma is ordered as a_1, a_1^* ,a_2, a_2^*, ...
*/
void diag_correction_of_A( matrix& A, matrix& gamma );


/**
@brief Call to calculate gamma according to Eq (9) of arXiv 2010.15595v3 in ordering a_1, a_1^*, a_2, a_2^* ....
@param Qinv An instace of matrix class containing the inverse of matrix Q calculated by method get_Qinv.
@param m The displacement \f$ \alpha \f$ defined by Eq (8) of Ref. arXiv 2010.15595
@param current_output The Fock representation of the current output for which the probability is calculated
*/
matrix CalcGamma( matrix& Qinv, matrix& m, PicState_int64& selected_modes );


}; //GaussianSimulationStrategy





} // PIC

#endif
