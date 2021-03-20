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
    // reuse constructors of the base class
    using GaussianSimulationStrategy::GaussianSimulationStrategy;



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


}; //GaussianSimulationStrategy





} // PIC

#endif
