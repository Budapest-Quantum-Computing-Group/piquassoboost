#ifndef THRESHOLD_BOSON_SAMPLING_H
#define THRESHOLD_BOSON_SAMPLING_H

#include "matrix.h"
#include "PicVector.hpp"
#include "PicState.h"
#include "PicStateHash.h"
#include "GaussianState_Cov.h"
#include "GaussianSimulationStrategy.h"
#include <random>

namespace pic {




/**
@brief Call to calculate sum of integers stored in a container
@param vec a container if integers
@return Returns with the sum of the elements of the container
*/
static int64_t sum( PicState_int64 &vec);

/**
@brief Class representing a threshold Gaussian boson sampling simulation strategy.

From a given a gaussian state it calculates the threshold boson sampling.
*/
class ThresholdBosonSampling : public GaussianSimulationStrategy{
// All members and fields are inherited from base class GaussianSimulationStrategy

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
ThresholdBosonSampling();

/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix describing the gaussian state
@return Returns with the instance of the class.
*/
ThresholdBosonSampling( matrix &covariance_matrix_in );


/**
@brief Destructor of the class
*/
virtual ~ThresholdBosonSampling();

/**
@brief Call to update the memory address of the matrix stored in covariance_matrix
@param covariance_matrix_in The covariance matrix describing the gaussian state
*/
void Update_covariance_matrix( matrix &covariance_matrix_in );


/**
@brief Call to get samples from the gaussian state
@param samples_number The number of shots for which the output should be determined
@return Returns with the samples of the gaussian state
*/
std::vector<PicState_int64> simulate( int samples_number );

protected:





/**
@brief Call to get one sample from the gaussian state
@return Returns with the a sample from a gaussian state
*/
PicState_int64 getSample();


/**
@brief Call to calculate the Hamilton matrix A defined by Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.

O = 1 - Qinv

@param Qinv An instace of matrix class conatining the inverse of matrix Q calculated by method get_Qinv.
@return Returns with the Hamilton matrix A.
*/
matrix calc_HamiltonMatrix( matrix& Qinv );


/**
@brief Call to calculate the probability associated with observing output state given by current_output

The calculation is based on Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.

@param Qinv An instace of matrix class conatining the inverse of matrix Q calculated by method get_Qinv.
@param Qdet The determinant of matrix Q.
@param O Hamilton matrix 
@param current_output The current conditions for which the conditional probability is calculated
@return Returns with the calculated probability
*/
virtual double calc_probability( matrix& Qinv, const double& Qdet, matrix& O, PicState_int64& current_output );


/**
@brief Call to create matrix O_S according to the main text below Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.
@param O Hamilton matrix O
@param current_output The fock representation of the current output for which the probability is calculated
@return Returns with the O_S matrix
*/
matrix create_O_S( matrix& O, PicState_int64& current_output );

}; //ThresholdBosonSampling





} // PIC

#endif // THRESHOLD_BOSON_SAMPLING_H
