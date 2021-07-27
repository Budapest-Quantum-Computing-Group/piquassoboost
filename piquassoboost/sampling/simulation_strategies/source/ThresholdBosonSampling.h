#ifndef THRESHOLD_BOSON_SAMPLING_H
#define THRESHOLD_BOSON_SAMPLING_H

// Limit for mode number to use pmfs (caching)
constexpr int limit_for_using_pmfs = 30;

#include "matrix.h"
#include "matrix_real.h"
#include "PicVector.hpp"
#include "PicState.h"
#include "PicStateHash.h"
#include "GaussianState_Cov.h"

#include <random>
#include <unordered_map>
#include <functional>

namespace pic {


/**
@brief Structure representing a state with the Hamilton matrix defined by Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.

This data is precalculated before the simulation since it is the same for all mode indices.
matrix O: the matrix describing the substructure of the gaussian state (Matrix O defined below Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.)
double Qdet_sqrt_rec: 1 over the square root of the determinant of the matrix Q defined by Eq. (2) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.
*/
struct ThresholdMeasurementSubstate{
    /**
    @brief Constructor of the class.
    @param O The matrix O
    @param Qdet_sqrt_rec 1 / sqrt( det(Q) )
    @return Returns with the instance of the class.
    */
    ThresholdMeasurementSubstate( matrix_real& O, double Qdet_sqrt_rec );

    /// the Hamilton matrix O defined by Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time
    matrix_real O;

    /// Determinant of the matrix Q
    double Qdet_sqrt_rec;
};


/**
@brief Call to calculate sum of integers stored in a container
@param vec a container if integers
@return Returns with the sum of the elements of the container
*/
static int64_t sum( PicState_int64& vec);

/**
@brief Class representing a threshold Gaussian boson sampling simulation strategy.

From a given a gaussian state it calculates the threshold boson sampling.
*/
class ThresholdBosonSampling {
// All members and fields are inherited from base class GaussianSimulationStrategy

public:

/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix describing the gaussian state
@return Returns with the instance of the class.
*/
ThresholdBosonSampling( matrix_real& covariance_matrix_in );


/**
@brief Destructor of the class
*/
virtual ~ThresholdBosonSampling();

/**
@brief Call to update the memory address of the matrix stored in covariance_matrix
@param covariance_matrix_in The covariance matrix describing the gaussian state
*/
void Update_covariance_matrix( matrix_real& covariance_matrix_in );


/**
@brief Call to get samples from the gaussian state
@param samples_number The number of shots for which the output should be determined
@return Returns with the samples of the gaussian state
*/
std::vector<PicState_int64> simulate( int samples_number );

protected:
    /// The individual probability layers of the possible occupation numbers 
    std::unordered_map<PicState_int64, double, PicStateHash> pmfs;
    /// The number of the input modes stored by the covariance matrix
    size_t number_of_modes;
    /// Space for storing the threshold measurement specific datas for a sample which are equal in all samples.
    std::vector<ThresholdMeasurementSubstate> substates;
    /// Covariance matrix of the gaussian state (xp-ordered)
    matrix_real covariance_matrix;

void fillSubstates( matrix_real& matrix, int mode_number );

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
matrix_real calc_HamiltonMatrix( matrix_real& Qinv );


/**
@brief Call to calculate the conditional probability associated with observing output state given by current_output.

If the size of the given current_output is smaller then limit_for_using_pmfs then we use cache for having faster probability calculation.
Otherwise we just calculate the probability with method calc_probability.
The current_output is the condition except it's last element.

@param current_output The current conditions for which the conditional probability is calculated
@return Returns with the calculated probability the method calc_probability.
*/
double calc_probability_from_cache( PicState_int64& current_output );


/**
@brief Call to calculate the conditional probability associated with observing output state given by current_output.

The calculation is based on Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.
The current_output is the condition except it's last element.

@param current_output The current conditions for which the conditional probability is calculated
@return Returns with the calculated conditional probability
*/
double calc_probability( PicState_int64& current_output );


/**
@brief Call to create matrix O_S according to the main text below Eq. (14) of Ref. Exact simulation of threshold boson sampling in polynomial space and exponential time.
@param O Hamilton matrix O
@param current_output The fock representation of the current output for which the probability is calculated
@return Returns with the O_S matrix
*/
matrix_real create_O_S( matrix_real& O, PicState_int64& current_output );


/**
@brief Call to get a reduced matrix determined by input indices (i.e. special submatrix represented by a subset of modes)
@param matrix The matrix which is to reduce
@param indices_to_reduce PicState_int64, an array of mode indices where at the i-th space 0 means i-th mode is not chosen, 1 means chosen
@return Returns with the reduced matrix.
*/
static
matrix_real
reduce( matrix_real& matrix, PicState_int64& indices_to_reduce );


/**
@brief Call to calculate the inverse of matrix Q defined by Eq (3) of Ref. arXiv 2010.15595 and the determinant of Q.
Since the determinant can be calculated by LU factorization, which is also necessary to calculate the inverse, we
calculatet the inverse and the determiant in one shot.
@param matrix Input matrix whose determinant and inverse is calculated
@param Qdet The calculated determinant of the input matrix is stored into this value.
@return Returns with the inverse matrix of the input matrix.
*/
matrix_real
calc_Qinv( matrix_real& matrix, double& Qdet );

}; //ThresholdBosonSampling





} // PIC

#endif // THRESHOLD_BOSON_SAMPLING_H
