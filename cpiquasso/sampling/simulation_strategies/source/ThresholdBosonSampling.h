#ifndef THRESHOLD_BOSON_SAMPLING_H
#define THRESHOLD_BOSON_SAMPLING_H

#include "matrix.h"
#include "PicVector.hpp"
#include "PicState.h"
#include "PicStateHash.h"
#include "GaussianState_Cov.h"
#include "GaussianSimulationStrategy.h"
#include <random>
#include <unordered_map>

namespace pic {

/**
@brief Structure representing a state with the Hamilton matrix defined by Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.

This data is precalculated before the simulation since it is the same for all mode indices.
matrix O: the matrix describing the substructure of the gaussian state (Matrix O defined below Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.)
double Qdet: determinant of the matrix Q defined by Eq. (2) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.
*/
struct ThresholdMeasurementSubstate{
    /**
    @brief Constructor of the class.
    @param O The matrix O
    @param Qdet Determinant of Q
    @return Returns with the instance of the class.
    */
    ThresholdMeasurementSubstate( matrix& O, double Qdet );

    /// the Hamilton matrix O defined by Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time
    matrix O;

    /// Determinant of the matrix Q
    double Qdet;
};



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
    /// The individual probability layers of the possible occupation numbers 
    std::unordered_map<PicState_int64, double, PicStateHash> pmfs;

    /// Space for storing the threshold measurement specific datas for a sample which are equal in all samples.
    std::vector<ThresholdMeasurementSubstate> substates;

    /// Boolean to store the information whether we have to calculate new torontonian or not
    bool torontonian_calculation_needed;

    /// Value of the torontonian in the current calculation. If the occupation number is 0 on the last mode, the torontonian does not change.
    double last_torontonian;

void fillSubstates( int mode_number );

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

@param Qdet The determinant of matrix Q.
@param O Hamilton matrix 
@param current_output The current conditions for which the conditional probability is calculated
@return Returns with the calculated probability
*/
virtual double calc_probability( const double& Qdet, matrix& O, PicState_int64& current_output );


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