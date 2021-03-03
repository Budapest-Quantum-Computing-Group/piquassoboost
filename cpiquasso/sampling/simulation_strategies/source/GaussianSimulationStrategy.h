#ifndef GaussianSimulationStrategy_H
#define GaussianSimulationStrategy_H

#include "matrix.h"
#include "PicVector.hpp"
#include "PicState.h"
#include "PicStateHash.h"
#include <unordered_map>
#include <random>


namespace pic {




/**
@brief Call to calculate sum of integers stored in a container
@param vec a container if integers
@return Returns with the sum of the elements of the container
*/
static int64_t sum( PicState_int64 &vec);

/**
@brief Class representing a generalized Cliffords simulation strategy
*/
class GaussianSimulationStrategy {

protected:

    /// The covariance matrix describing the gaussian state
    matrix covariance_matrix;

    /// The number of photons
    int64_t number_of_input_photons;
    /// The individual probability layers of the possible output states
    std::unordered_map<PicState_int64, matrix_base<double>, PicStateHash> pmfs;
    /// The possible output sates organized by keys of the inducing input states
    std::unordered_map<PicState_int64, PicStates, PicStateHash> possible_output_states;
    /// The input state entering the interferometer
    PicState_int64 input_state; // can be aligned?
    /// Map of all the possible substates of input_state (including 0 and the input state), labeled with number of particles in this state
    /// the number of particles is given by the ordinal number of the vector element, i.e. labeled_states[n] gives the substates with n occupied particles.
    std::vector<concurrent_PicStates> labeled_states;
    /// The vector of indices corresponding to values greater than 0 in the input state
    PicVector<int64_t> input_state_inidices;
    /// random number generator
    std::default_random_engine generator;


public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategy();

/**
@brief Constructor of the class.
@param interferometer_matrix_in The matrix describing the interferometer
@return Returns with the instance of the class.
*/
GaussianSimulationStrategy( matrix &covariance_matrix_in );

/**
@brief Destructor of the class
*/
~GaussianSimulationStrategy();

/**
@brief Call to update the memory address of the matrix stored in covariance_matrix
@param covariance_matrix_in The covariance matrix describing the gaussian state
*/
void Update_covariance_matrix( matrix &covariance_matrix_in );

/**
@brief Call to determine the resultant state after traversing through linear interferometer.
@param input_state_in The input state.
@param samples_number The number of shots for which the output should be determined
@return Returns with the resultant state after traversing through linear interferometer.
*/
std::vector<PicState_int64> simulate( int samples_number );

protected:


}; //GaussianSimulationStrategy





} // PIC

#endif
