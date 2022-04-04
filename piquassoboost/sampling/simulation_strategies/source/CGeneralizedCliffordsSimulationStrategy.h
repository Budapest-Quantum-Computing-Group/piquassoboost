/**
 * Copyright 2021 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CGeneralizedCliffordsSimulationStrategy_H
#define CGeneralizedCliffordsSimulationStrategy_H

#include "matrix.h"
#include "PicVector.hpp"
#include "PicState.h"
#include "PicStateHash.h"
#include <unordered_map>
#include <stdlib.h>
#include <time.h>


namespace pic {



/**
@brief Class representing a generalized Cliffords simulation strategy
*/
class CGeneralizedCliffordsSimulationStrategy {

protected:

    /// The number of photons
    int64_t number_of_input_photons;
    /// The individual probability layers of the possible output states
    std::unordered_map<PicState_int64, matrix_base<double>, PicStateHash_int64> pmfs;
    /// The possible output sates organized by keys of the inducing input states
    std::unordered_map<PicState_int64, PicStates, PicStateHash_int64> possible_output_states;
    /// The matrix describing the interferometer
    matrix interferometer_matrix;
    /// The input state entering the interferometer
    PicState_int64 input_state; // can be aligned?
    /// Map of all the possible substates of input_state (including 0 and the input state), labeled with number of particles in this state
    /// the number of particles is given by the ordinal number of the vector element, i.e. labeled_states[n] gives the substates with n occupied particles.
    std::vector<concurrent_PicStates> labeled_states;
    /// The vector of indices corresponding to values greater than 0 in the input state
    PicVector<int64_t> input_state_inidices;

#ifdef __MPI__
    /// The number of processes
    int world_size;
    /// The rank of the MPI process
    int current_rank;

    int MPI_start_index;
    int MPI_end_index;
#endif


public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
CGeneralizedCliffordsSimulationStrategy();

/**
@brief Constructor of the class.
@param interferometer_matrix_in The matrix describing the interferometer
@return Returns with the instance of the class.
*/
CGeneralizedCliffordsSimulationStrategy( matrix &interferometer_matrix_in );

/**
@brief Destructor of the class
*/
~CGeneralizedCliffordsSimulationStrategy();


/**
@brief Seeds the simulation with a specified value
@param value The value to seed with
*/
void seed(unsigned long long int value);

/**
@brief Call to update the memor address of the stored matrix iinterferometer_matrix
@param interferometer_matrix_in The matrix describing the interferometer
*/
void Update_interferometer_matrix( matrix &interferometer_matrix_in );

/**
@brief Call to determine the resultant state after traversing through linear interferometer.
@param input_state_in The input state.
@param samples_number The number of shots for which the output should be determined
@return Returns with the resultant state after traversing through linear interferometer.
*/
std::vector<PicState_int64> simulate( PicState_int64 &input_state_in, int samples_number );

protected:

/**
@brief Call to determine and sort all the substates of the input. They will later be used to calculate output probabilities.
*/
void get_sorted_possible_states();


/**
@brief Call to recursively add substates to the hashmap of labeled states.
*/
void
append_substate_to_labeled_states( PicState_int64& iter_value);


/**
@brief Call to calculate and fill the output states for the individual shots.
@param sample The current sample state represented by a PicState_int64 class
*/
void fill_r_sample( PicState_int64& sample );


/**
@brief Call to calculate a new layer of probabilities of ....?????????????
@param sample A preallocated PicState_int64 for one output
@param possible_outputs A preallocated vector of possible output states
*/
void calculate_new_layer_of_pmfs( PicState_int64& sample, PicStates &possible_outputs );



/**
@brief Call to pick a new sample from the possible output states according to the calculated probability distribution stored in pmfs.
@param sample The current sample represanted by a PicState_int64 class that would be replaced by the new sample.
*/
void sample_from_latest_pmf( PicState_int64& sample );


}; //CGeneralizedCliffordsSimulationStrategy




/**
@brief Call to calculate weigths for the possible input states
@param input_state The input state entering the interferometer
@param input_state_inidices The vector of indices corresponding to values greater than 0 in the input state
@param possible_input_state Other possible input states
@param multinomial_coefficients The multinomial coefficients associated with the possible k vectors (referenced for output)
@param wieght_norm The squared sum of the calculated weigts (referenced for output)
*/
void calculate_weights( tbb::blocked_range<size_t> &r, PicState_int64 &input_state, PicVector<int64_t> &input_state_inidices, concurrent_PicStates& possible_input_states, matrix_base<double> &multinomial_coefficients, double& wieght_norm );





/**
@brief Call to generate possible output state
@param sample The current output sample for which the probabilities are calculated
@param possible_outputs Vector of possible output states
*/
void generate_output_states( PicState_int64& sample, PicStates &possible_outputs );


/**
 *  @brief Call to determine the output probability of associated with the input and output states
 *  @param interferometer_mtx The matrix of the interferometer.
 *  @param input_state The input state.
 *  @param output_state The output state.
 */
double calculate_outputs_probability(matrix &interferometer_mtx, PicState_int64 &input_state, PicState_int64 &output_state);


/** @brief Creates a matrix from the `interferometerMatrix` corresponding to the parameters `input_state` and `output_state`.
 *         Corresponding rows and columns are multipled based on output and input states.
 *  @param interferometerMatrix Unitary matrix describing a quantum circuit
 *  @param input_state_in The input state
 *  @param output_state_in The output state
 *  @return Returns with the created matrix
 */
matrix
adaptInterferometerGlynnMultiplied(
    matrix& interferometerMatrix,
    PicState_int64 &input_state,
    PicState_int64 &output_state
);


/** @brief Creates a matrix from the `interferometerMatrix` corresponding to 
 *         the parameters `input_state` and `output_state`.
 *         Does not adapt input and ouput states. They have to be adapted explicitly.
 *         Those matrix rows and columns remain in the adapted matrix where the multiplicity
 *         given by the input and ouput states is nonzero.
 *  @param interferometerMatrix Unitary matrix describing a quantum circuit
 *  @param input_state_in The input state
 *  @param output_state_in The output state
 *  @return Returns with the created matrix
 */
matrix
adaptInterferometer(
    matrix& interferometerMatrix,
    PicState_int64 &input_state,
    PicState_int64 &output_state
);


} // PIC

#endif
