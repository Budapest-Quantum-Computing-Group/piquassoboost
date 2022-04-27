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

#ifndef CGeneralizedCliffordsBSimulationStrategy_H
#define CGeneralizedCliffordsBSimulationStrategy_H

#include "matrix.h"
#include "matrix_real.h"
#include "PicVector.hpp"
#include "PicState.h"
#include "PicStateHash.h"
#include <unordered_map>
#include <stdlib.h>
#include <time.h>

#include "BatchedPermanentCalculator.h"



namespace pic {



/**
@brief Class representing a generalized Cliffords simulation strategy
*/
class CGeneralizedCliffordsBSimulationStrategy {

protected:

    /// The number of photons
    int64_t number_of_input_photons;
    /// The matrix describing the interferometer
    matrix interferometer_matrix;
    /// The input state entering the interferometer
    PicState_int64 input_state;
    /// ????
    PicState_int64 current_input;
    /// ????
    PicState_int64 working_input_state;
    ///
    matrix_real pmf;
    /// The vector of indices corresponding to values greater than 0 in the input state
    PicVector<int64_t> input_state_inidices;
    ///
    int lib;

#ifdef __MPI__
    /// The number of processes
    int world_size;
    /// The rank of the MPI process
    int current_rank;

    int MPI_start_index;
    int MPI_end_index;
#endif

#ifdef __DFE__
    const int useDual = 0;
#endif

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
CGeneralizedCliffordsBSimulationStrategy();

/**
@brief Constructor of the class.
@param interferometer_matrix_in The matrix describing the interferometer
@return Returns with the instance of the class.
*/
CGeneralizedCliffordsBSimulationStrategy( matrix &interferometer_matrix_in, int lib );

/**
@brief Destructor of the class
*/
~CGeneralizedCliffordsBSimulationStrategy();


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
@brief Call to randomly increase the current input state by a single photon
*/
void update_current_input();


/**
@brief Call to calculate new layer of probabilities from which an intermediate (or final) output state is sampled
*/
void compute_pmf(PicState_int64& sample);

/**
@brief Call to recursively add substates to the hashmap of labeled states.
*/
//void append_substate_to_labeled_states( PicState_int64& iter_value);


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
//void calculate_new_layer_of_pmfs( PicState_int64& sample, PicStates &possible_outputs );



/**
@brief Call to pick a new sample from the possible output states according to the calculated probability distribution stored in pmfs.
@param sample The current sample represanted by a PicState_int64 class that would be replaced by the new sample.
*/
void sample_from_pmf( PicState_int64& sample );


}; //CGeneralizedCliffordsBSimulationStrategy



/**
@brief Call to ????????????????
*/
PicState_int64 modes_state_to_particle_state( const PicState_int64& mode_state );



/**
 *  @brief Call to determine the output probability of associated with the input and output states
 *  @param interferometer_mtx The matrix of the interferometer.
 *  @param input_state The input state.
 *  @param output_state The output state.
 */
double calculate_outputs_probability(matrix &interferometer_mtx, PicState_int64 &input_state, PicState_int64 &output_state, int lib);


/**
 *  @brief Call to determine the output probability of associated with the input and output states
 *  @param interferometer_mtx The matrix of the interferometer.
 *  @param input_state The input state.
 *  @param output_state The output state.
 */
double calculate_outputs_probability(Complex16& permanent, PicState_int64 &input_state, PicState_int64 &output_state);



} // PIC

#endif
