/**
 * Copyright 2022 Budapest Quantum Computing Group
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

#ifndef PIC_SAMPLING_SIMULATION_STRATEGIES_CGeneralizedCliffordsBUniformLossesSimulationStrategy_H_INCLUDED
#define PIC_SAMPLING_SIMULATION_STRATEGIES_CGeneralizedCliffordsBUniformLossesSimulationStrategy_H_INCLUDED

#include "matrix.h"
#include "matrix_real.h"
#include "PicVector.hpp"
#include "PicState.h"

#include "BatchedPermanentCalculator.h"



namespace pic {



/**
 * @brief Class representing a generalized Cliffords simulation strategy with uniform losses
 */
class CGeneralizedCliffordsBUniformLossesSimulationStrategy {

protected:

    /// The number of photons
    //int64_t number_of_output_photons;
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
    /// probablities of the output particle numbers 0..n, where n is the input photon number
    matrix_real particle_number_probabilities;
    /// Transmittance value as :math:`t = \cos \theta`. This gives
    /// :math:`1 - t^2` is the probability of a particle loss.
    double transmittance;
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
    bool out_of_memory;
#endif

public:

/**
 * @brief Default constructor of the class.
 * @return Returns with the instance of the class.
 */
CGeneralizedCliffordsBUniformLossesSimulationStrategy();

/**
 * @brief Constructor of the class.
 * @param interferometer_matrix_in The matrix describing the interferometer
 * @param transmittance transmittance of the loss. This value squared gives
 *                      the probability of a particle remaining in the circuit.
 *                      This is called transmissivity from python side.
 * @param lib value determining the lib which is used
 * @return Returns with the instance of the class.
 */
CGeneralizedCliffordsBUniformLossesSimulationStrategy( matrix &interferometer_matrix_in, double transmittance, int lib );

/**
 * @brief Destructor of the class
 */
~CGeneralizedCliffordsBUniformLossesSimulationStrategy();


/**
 * @brief Seeds the simulation with a specified value
 * @param value The value to seed with
 */
void seed(unsigned long long int value);

/**
 * @brief Call to update the memor address of the stored matrix iinterferometer_matrix
 * @param interferometer_matrix_in The matrix describing the interferometer
 */
void Update_interferometer_matrix( matrix &interferometer_matrix_in );

/**
 * @brief Call to determine the resultant state after traversing through linear interferometer.
 * @param input_state_in The input state.
 * @param samples_number The number of shots for which the output should be determined
 * @return Returns with the resultant state after traversing through linear interferometer.
 */
std::vector<PicState_int64> simulate( PicState_int64 &input_state_in, int samples_number );

protected:

/**
 * @brief Call to randomly increase the current input state by a single photon
 */
void update_current_input();


/**
 * @brief Call to calculate new layer of probabilities from which an intermediate (or final) output state is sampled
 */
void compute_pmf(PicState_int64& sample);

/**
 * @brief Call to recursively add substates to the hashmap of labeled states.
 */

/**
 * @brief Call to calculate and fill the output states for the individual shots.
 * @param sample The current sample state represented by a PicState_int64 class
 */
void fill_r_sample( PicState_int64& sample, int64_t number_of_output_photons );


/**
 * @brief Call to pick a new sample from the possible output states according to the calculated probability distribution stored in pmfs.
 * @param sample The current sample represanted by a PicState_int64 class that would be replaced by the new sample.
 */
void sample_from_pmf( PicState_int64& sample );

/**
 *  @brief Samples remaining particles number using binomial weights computed earlier and stored in member variable.
 *  Different calls might return different values.
 * 
 *  @return: Number of remaining particles at the current sample.
 */
int64_t calculate_current_photon_number();

/**
 *  @brief In BS with uniform losses the probabilities of obtaining a specific number of
 *  bosons in the output are given by the binomial weights. This method computes
 *  these weights for the use during sampling.
 */
void calculate_particle_number_probabilities();

}; //CGeneralizedCliffordsBUniformLossesSimulationStrategy



/**
 * @brief Call to ????????????????
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

#endif // PIC_SAMPLING_SIMULATION_STRATEGIES_CGeneralizedCliffordsBUniformLossesSimulationStrategy_H_INCLUDED
