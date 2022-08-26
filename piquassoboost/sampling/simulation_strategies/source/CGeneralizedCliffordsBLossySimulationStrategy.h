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

#ifndef CGeneralizedCliffordsBLossySimulationStrategy_H
#define CGeneralizedCliffordsBLossySimulationStrategy_H

#include "matrix.h"
#include "matrix_real.h"
#include "PicVector.hpp"
#include "PicState.h"
#include "PicStateHash.h"
#include <unordered_map>
#include <stdlib.h>
#include <vector>
#include <time.h>

#include "BatchedPermanentCalculator.h"



namespace pic {



/**
@brief Class representing a generalized Cliffords simulation strategy
*/
class CGeneralizedCliffordsBLossySimulationStrategy {

protected:

    /// The matrix describing the interferometer
    matrix interferometer_matrix;
    ///
    int lib;
    /// number describing the amount of modes being approximated together
    size_t number_of_approximated_modes;
    /// approximated modes sum of particle number
    size_t particle_number_in_approximated_modes;
    /**
     * Probablities of the output particle numbers 0..n, where
     * n is the input photon number.
     */
    std::vector<matrix_real> binomial_weights;
    /** 
     * Probability of the uniform loss of a particle
     * (probability of the disappear of a photon)
     */
    double uniform_loss;

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
CGeneralizedCliffordsBLossySimulationStrategy();


/**
 * @brief Constructor of the class.
 * @param interferometer_matrix_in The matrix describing
 *        the interferometer
 * @return Returns with the instance of the class.
 */
CGeneralizedCliffordsBLossySimulationStrategy( matrix &interferometer_matrix_in, int lib );


/**
 * @brief Constructor of the class.
 * @param interferometer_matrix_in The matrix describing the
 *        interferometer
 * @param number_of_approximated_modes_in Number which describes
 *        the number of modes which are approximated
 * @param lib
 * @return Returns with the instance of the class.
 */
CGeneralizedCliffordsBLossySimulationStrategy( matrix &interferometer_matrix_in, int number_of_approximated_modes_in, int lib );


/**
 * @brief Destructor of the class
 */
~CGeneralizedCliffordsBLossySimulationStrategy();


/**
 * @brief Seeds the simulation with a specified value
 * @param value The value to seed with
 */
void seed(unsigned long long int value);


/**
 * @brief Setter for approximated number of modes
 * @param value The value to set
 */
void set_approximated_modes_number(int value);


/**
 * @brief Call to update the memory address of the
 *        stored matrix interferometer_matrix
 * @param interferometer_matrix_in The matrix describing
 *        the interferometer
 */
void Update_interferometer_matrix( matrix &interferometer_matrix_in );


/**
 * @brief Call to determine the resultant state after traversing
 *        through linear interferometer.
 * @param input_state_in The input state.
 * @param samples_number The number of shots for which the output
 *        should be determined
 * @return Returns with the resultant state after traversing through
 *         linear interferometer.
 */
std::vector<PicState_int64> simulate(
    PicState_int64 &input_state_in,
    int samples_number
);


protected:

/**
 * @brief Call to randomly increase the current input state
 *        by a single photon
 * 
 * This method randomly chooses a particle from
 * `working_input_state` and fills it into the current
 * input and the particle is eliminated from 
 * `working_input_state`.
 * @param current_input Current input state to update
 * @param working_input_state The indices of the input particle
 *        from which we choose randomly
 */
void update_input_by_single_photon(
    PicState_int64& current_input,
    PicState_int64& working_input_state
);


/**
 * @brief Call to calculate new layer of probabilities from
 *        which an intermediate output state is sampled.
 * 
 * @return The vector containing the probabilities.
 */
matrix_real compute_pmf( PicState_int64& sample, PicState_int64& current_input );


/**
 * @brief Call to calculate and fill the output states for the individual shots.
 * @param sample The current sample state represented by a PicState_int64 class
 * @param current_input The input state where we are currently (starting with all 0's)
 * @param working_input_state The input state we want to reach
 */
void fill_r_sample(
    PicState_int64& sample,
    PicState_int64& current_input,
    PicState_int64& working_particle_input_state
);


/**
 * @brief Call to pick a new sample from the possible output states according to the calculated probability distribution stored in pmfs.
 * @param pmf Probabilities to sample from
 * @param sample The current sample represanted by a PicState_int64 class that would be replaced by the new sample.
 */
void sample_from_pmf( matrix_real &pmf, PicState_int64& sample );


/**
 *  @brief In BS with uniform losses the probabilities of obtaining a specific number of
 *  bosons in the output are given by the binomial weights. This method computes
 *  these weights for the use during sampling.
 * 
 *  @param maximal_particle_number Maximal number of particles in a single mode of the input state. Recall that
 *                                 we expect the input state to already be in the proper form.
 */
void calculate_particle_number_probabilities(int maximal_particle_number);


/**
 * @brief Eliminate uniform losses from the interferometer
 * 
 * This method calculates the losses from the interferometer matrix
 * using svd decomposition and the minimal loss is filled into 
 * `uniform_loss` and the interferometer matrix is updated based
 * on the eliminated loss value.
 */
void extract_losses_from_interferometer();


/**
 * @brief Creates a lossy particle input from the given input
 * 
 * Creates a random input from the particles. The returned state
 * contains indices: the k'th index show the index of mode where
 * the k'th particle is.
 * 
 * @param input_state_without_approx_modes the input which is
 *        modified to sample from.
 * @return The randomly created lossy input
 */
PicState_int64 compute_lossy_particle_input(
    PicState_int64 &input_state_without_approx_modes
);


/**
 * @brief Gives a random particle number based on a mode
 * @param particle_number The maximal number of particles on
 *        the specific mode.
 * @return Random particle number on the current mode.
 */
size_t random_particle_number(size_t particle_number);


/**
 * @brief Changes the interferometer matrix with the approximation
 *        strategy.
 */
void update_matrix_for_approximate_sampling();


}; //CGeneralizedCliffordsBLossySimulationStrategy

matrix quantum_fourier_transform_matrix(size_t n);

/** random phases in vector form!
 */
matrix random_phases_vector(size_t n);

} // PIC

#endif // CGeneralizedCliffordsBLossySimulationStrategy_H
