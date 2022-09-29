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

#ifndef SAMPLING_HELPER_FUNCTIONS_H_INCLUDED
#define SAMPLING_HELPER_FUNCTIONS_H_INCLUDED


#include "matrix.h"
#include "matrix_real.h"
#include "PicState.h"


namespace pic{


/**
 * @brief Call to calculate new layer of probabilities from
 *        which an intermediate output state is sampled.
 * @param interferometer_matrix The interferometer which generally describes the circuit,
 * @param sample The current sample which is filled with the output particles,
 * @param current_input The vector containing the input particles,
 * @return The vector containing the probabilities.
 */
matrix_real compute_pmf(matrix &interferometer_matrix, PicState_int64& sample, PicState_int64 &current_input);


/**
 * @brief Call to pick a new sample from the possible output states according to the calculated probability distribution stored in pmfs.
 * @param pmf Probabilities to sample from (probability measure function)
 * @param sample The current sample represanted by a PicState_int64 class that would be replaced by the new sample.
 */
void sample_from_pmf( PicState_int64& sample, matrix_real &pmf );


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
void update_input_by_single_photon( PicState_int64& current_input, PicState_int64& working_input_state );


matrix quantum_fourier_transform_matrix(size_t n);


/** random phases in vector form!
 */
matrix random_phases_vector(size_t n);


} // namespace pic


#endif // SAMPLING_HELPER_FUNCTIONS_H_INCLUDED
