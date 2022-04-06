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

#include <iostream>
#include "CGeneralizedCliffordsSimulationStrategy.h"
#include "CChinHuhPermanentCalculator.h"
#include "GlynnPermanentCalculator.h"
#include "GlynnPermanentCalculatorRepeated.h"
#include "common_functionalities.h"
#include <math.h>
#include <tbb/tbb.h>
#include <chrono>


#ifdef __MPI__
#include <mpi.h>
#endif // MPI


namespace pic {
/*
    double rand_nums[40] = {0.929965, 0.961441, 0.46097, 0.090787, 0.137104, 0.499059, 0.951187, 0.373533, 0.634074, 0.0886671, 0.0856861, 0.999702, 0.419755, 0.376557, 0.947568, 0.705106, 0.0520666, 0.45318,
            0.874288, 0.656594, 0.287817, 0.484918, 0.854716, 0.31408, 0.516911, 0.374158, 0.0124914, 0.878496, 0.322593, 0.699271, 0.0583747, 0.56629, 0.195314, 0.00059639, 0.443711, 0.652659, 0.350379, 0.839752, 0.710161, 0.28553};
    int rand_num_idx = 0;
*/



/**
@brief Call to calculate sum of integers stored in a PicState
@param vec a container if integers
@return Returns with the sum of the elements of the container
*/
static inline int64_t
sum( PicState_int64 &vec) {

    int64_t ret=0;

    size_t element_num = vec.cols;
    int64_t* data = vec.get_data();
    for (size_t idx=0; idx<element_num; idx++ ) {
        ret = ret + data[idx];
    }
    return ret;
}



/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
CGeneralizedCliffordsSimulationStrategy::CGeneralizedCliffordsSimulationStrategy() {
    // seed the random generator
    seed(time(NULL));

#ifdef __MPI__
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

#endif

}


/**
@brief Constructor of the class.
@param interferometer_matrix_in The matrix describing the interferometer
@return Returns with the instance of the class.
*/
CGeneralizedCliffordsSimulationStrategy::CGeneralizedCliffordsSimulationStrategy( matrix &interferometer_matrix_in ) {

    Update_interferometer_matrix( interferometer_matrix_in );

    // seed the random generator
    seed(time(NULL));

#ifdef __MPI__
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

#endif
}


/**
@brief Destructor of the class
*/
CGeneralizedCliffordsSimulationStrategy::~CGeneralizedCliffordsSimulationStrategy() {
}

/**
@brief Seeds the simulation with a specified value
@param value The value to seed with
*/
void
CGeneralizedCliffordsSimulationStrategy::seed(unsigned long long int value) {
    srand(value);
}

/**
@brief Call to update the memor address of the stored matrix iinterferometer_matrix
@param interferometer_matrix_in The matrix describing the interferometer
*/
void
CGeneralizedCliffordsSimulationStrategy::Update_interferometer_matrix( matrix &interferometer_matrix_in ) {

    interferometer_matrix = interferometer_matrix_in;

}



/**
@brief Call to determine the resultant state after traversing through linear interferometer.
@param input_state_in The input state.
@return Returns with the resultant state after traversing through linear interferometer.
*/
std::vector<PicState_int64>
CGeneralizedCliffordsSimulationStrategy::simulate( PicState_int64 &input_state_in, int samples_number ) {

    input_state = input_state_in;

    // get the possible substates of the input state and their weight for the probability calculations
    get_sorted_possible_states();


    // preallocate the memory for the output states
    std::vector<PicState_int64> samples;

    if ( samples_number <= 0 ) return;

    samples.reserve( samples_number );


#ifdef __MPI__

    int samples_number_per_process = samples_number/world_size;

    // calculate the first iteration of the sampling process
    PicState_int64 sample(input_state_in.cols, 0);
    sample.number_of_photons = 0;
    fill_r_sample( sample );
    
    
    // calculate the individual outputs for the shots and send the calculated outputs to other MPI processes in parallel
    PicState_int64 sample_new;
    for (int idx=1; idx<samples_number_per_process; idx++) {

        tbb::parallel_invoke(

            [&]{
                sample_new = PicState_int64(input_state_in.cols, 0);
                sample_new.number_of_photons = 0;
                fill_r_sample( sample_new );
            },
            [&]{
    
                // gather the samples over the MPI processes
                PicState_int64 sample_gathered( sample.size()*world_size );
                int bytes = sample.size()*sizeof(int64_t);
  
                MPI_Allgather(sample.get_data(), bytes, MPI_BYTE, sample_gathered.get_data(), bytes, MPI_BYTE, MPI_COMM_WORLD);
        
                for( int rank=0; rank<world_size; rank++) {
                    PicState_int64 sample_local( sample_gathered.get_data()+rank*sample.size(), sample.size() );
                    samples.push_back( sample_local.copy() );
                }

            }

        ); // parallel invoke     

        sample = sample_new;
        

    }

   
    // gather the samples over the MPI processes of the last iteration
    PicState_int64 sample_gathered( sample.size()*world_size );
    int bytes = sample.size()*sizeof(int64_t);
    MPI_Allgather(sample.get_data(), bytes, MPI_BYTE, sample_gathered.get_data(), bytes, MPI_BYTE, MPI_COMM_WORLD);
        
    for( int rank=0; rank<world_size; rank++) {
        PicState_int64 sample_local( sample_gathered.get_data()+rank*sample.size(), sample.size() );
        samples.push_back( sample_local.copy() );
    }


#else

    // calculate the individual outputs for the shots
    for (int idx=0; idx<samples_number; idx++) {
        PicState_int64 sample(input_state_in.cols, 0);
        sample.number_of_photons = 0;
        fill_r_sample( sample );
        samples.push_back( sample );
    }


#endif



    // clear the disctionaries
    pmfs.clear();
    possible_output_states.clear();
    labeled_states.clear();
    input_state_inidices.clear();

    return samples;
}




/**
@brief Call to determine and sort all the substates of the input. They will later be used to calculate output probabilities.
*/
void
CGeneralizedCliffordsSimulationStrategy::get_sorted_possible_states() {


    // locate nonzero elements of input state and count the number of photons
    number_of_input_photons = 0;
    input_state_inidices.reserve( input_state.rows*input_state.cols );
    input_state_inidices.number_of_photons = 0;

    for ( size_t idx = 0; idx<input_state.rows*input_state.cols; idx++) {
        if ( input_state[idx] > 0 ) {
            input_state_inidices.push_back(idx);
            input_state_inidices.number_of_photons++;
            number_of_input_photons = number_of_input_photons + input_state[idx];
        }
    }


    // preallocate elements for labeled states
    labeled_states.reserve(number_of_input_photons+1);
    for (int64_t idx=0; idx<=number_of_input_photons; idx++) {
        concurrent_PicStates tmp;
        labeled_states.push_back(tmp);
    }

    if (input_state_inidices.size() == 0) return;
    // creaint recursively possible output states
    tbb::parallel_for ((int64_t)0, input_state[input_state_inidices[0]]+1, (int64_t)1, [&](int64_t idx){

        PicState_int64 iter_value( input_state_inidices.size(), 0);
        iter_value[0] = idx;
        if (idx>0) {
            iter_value.number_of_photons = idx;
        }
        else{
            iter_value.number_of_photons = 0;
        }

        append_substate_to_labeled_states( iter_value );

    });


}


/**
@brief Call to recursively add substates to the hashmap of labeled states.
*/
void
CGeneralizedCliffordsSimulationStrategy::append_substate_to_labeled_states( PicState_int64& iter_value) {

        // creating the v_vector
        PicState_int64 substate(input_state.cols,0);
        size_t idx_max = 0;
        for ( size_t idx=0; idx<iter_value.size(); idx++) {
            if ( iter_value[idx] > 0 ) {
                substate[input_state_inidices[idx]] = iter_value[idx];
                idx_max = idx;
            }
            substate.number_of_photons = iter_value.number_of_photons;
        }

        labeled_states[substate.number_of_photons].push_back(substate);


        // adding new substates to the do cycle
        tbb::parallel_for ( idx_max+1, iter_value.size(), (size_t)1, [&](size_t idx) {
            for ( int jdx=1; jdx<=input_state[input_state_inidices[idx]]; jdx++) {
                PicState_int64 iter_value_next = iter_value.copy();
                iter_value_next[idx] = jdx;
                iter_value_next.number_of_photons = iter_value.number_of_photons + 1;

                append_substate_to_labeled_states( iter_value_next );
                //feeder.add(iter_value_next);
            }
        });



}


/**
@brief Call to calculate and fill the output states for the individual shots.
@param sample The current sample state represented by a PicState_int64 class
*/
void
CGeneralizedCliffordsSimulationStrategy::fill_r_sample( PicState_int64& sample ) {


    while (number_of_input_photons > sample.number_of_photons) {

        if ( pmfs.count(sample) == 0) {

            // preallocate states for possible output states
            PicStates possible_outputs;
            possible_outputs.reserve(sample.size());
            for (size_t idx=0; idx<sample.size(); idx++) {
                PicState_int64 possible_output(sample.size(), 0);
                possible_outputs.push_back( possible_output );
            }

            // create a new key for the hash table
            PicState_int64 key = sample.copy();
            calculate_new_layer_of_pmfs( key, possible_outputs );
            possible_output_states[key] = possible_outputs;

        }

        // pick a new sample from the possible output states according to the calculated probability distribution stored in pmfs
        sample_from_latest_pmf(sample);


    }


}




/**
@brief Call to calculate a new layer of probabilities of the possible output states
@param sample A preallocated PicState_int64 for one output
@param possible_outputs A preallocated vector of possible output states
*/
void
CGeneralizedCliffordsSimulationStrategy::calculate_new_layer_of_pmfs( PicState_int64& sample, PicStates &possible_outputs ) {

    int64_t number_of_particle_to_sample = sample.number_of_photons + 1;
    concurrent_PicStates &possible_input_states = labeled_states[number_of_particle_to_sample];

    // parallel loop to calculate the weights from possible input states and their weight
    matrix_base<double> multinomial_coefficients = matrix_base<double>(1, possible_input_states.size());
    tbb::combinable<double> wieght_norm{0.0};

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, possible_input_states.size()), [&](tbb::blocked_range<size_t> r ) {
        // thread local storage for weight_norm
        double &wieght_norm_priv = wieght_norm.local();

        // calculate the representation of the input states and their weights
        calculate_weights( r, input_state, input_state_inidices, possible_input_states, multinomial_coefficients, wieght_norm_priv );
    }); // parallel for

    // normalize the calculated weights
    double weight_norm_total=0;
    wieght_norm.combine_each([&](double a) {  // combine thread local data into a total one
        weight_norm_total = weight_norm_total + a;
    });

    weight_norm_total = sqrt(weight_norm_total);
    for (size_t idx=0; idx<multinomial_coefficients.size(); idx++ ) {
        multinomial_coefficients[idx] = multinomial_coefficients[idx]/weight_norm_total;
    }

    // container to store the new layer of output probabilities
    matrix_base<double> pmf(1, possible_outputs.size());
    memset( pmf.get_data(), 0.0, pmf.size()*sizeof(double) );

    // Generate output states
    generate_output_states( sample, possible_outputs );

    // normalize the probabilities:
    double probability_sum_total = 0;

    // calculate the individual probabilities associated with the output states
    for ( int idx=0; idx<possible_outputs.size(); idx++) {

        pmf[idx] = 0;

        // calculate the individual probabilities
        for ( size_t jdx=0; jdx<possible_input_states.size(); jdx++ ) {
            double probability = calculate_outputs_probability(interferometer_matrix, possible_input_states[jdx], possible_outputs[idx]);
            probability = probability*multinomial_coefficients[jdx]*multinomial_coefficients[jdx];
            pmf[idx] = pmf[idx] + probability;
        }

        probability_sum_total = probability_sum_total + pmf[idx];


    }


    for (size_t idx=0;idx<pmf.size(); idx++) {
        pmf[idx] = pmf[idx]/probability_sum_total;
    }


    // store the calculated probability layer
    PicState_int64 key = sample.copy();
    pmfs[key] = pmf;


}


/**
@brief Call to pick a new sample from the possible output states according to the calculated probability distribution stored in pmfs.
@param sample The current sample represanted by a PicState_int64 class that would be replaced by the new sample.
*/
void
CGeneralizedCliffordsSimulationStrategy::sample_from_latest_pmf( PicState_int64& sample ) {


    // create a random double
    double rand_num = (double)rand()/RAND_MAX;
   //double rand_num = rand_nums[rand_num_idx];//distribution(generator);
    //rand_num_idx = rand_num_idx + 1;



    // get the probabilities of the outputs
    matrix_base<double> pmf = pmfs[sample];

    // determine the random index according to the distribution described by pmf
    int sampled_index=0;
    double prob_sum = 0.0;
    for (int idx=0; idx<pmf.size(); idx++) {
        prob_sum = prob_sum + pmf[idx];
        if ( prob_sum >= rand_num) {
            sampled_index = idx;
            break;
        }
    }

    // copy the new key
    PicState_int64 &new_key = possible_output_states[sample][sampled_index];
    int64_t* sample_data = sample.get_data();
    int64_t* new_key_data = new_key.get_data();

    memcpy( sample_data, new_key_data, sample.size()*sizeof(int64_t));
    sample.number_of_photons = new_key.number_of_photons;
}







/**
@brief Constructor of the class.
@param parameters_in The input state entering the interferometer
@param possible_input_state_in Other possible input states
@return Returns with the instance of the class.
*/
void calculate_weights( tbb::blocked_range<size_t> &r, PicState_int64 &input_state, PicVector<int64_t> &input_state_inidices, concurrent_PicStates& possible_input_states, matrix_base<double> &multinomial_coefficients, double& wieght_norm  ) {



    PicState_int64 corresponding_k_vector = PicState_int64(input_state.size(),0);

    for (auto idx=r.begin(); idx!=r.end(); idx++) {

        PicState_int64 &possible_input_state = possible_input_states[idx];

        // create current k vector
        corresponding_k_vector.number_of_photons = 0;
        for (size_t iidx=0; iidx<input_state_inidices.size(); iidx++) {
            int64_t element = input_state[input_state_inidices[iidx]] - possible_input_state[input_state_inidices[iidx]];
            corresponding_k_vector[input_state_inidices[iidx]] = element;
            corresponding_k_vector.number_of_photons = corresponding_k_vector.number_of_photons + element;
        }

        // calculate_multinomial_coefficient
        double multinomial_coefficient = factorial(corresponding_k_vector.number_of_photons);
        for (size_t iidx=0; iidx<input_state_inidices.size(); iidx++) {
            multinomial_coefficient = multinomial_coefficient/factorial(corresponding_k_vector[input_state_inidices[iidx]]);
        }

        multinomial_coefficients[idx] = multinomial_coefficient;
        wieght_norm = wieght_norm + multinomial_coefficient*multinomial_coefficient;

    }

}




/**
@brief Call to generate possible output state
@param sample The current output sample for which the probabilities are calculated
@param possible_outputs Vector of possible output states
*/
void
generate_output_states( PicState_int64& sample, PicStates &possible_outputs ) {

    int64_t* sample_data = sample.get_data();
    for ( size_t idx=0; idx<possible_outputs.size(); idx++) {
        int64_t* output_data = possible_outputs[idx].get_data();
        memcpy( output_data, sample_data, sample.size()*sizeof(int64_t) );
        output_data[idx] = output_data[idx]+1;
        possible_outputs[idx].number_of_photons = sample.number_of_photons + 1;
    }
}


/**
@brief Call to determine the output probability of associated with the input and output states
@param interferometer_mtx The matrix of the interferometer.
@param input_state The input state.
@param output_state The output state.
*/
double
calculate_outputs_probability(
    matrix &interferometer_mtx,
    PicState_int64 &input_state,
    PicState_int64 &output_state
) {

    matrix modifiedInterferometerMatrix = adaptInterferometer(
        interferometer_mtx,
        input_state,
        output_state
    );

    std::function<bool(int64_t)> filterNonZero = [](int64_t elem) { 
        return elem > 0;
    };

    PicState_int64 adapted_input_state = input_state.filter(filterNonZero);
    PicState_int64 adapted_output_state = output_state.filter(filterNonZero);

    GlynnPermanentCalculatorRepeated permanentCalculatorRecursive;

    Complex16 permanent = permanentCalculatorRecursive.calculate(
        modifiedInterferometerMatrix,
        adapted_input_state,
        adapted_output_state
    );

    double probability =
        permanent.real()*permanent.real() + permanent.imag()*permanent.imag();
        // squared magnitude norm(a+ib) = a^2 + b^2 !!!

    int64_t photon_num = 0;
    for (size_t idx=0; idx<input_state.size(); idx++) {
        photon_num = photon_num + input_state[idx];
    }
    probability = probability/factorial( photon_num );


    for (size_t idx=0; idx<input_state.size(); idx++) {
        probability = probability/factorial( input_state[idx] );
    }

    return probability;
}


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
) {
    int n = interferometerMatrix.rows;

    int64_t sum = 0;
    for (size_t i = 0; i < input_state.size(); i++){
        sum += input_state[i];
    }
    matrix mtx(sum, sum);

    int row_idx = 0;
    for (int i = 0; i < n; i++){
        for (int db_row = 0; db_row < output_state[i]; db_row++){
            int col_idx = 0;
            for (int j = 0; j < n; j++){
                for (int db_col = 0; db_col < input_state[j]; db_col++){
                    mtx[row_idx * mtx.stride + col_idx] =
                        interferometerMatrix[i * interferometerMatrix.stride + j];

                    col_idx++;
                }
            }

            row_idx++;
        }
    }

    return mtx;

}


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
) {
    int sumInput = 0;
    for (int i = 0; i < input_state.size(); i++){
        if (input_state[i] > 0){
            sumInput++;
        }
    }
    int sumOutput = 0;
    for (int i = 0; i < output_state.size(); i++){
        if (output_state[i] > 0){
            sumOutput++;
        }
    }

    matrix new_mtx(sumOutput, sumInput);    

    int n = interferometerMatrix.rows;

    int row_idx = 0;
    for (int i = 0; i < n; i++){
        if (output_state[i] > 0){
            int col_idx = 0;
            for (int j = 0; j < n; j++){
                if (input_state[j] > 0){
                    new_mtx[row_idx * new_mtx.stride + col_idx] =
                        interferometerMatrix[i * interferometerMatrix.stride + j];
                    col_idx++;
                }
            }

            row_idx++;
        }
    }
    

    return new_mtx;

}













} // PIC
