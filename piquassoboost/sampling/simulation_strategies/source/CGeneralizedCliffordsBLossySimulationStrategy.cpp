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
#include "CGeneralizedCliffordsBLossySimulationStrategy.h"
#include "CChinHuhPermanentCalculator.h"
#include "GlynnPermanentCalculatorRepeated.h"
#include "BBFGPermanentCalculatorRepeated.h"
#ifdef __DFE__
#include "GlynnPermanentCalculatorDFE.h"
#include "GlynnPermanentCalculatorRepeatedDFE.h"
#endif
#include "CChinHuhPermanentCalculator.h"
#include "common_functionalities.h"
#include "samplingHelperFunctions.h"
#include "dot.h"
#include <math.h>
#include <tbb/tbb.h>
#include <chrono>

#ifdef __MPI__
#include <mpi.h>
#endif // MPI



extern "C" {
    #ifndef LAPACK_ROW_MAJOR
    #define LAPACK_ROW_MAJOR 101
    #endif

    int LAPACKE_zgesvd(int matrix_layout, char, char, int, int, pic::Complex16 *, int, double *, pic::Complex16 *, int, pic::Complex16 *, int, pic::Complex16 *, int, double*, int* );
}


namespace pic {

static double t_perm_accumulator=0.0;
static double t_DFE=0.0;
static double t_DFE_pure=0.0;
static double t_DFE_prepare=0.0;
static double t_CPU_permanent=0.0;
static double t_CPU=0.0;
static double t_CPU_permanent_Glynn=0.0;

CGeneralizedCliffordsBLossySimulationStrategy::CGeneralizedCliffordsBLossySimulationStrategy() {
    // seed the random generator
    seed(time(NULL));
    
    // default number of approximated modes is 0
    number_of_approximated_modes = 0;

#ifdef __MPI__
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

#endif

}


CGeneralizedCliffordsBLossySimulationStrategy::CGeneralizedCliffordsBLossySimulationStrategy( matrix &interferometer_matrix_in, int lib ) {
    this->lib = lib;
    Update_interferometer_matrix( interferometer_matrix_in );

    // seed the random generator
    seed(time(NULL));

    // default number of approximated modes is 0
    number_of_approximated_modes = 0;

#ifdef __MPI__
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

#endif

}


CGeneralizedCliffordsBLossySimulationStrategy::
CGeneralizedCliffordsBLossySimulationStrategy(
    matrix &interferometer_matrix_in,
    int number_of_approximated_modes_in,
    int lib
) {
    this->lib = lib;
    Update_interferometer_matrix( interferometer_matrix_in );

    // seed the random generator
    seed(time(NULL));

    number_of_approximated_modes = number_of_approximated_modes_in;

#ifdef __MPI__
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

#endif

}


CGeneralizedCliffordsBLossySimulationStrategy::~CGeneralizedCliffordsBLossySimulationStrategy() {

}


void
CGeneralizedCliffordsBLossySimulationStrategy::seed(unsigned long long int value) {
    srand(value);
}


void
CGeneralizedCliffordsBLossySimulationStrategy::set_approximated_modes_number(int value) {
    number_of_approximated_modes = value;
}


void
CGeneralizedCliffordsBLossySimulationStrategy::Update_interferometer_matrix( matrix &interferometer_matrix_in ) {

    interferometer_matrix = interferometer_matrix_in;
    //perm_accumulator = BatchednPermanentCalculator( interferometer_matrix );

}


std::vector<PicState_int64>
CGeneralizedCliffordsBLossySimulationStrategy::simulate( PicState_int64 &input_state, int samples_number ) {

#ifdef __MPI__
    // at MPI just the root process has to extract the losses from the interferometer
    if (current_rank == root_rank){
        extract_losses_from_interferometer();
    }
    MPI_Bcast(
        &uniform_loss,
        1,
        MPI_DOUBLE,
        root_rank,
        MPI_COMM_WORLD
    );

    MPI_Bcast(
        interferometer_matrix.get_data(),
        interferometer_matrix.size() * 2 /*complex values*/,
        MPI_DOUBLE,
        root_rank,
        MPI_COMM_WORLD
    );

#else
    extract_losses_from_interferometer();
#endif


#ifdef __DFE__
    lock_lib();
    init_dfe_lib(DFE_REP, useDual); 
    out_of_memory = false;  
#endif

    // input states with 0's on the first modes which are approximated
    PicState_int64 input_state_without_approximated_modes = input_state.copy();

    particle_number_in_approximated_modes = 0;
    
    for (size_t idx = 0; idx < number_of_approximated_modes; idx++){
        particle_number_in_approximated_modes += input_state[idx];
        input_state_without_approximated_modes[idx] = 0;
    }

    // calculating maximal particle number on one mode
    int maximum_particle_number_in = particle_number_in_approximated_modes;

    for (size_t idx = number_of_approximated_modes; idx < input_state.size(); idx++){
        if (input_state[idx] > maximum_particle_number_in)
            maximum_particle_number_in = input_state[idx];
    }

    calculate_particle_number_probabilities(maximum_particle_number_in);

    std::vector<PicState_int64> samples;
    if ( samples_number > 0 ) {    
        // preallocate the memory for the output states
        samples.reserve( samples_number );

        
#ifdef __MPI__
        int samples_number_per_process = samples_number/world_size;
    
        // calculate the first iteration of the sampling process
        PicState_int64 sample(input_state.cols, 0);
        sample.number_of_photons = 0;

        PicState_int64 current_input = PicState_int64(sample.size(), 0);
        current_input.number_of_photons = 0;

        // randomly generated approximated particle input state
        // the particle input means: the k-th element of it
        // gives that the k-th photon is on which optical mode
        PicState_int64 approximated_particle_input_state =
            compute_lossy_particle_input(
                input_state_without_approximated_modes
            );

        fill_r_sample( sample, current_input, approximated_particle_input_state );
        
        // calculate the individual outputs for the shots and send the calculated outputs to other MPI processes in parallel
        PicState_int64 sample_new;
        for (int idx=1; idx<samples_number_per_process; idx++) {


            std::thread mpi_thread( [&](){

                    // gather the samples over the MPI processes
                    PicState_int64 sample_gathered( sample.size()*world_size );
                    int bytes = sample.size()*sizeof(int64_t);

                    MPI_Allgather(sample.get_data(), bytes, MPI_BYTE, sample_gathered.get_data(), bytes, MPI_BYTE, MPI_COMM_WORLD);

                    for( int rank=0; rank<world_size; rank++) {
                        PicState_int64 sample_local( sample_gathered.get_data()+rank*sample.size(), sample.size() );
                        samples.push_back( sample_local.copy() );
                    }

            });      



            sample_new = PicState_int64(input_state.cols, 0);
            sample_new.number_of_photons = 0;

            PicState_int64 local_current_input = PicState_int64(sample.size(), 0);
            local_current_input.number_of_photons = 0;

            PicState_int64 local_approximated_particle_input_state = compute_lossy_particle_input( input_state_without_approximated_modes );

            fill_r_sample( sample_new, local_current_input, local_approximated_particle_input_state );
   


            // Makes the main thread wait for the mpi thread to finish execution, therefore blocks its own execution.
            mpi_thread.join();
  
    
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
            // randomly generated approximated particle input state
            // the particle input means: the k-th element of it
            // gives that the k-th photon is on which optical mode
            PicState_int64 approximated_particle_input_state =
                compute_lossy_particle_input(
                    input_state_without_approximated_modes
                );

            // current sample to fill
            PicState_int64 sample(input_state.cols, 0);
            sample.number_of_photons = 0;

            // starting with 0 input we always add 1 particle
            PicState_int64 current_input = PicState_int64(sample.size(), 0);
            current_input.number_of_photons = 0;

            PicState_int64 working_input_state = approximated_particle_input_state.copy();

            fill_r_sample( sample, current_input, working_input_state );
            
#ifdef __DFE__
            if (out_of_memory) {
                out_of_memory = false;
                continue;
            }
#endif


            samples.push_back( sample );
        }


#endif
    }

#ifdef __DFE__
    unlock_lib();  
#endif      

    return samples;
}


void
CGeneralizedCliffordsBLossySimulationStrategy::fill_r_sample(
    PicState_int64& sample,
    PicState_int64& current_input,
    PicState_int64& working_particle_input_state
){
    size_t number_of_current_input_photons = working_particle_input_state.size();

    while (number_of_current_input_photons > sample.number_of_photons) {

        // randomly pick up an incoming photon and add it to current input state
        update_input_by_single_photon( current_input, working_particle_input_state );

        // calculate new layer of probabilities from which an intermediate
        // (or final) output state is sampled
        matrix_real pmf_local = compute_pmf( interferometer_matrix, sample, current_input );
        
#ifdef __DFE__
        if (out_of_memory) {
            return;
        }
#endif

        sample_from_pmf( sample, pmf_local);

    }
}


void
CGeneralizedCliffordsBLossySimulationStrategy::calculate_particle_number_probabilities(int maximal_particle_number){
    double eta = 1.0 - uniform_loss;

    size_t number_of_possible_particle_number = maximal_particle_number + 1;

    binomial_weights = std::vector<matrix_real>(number_of_possible_particle_number);

    for (size_t n = 1; n < number_of_possible_particle_number; n++){
        binomial_weights[n] = matrix_real(1, n+1);
            for (size_t i = 0; i < n+1; i++){
                binomial_weights[n][i] =
                    pow(eta, i)
                    * binomialCoeffInt64(n, i)
                    * pow(1.0 - eta, n - i);
        }
    }
} 

void
CGeneralizedCliffordsBLossySimulationStrategy::extract_losses_from_interferometer(){
    size_t n = interferometer_matrix.rows;
    
    matrix u = matrix(n, n);
    matrix_real s = matrix_real(1, n);
    matrix vt = matrix(n, n);

    const size_t rwork_n = 5 * n;
    matrix work = matrix( 1, n );
    matrix_real rwork = matrix_real( 1, rwork_n);

    int info = -99;

    // svd by Lapacke
    LAPACKE_zgesvd(
        LAPACK_ROW_MAJOR,
        'A',
        'A',
        n,
        n,
        interferometer_matrix.get_data(),
        n,
        s.get_data(),
        u.get_data(),
        n,
        vt.get_data(),
        n,
        work.get_data(),
        n,
        rwork.get_data(),
        &info
    );


    matrix_real transmissivities = matrix_real( 1, n );

    double maximal_transmissivity = 0.0;
    for (size_t idx = 0; idx < n; idx++){
        transmissivities[idx] = s[idx] * s[idx];
        
        if (transmissivities[idx] > maximal_transmissivity){
            maximal_transmissivity = transmissivities[idx];
        }
    }
    uniform_loss = 1.0 - maximal_transmissivity;

    for (size_t idx = 0; idx < n; idx++){
        transmissivities[idx] /= maximal_transmissivity;
        s[idx] = std::sqrt(transmissivities[idx]);
    }
    

    // matrix multiplication is not implemented for real and complex matrices yet
    // doing manually
    Complex16 *row_pointer = vt.get_data();
    for (size_t row_idx = 0; row_idx < n; row_idx++){
        double current_transmissivity = s[row_idx];
        for (size_t col_idx = 0; col_idx < n; col_idx++){
            *row_pointer *= current_transmissivity; // multiplying with corresponding sigma value
            row_pointer++; // jumping to next value
        }
    }
    
    interferometer_matrix = dot(u, vt);
}

PicState_int64
CGeneralizedCliffordsBLossySimulationStrategy::
compute_lossy_particle_input(PicState_int64 &input_state_without_approx_modes){

    PicState_int64 lossy_input_state = input_state_without_approx_modes.copy();

    // fill all particles into one random mode on the approximated modes
    if (number_of_approximated_modes > 0){
        size_t mode_index_of_approximation = rand() % number_of_approximated_modes;

        lossy_input_state[mode_index_of_approximation] = random_particle_number(particle_number_in_approximated_modes);
    }

    // update particle numbers on all modes after the approximated modes
    if (uniform_loss > 0.0){
        for (size_t idx = number_of_approximated_modes; idx < lossy_input_state.size(); idx++){
            if (lossy_input_state[idx] > 0){
                lossy_input_state[idx] = random_particle_number(lossy_input_state[idx]);
            }
        }

    }

    size_t number_of_input_photons = sum(lossy_input_state);

    PicState_int64 lossy_particle_input_state =
        PicState_int64(number_of_input_photons);

    size_t index = 0;
    for ( size_t mode_idx=0; mode_idx<lossy_input_state.size(); mode_idx++ ) {
        for(
            int64_t particle_index=0;
            particle_index < lossy_input_state[mode_idx];
            particle_index++
        ) {
            lossy_particle_input_state[index] = mode_idx;
            index++;
        }
    }
    
    return lossy_particle_input_state;
}

size_t CGeneralizedCliffordsBLossySimulationStrategy::random_particle_number(size_t maximal_particle_number){

    double rand_num = (double)rand()/RAND_MAX;
    
    size_t idx = 0;
    double *weights = binomial_weights[maximal_particle_number].get_data();
    double weight = *weights;
    weights++;
    while(weight < rand_num){
        weight += *weights;
        weights++;
        idx++;
    }

    return idx;
}


void CGeneralizedCliffordsBLossySimulationStrategy::update_matrix_for_approximate_sampling(){
    matrix qft = quantum_fourier_transform_matrix(number_of_approximated_modes);

    matrix random_phases = random_phases_vector(number_of_approximated_modes);

    Complex16 *qft_data = qft.get_data();
    // interferometer_matrix = interferometer_matrix @ random_phases_extended @ qft_extended python code
    // multiplying by a diagonal matrix from left means multiplying lines by lines with the diagonal elements
    for (size_t row_idx = 0; row_idx < qft.rows; row_idx++){
        Complex16 *current_row = qft_data + row_idx * qft.stride;
        for (size_t col_idx = 0; col_idx < qft.cols; col_idx++){
            current_row[col_idx] *= random_phases[row_idx];
        }
    }

    matrix new_interferometer = matrix(
        interferometer_matrix.rows,
        interferometer_matrix.cols
    );

    Complex16 *interferometer_data = interferometer_matrix.get_data();
    // multiplying interferometer with modified qft with additional identity block matrix
    for (size_t row_idx = 0; row_idx < interferometer_matrix.rows; row_idx++){
        Complex16 *interferometer_current_row = interferometer_data + row_idx * interferometer_matrix.stride;
        // just the first qft.cols elements has to be updated
        for (size_t col_idx = 0; col_idx < qft.cols; col_idx++){
            Complex16 *new_interferometer_elem = new_interferometer.get_data() + row_idx * new_interferometer.stride + col_idx;
            *new_interferometer_elem = 0.0;
            for (size_t k = 0; k < qft.cols; k++){
                *new_interferometer_elem +=
                    interferometer_current_row[k]
                    * qft[k * qft.stride + col_idx];
            } 
        }
        // outside of the first qft.cols elements the interferometer row remains the same
        for (size_t col_idx = qft.cols; col_idx < interferometer_matrix.cols; col_idx++){
                new_interferometer[row_idx * new_interferometer.stride + col_idx] =
                    interferometer_current_row[col_idx];
        }
    }

    // update interferometer matrix
    interferometer_matrix = new_interferometer;
}


} // PIC
