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


CGeneralizedCliffordsBLossySimulationStrategy::CGeneralizedCliffordsBLossySimulationStrategy() {
    // seed the random generator
    seed(time(NULL));
    
    // default number of approximated modes is 0
    number_of_approximated_modes = 0;

#ifdef __MPI__
    int done_already;
    MPI_Initialized(&done_already);
    if (!done_already)
        MPI_Init(NULL, NULL);
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
    int done_already;
    MPI_Initialized(&done_already);
    if (!done_already)
        MPI_Init(NULL, NULL);
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
    int done_already;
    MPI_Initialized(&done_already);
    if (!done_already)
        MPI_Init(NULL, NULL);
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
CGeneralizedCliffordsBLossySimulationStrategy::
simulate( PicState_int64 &input_state, int samples_number ) {
    std::cout << "CGeneralizedCliffordsBLossySimulationStrategy::simulate\n";
    std::cout << "First input: ";
    for (int i = 0; i < input_state.size(); i++){
        std::cout << input_state[i] <<" ";
    }
    std::cout << std::endl;

    std::cout << "Approximated modes number: "<<number_of_approximated_modes<<std::endl;

    extract_losses_from_interferometer();

#ifdef __DFE__
    lock_lib();
    init_dfe_lib(DFE_MAIN, useDual); 
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

    size_t counter = 0;
    std::cout << "Particle number probabilities\n";
    for (auto&& weights : binomial_weights){
        for (int i = 0; i < weights.size(); i++){
            std::cout << counter << " "<<i<<":"<<weights[i]<<std::endl; 
        }
        counter++;
    }

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

        std::cout << "samples_number_per_process: "<<samples_number_per_process<<std::endl;
        
        // calculate the individual outputs for the shots and send the calculated outputs to other MPI processes in parallel
        PicState_int64 sample_new;
        for (int idx=1; idx<samples_number_per_process; idx++) {
    
            tbb::parallel_invoke(
    
                [&]{
                    sample_new = PicState_int64(input_state.cols, 0);
                    sample_new.number_of_photons = 0;

                    PicState_int64 local_current_input = PicState_int64(sample.size(), 0);
                    local_current_input.number_of_photons = 0;

                    PicState_int64 local_approximated_particle_input_state =
                        compute_lossy_particle_input(
                            input_state_without_approximated_modes
                        );

                    fill_r_sample( sample_new, local_current_input, local_approximated_particle_input_state );

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

            std::cout << "Input: ";
            for (int i = 0; i < current_input.size(); i++)
                std::cout << current_input[i] << " ";
            std::cout << std::endl;

            std::cout << "Output: ";
            for (int i = 0; i < sample.size(); i++)
                std::cout << sample[i] << " ";
            std::cout << std::endl;


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
        matrix_real current_pmf = compute_pmf( sample, current_input );
        
#ifdef __DFE__
        if (out_of_memory) {
            return;
        }
#endif

        sample_from_pmf(current_pmf, sample);

    }

}


matrix_real
CGeneralizedCliffordsBLossySimulationStrategy::compute_pmf( PicState_int64& sample, PicState_int64& current_input ) {

    std::cout << "compute_pmf is running with\n";
    std::cout << "Sample:";
    sample.print_matrix();
    std::cout << "current:";
    current_input.print_matrix();
    

    // reset the previously calculated pmf layer
    matrix_real pmf = matrix_real(1, sample.size());

    // define function to filter out nonzero elements
    std::function<bool(int64_t)> filterNonZero = [](int64_t elem) { 
        return elem > 0;
    };

    // total sum of probabilities
    double probability_sum = 0.0;


    std::vector<matrix> matrices;
    matrices.reserve( current_input.size() );

    // calculate permanents of submatrices
    matrix permanent_addends(1, current_input.size());
    memset( permanent_addends.get_data(), 0.0, permanent_addends.size()*sizeof(Complex16) );

    matrix permanent_addends_tmp(1, current_input.size());
    memset( permanent_addends_tmp.get_data(), 0.0, permanent_addends_tmp.size()*sizeof(Complex16) );

//#ifdef __DFE__


    // determine the number of nonzero elements in the current input/output
    size_t nonzero_output_elements = 0;
    for (size_t jdx=0; jdx<sample.size(); jdx++) {      

        if ( sample[jdx] > 0 ) {
            nonzero_output_elements++;
        }
    }


    

    std::vector<unsigned char> colIndices;
    colIndices.reserve(current_input.size());
    for (size_t i = 0; i < current_input.size(); i++) {
        if ( current_input[i] > 0 ) {
            colIndices.push_back(i);        
        }
    }

#ifdef __DFE__
    const size_t nonzero_output_elements_threshold = 10; 
//double photon_density = (double)(sample.number_of_photons)/nonzero_output_elements;
//std::cout << nonzero_output_elements << " " << photon_density << " " << effective_dim <<std::endl;

    if ( nonzero_output_elements < nonzero_output_elements_threshold || input_state.number_of_photons > 36 ) {
#endif


        //GlynnPermanentCalculatorRepeatedLongDouble permanentCalculator;
        GlynnPermanentCalculatorRepeatedDouble permanentCalculator;
        BBFGPermanentCalculatorRepeated BBFGpermanentCalculator;

        tbb::parallel_for( (size_t)0, colIndices.size(), (size_t)1, [&](size_t idx) {
        //for (size_t idx=0; idx<colIndices.size(); idx++) {

            // remove a photon from the input state
            PicState_int64&& input_state_loc = current_input.copy();
            input_state_loc[colIndices[idx]]--;  
            input_state_loc.number_of_photons--; 

            
            matrix&& modifiedInterferometerMatrix = adaptInterferometer( interferometer_matrix, input_state_loc, sample );
            PicState_int64 adapted_input_state = input_state_loc.filter(filterNonZero);
            PicState_int64 adapted_output_state = sample.filter(filterNonZero);
            std::cout << "permanent calculated from\n";
            std::cout <<"adapted_input_state";
            adapted_input_state.print_matrix();
            std::cout <<"adapted_output_state";
            adapted_output_state.print_matrix();
            permanent_addends[colIndices[idx]] = BBFGpermanentCalculator.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state, false);


        //}
        });
#ifdef __DFE__
    }
    else {


        
        // split the work between CPU and DFE
        size_t idx_max_CPU = colIndices.size()/6;
        
        tbb::parallel_invoke( 
        
            [&](){
            
                tbb::parallel_for( (size_t)0, idx_max_CPU, (size_t)1, [&](size_t idx) {            
                    PicState_int64&& input_state_loc = current_input.copy();
                    input_state_loc[colIndices[idx]]--;  
                    input_state_loc.number_of_photons--; 

                    BBFGPermanentCalculatorRepeated permanentCalculator;
                    matrix&& modifiedInterferometerMatrix = adaptInterferometer( interferometer_matrix, input_state_loc, sample );
                    PicState_int64 adapted_input_state = input_state_loc.filter(filterNonZero);
                    PicState_int64 adapted_output_state = sample.filter(filterNonZero);

                    permanent_addends_tmp[colIndices[idx]] = permanentCalculator.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state);     
                    
                });     
            
            },
            [&](){
        

                const int useDual = 0;
                const int useFloat = 0;


                // reduce the columns of the interferometer matrix according to the input states
                matrix column_reduced_matrix(interferometer_matrix.rows, colIndices.size());
                PicState_int64 reduced_input_state(colIndices.size());
                for( size_t row_idx=0; row_idx<interferometer_matrix.rows; row_idx++) {

                    size_t row_offset_reduced = row_idx*column_reduced_matrix.stride;
                    size_t row_offset = row_idx*interferometer_matrix.stride;

                    for( size_t idx=0; idx<colIndices.size(); idx++) {
                        column_reduced_matrix[ row_offset_reduced + idx ] = interferometer_matrix[row_offset+colIndices[idx]];
                    }
                }

                for( size_t idx=0; idx<colIndices.size(); idx++) {
                    reduced_input_state[idx] = input_state[colIndices[idx]];
                }
                reduced_input_state.number_of_photons = input_state.number_of_photons;


                // create storage for batched input states
                std::vector<std::vector<PicState_int64>> batched_input_states;
                batched_input_states.resize(1);
                std::vector<PicState_int64>& input_states_DFE = batched_input_states[0];
                input_states_DFE.reserve(colIndices.size()-idx_max_CPU);

                // create storage for the calculated permanents
                std::vector<std::vector<Complex16>> batched_perms;
                batched_perms.resize(1);
                std::vector<Complex16>& perms_DFE = batched_perms[0];
                perms_DFE.resize(colIndices.size()-idx_max_CPU);

                //create storage for the batched output states
                std::vector<PicState_int64> output_states_DFE;
                output_states_DFE.push_back(sample);        

                for (size_t idx=idx_max_CPU; idx<colIndices.size(); idx++) {

                    // remove a photon from the input state



                    tbb::tick_count t0c = tbb::tick_count::now();////////////////////////// 
                    PicState_int64&& input_state_loc_tmp = reduced_input_state.copy();
                    input_state_loc_tmp[idx]--;  
                    input_state_loc_tmp.number_of_photons--; 
                    input_states_DFE.push_back(input_state_loc_tmp);

                    tbb::tick_count t1c = tbb::tick_count::now();////////////////////////// 
                    t_DFE_prepare += (t1c-t0c).seconds(); //////////////////////////   
/*
                    Complex16& perm = permanent_addends[colIndices[idx]];
                    GlynnPermanentCalculatorRepeated_DFE(interferometer_matrix, input_state_loc, sample, perm, useDual, useFloat);
*/
/*
                    tbb::tick_count t0 = tbb::tick_count::now();//////////////////////////
                    PicState_int64&& input_state_loc = current_input.copy();
                    input_state_loc[colIndices[idx]]--;  
                    input_state_loc.number_of_photons--; 

                    BBFGPermanentCalculatorRepeated permanentCalculator;
                    matrix&& modifiedInterferometerMatrix = adaptInterferometer( interferometer_matrix, input_state_loc, sample );
                    PicState_int64 adapted_input_state = input_state_loc.filter(filterNonZero);
                    PicState_int64 adapted_output_state = sample.filter(filterNonZero);

                    permanent_addends_tmp[colIndices[idx]] = permanentCalculator.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state); 
                    tbb::tick_count t1 = tbb::tick_count::now();////////////////////////// 
                    t_CPU_permanent += (t1-t0).seconds();    ////////////////////////// 
*/
                }


                tbb::tick_count t0b = tbb::tick_count::now();////////////////////////// 

                GlynnPermanentCalculatorRepeatedInputBatch_DFE(column_reduced_matrix, batched_input_states, output_states_DFE, batched_perms, useDual, useFloat);
                for (size_t idx=idx_max_CPU; idx<colIndices.size(); idx++) {

                    permanent_addends[colIndices[idx]] = perms_DFE[idx-idx_max_CPU];
/*
                    if ( std::norm( permanent_addends[colIndices[idx]] - permanent_addends_tmp[colIndices[idx]] )/std::norm( permanent_addends[colIndices[idx]]) > 1e-3 ) {
                        std::cout << "difference in idx=" << idx << " " << permanent_addends[colIndices[idx]] << " " << permanent_addends_tmp[colIndices[idx]] <<  std::endl;
                    }  
*/
                }
    

                tbb::tick_count t1b = tbb::tick_count::now();////////////////////////// 
                t_DFE += (t1b-t0b).seconds(); ////////////////////////// 

        });

    }

#endif
     


    // calculate probabilities by taking into account the a new particle can come in any new mode
    for (size_t mdx=0; mdx<sample.size(); mdx++ ) {

        Complex16 permanent(0.0,0.0);
        for (size_t idx=0; idx<current_input.size(); idx++) {
            permanent = current_input[idx] == 0 ? permanent : permanent + current_input[idx]*permanent_addends[idx]*interferometer_matrix[mdx*interferometer_matrix.stride + idx];
        }
/*
        
        // add a photon to the current output state
        PicState_int64&& output_state = sample.copy();
        output_state[mdx]++;

        GlynnPermanentCalculatorRepeatedLongDouble permanentCalculator;

        matrix&& modifiedInterferometerMatrix = adaptInterferometer( interferometer_matrix, current_input, output_state );
        PicState_int64 adapted_input_state = current_input.filter(filterNonZero);
        PicState_int64 adapted_output_state = output_state.filter(filterNonZero);    
       
        Complex16 permanent = permanentCalculator.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state);
*/


        pmf[mdx] = permanent.real()*permanent.real() + permanent.imag()*permanent.imag();
        probability_sum += pmf[mdx];

    }


    // renormalize the probability layer
    for (size_t mdx=0; mdx<sample.size(); mdx++ ) {
        pmf[mdx] = pmf[mdx]/probability_sum;
    }

    return pmf;

}


void 
CGeneralizedCliffordsBLossySimulationStrategy::
update_input_by_single_photon(
    PicState_int64& current_input,
    PicState_int64& working_input_state
) {

    if ( working_input_state.size() == 0 ) {
        std::string error("CGeneralizedCliffordsBSimulationStrategy::update_input_by_single_photon:  the size of working_input_state is zero");
        throw error;
    }

    // determine a random index
    size_t rand_index = rand() % working_input_state.size();

    current_input[working_input_state[rand_index]]++;
    current_input.number_of_photons++;

    // remove an item from working_input_state 
    PicState_int64 working_input_state_reduced(working_input_state.size()-1);
    if ( rand_index>0 ) {
        memcpy( working_input_state_reduced.get_data(), working_input_state.get_data(), rand_index*sizeof(int64_t) );
    }
    if ( rand_index < working_input_state.size()-1 ) {
        memcpy( working_input_state_reduced.get_data()+rand_index, working_input_state.get_data()+rand_index+1, (working_input_state.size()-rand_index-1)*sizeof(int64_t) );
    }

    // replace the working input state with the reduced one  
    working_input_state = working_input_state_reduced;

}


void
CGeneralizedCliffordsBLossySimulationStrategy::
sample_from_pmf( matrix_real &pmf, PicState_int64 &sample ) {

    // create a random double
    double rand_num = (double)rand()/RAND_MAX;
   //double rand_num = rand_nums[rand_num_idx];//distribution(generator);
    //rand_num_idx = rand_num_idx + 1;

  

    // determine the random index according to the distribution described by pmf
    size_t sampled_index=0;
    double prob_sum = 0.0;
    for (size_t idx=0; idx<pmf.size(); idx++) {
        prob_sum = prob_sum + pmf[idx];
        if ( prob_sum >= rand_num) {
            sampled_index = idx;
            break;
        }
    }

    sample[sampled_index]++;
    sample.number_of_photons++;
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
    interferometer_matrix.print_matrix();
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
    std::cout << "lossy input: ";
    for(int i = 0; i < lossy_input_state.size(); i++){
        std::cout << lossy_input_state[i]<<" ";
    }
    std::cout <<std::endl;

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

size_t CGeneralizedCliffordsBLossySimulationStrategy::random_particle_number(size_t particle_number){

    double rand_num = (double)rand()/RAND_MAX;
    
    size_t idx = 0;
    double *weights = binomial_weights[particle_number].get_data();
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


matrix quantum_fourier_transform_matrix(size_t n){
    Complex16 j = Complex16(0.0, 1.0);
    Complex16 e = M_E;
    auto omega_std = pow(e, j * 2.0 * M_PI / (double)n);
    Complex16 omega = Complex16(omega_std.real(), omega_std.imag());

    matrix qft(n, n);

    double sqrt_of_n = std::sqrt(n);

    for (size_t row_idx = 0; row_idx < n; row_idx++){
        for (size_t col_idx = 0; col_idx < n; col_idx++){
            qft[row_idx * qft.stride + col_idx] =
                pow(omega, row_idx * col_idx) / sqrt_of_n;
        }
    }

    return qft;
}


matrix random_phases_vector(size_t n){
    matrix phases = matrix(n, 1);

    Complex16 j = Complex16(0.0, 1.0);
    Complex16 e = M_E;

    for (size_t idx = 0; idx < n; idx++){
        double rand_num = (double)rand()/RAND_MAX;

        auto omega_std = pow(e, j * 2.0 * M_PI / rand_num);
        Complex16 omega = Complex16(omega_std.real(), omega_std.imag());

        phases[idx] = omega;
    }

    return phases;
}



} // PIC
