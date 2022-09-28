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

#include "CGeneralizedCliffordsBUniformLossesSimulationStrategy.h"
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

#include <math.h>
#include <tbb/tbb.h>
#include <chrono>
#include <unordered_map>
#include <stdlib.h>
#include <time.h>
#include <iostream>


#ifdef __MPI__
#include <mpi.h>
#endif // MPI


namespace pic {


static double t_perm_accumulator=0.0;
static double t_DFE=0.0;
static double t_DFE_pure=0.0;
static double t_DFE_prepare=0.0;
static double t_CPU_permanent=0.0;
static double t_CPU=0.0;
static double t_CPU_permanent_Glynn=0.0;


CGeneralizedCliffordsBUniformLossesSimulationStrategy::CGeneralizedCliffordsBUniformLossesSimulationStrategy() {
    // seed the random generator
    seed(time(NULL));

    // if there is no transmittance available we set the transmittance to 1.0
    this->transmittance = 1.0;
  
#ifdef __MPI__
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

#endif


}


CGeneralizedCliffordsBUniformLossesSimulationStrategy::CGeneralizedCliffordsBUniformLossesSimulationStrategy( matrix &interferometer_matrix_in, double transmittance, int lib ) {
    this->lib = lib;
    Update_interferometer_matrix( interferometer_matrix_in );

    // seed the random generator
    seed(time(NULL));

    this->transmittance = transmittance;

#ifdef __MPI__
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

#endif



}


CGeneralizedCliffordsBUniformLossesSimulationStrategy::~CGeneralizedCliffordsBUniformLossesSimulationStrategy() {

}


void
CGeneralizedCliffordsBUniformLossesSimulationStrategy::seed(unsigned long long int value) {
    srand(value);
}


void
CGeneralizedCliffordsBUniformLossesSimulationStrategy::Update_interferometer_matrix( matrix &interferometer_matrix_in ) {

    interferometer_matrix = interferometer_matrix_in;
    //perm_accumulator = BatchednPermanentCalculator( interferometer_matrix );

}


std::vector<PicState_int64>
CGeneralizedCliffordsBUniformLossesSimulationStrategy::simulate( PicState_int64 &input_state_in, int samples_number ) {

#ifdef __DFE__
    lock_lib();
    init_dfe_lib(DFE_REP, useDual); 
    out_of_memory = false;  
#endif

    input_state = input_state_in;
    
    calculate_particle_number_probabilities();

    // the k-th element of particle_input_state gives that the k-th photon is on which optical mode
    PicState_int64&& particle_input_state = modes_state_to_particle_state(input_state);


    std::vector<PicState_int64> samples;
    if ( samples_number > 0 ) {    
        // preallocate the memory for the output states
        samples.reserve( samples_number );
#ifdef __MPI__

        int samples_number_per_process = samples_number/world_size;
    
        // calculate the first iteration of the sampling process
        PicState_int64 sample(input_state_in.cols, 0);
        sample.number_of_photons = 0;

        current_input = PicState_int64(sample.size(), 0);
        current_input.number_of_photons = 0;

        working_input_state = particle_input_state.copy();

        int64_t number_of_output_photons = calculate_current_photon_number();//sum(input_state);
        fill_r_sample( sample, number_of_output_photons);
        
        
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



            sample_new = PicState_int64(input_state_in.cols, 0);
            sample_new.number_of_photons = 0;

            current_input = PicState_int64(sample.size(), 0);
            current_input.number_of_photons = 0;

            working_input_state = particle_input_state.copy();

            int64_t number_of_output_photons = calculate_current_photon_number();//sum(input_state);
            fill_r_sample( sample_new, number_of_output_photons);
   


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
            PicState_int64 sample(input_state_in.cols, 0);
            sample.number_of_photons = 0;

            current_input = PicState_int64(sample.size(), 0);
            current_input.number_of_photons = 0;

            working_input_state = particle_input_state.copy();

            int64_t number_of_output_photons = calculate_current_photon_number();//sum(input_state);
            fill_r_sample( sample, number_of_output_photons);
            
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
CGeneralizedCliffordsBUniformLossesSimulationStrategy::fill_r_sample( PicState_int64& sample, int64_t number_of_output_photons ) {



    while (number_of_output_photons > sample.number_of_photons) {
//tbb::tick_count t0 = tbb::tick_count::now();
       
        // randomly pick up an incomming photon and att it to current input state
        update_input_by_single_photon(current_input, working_input_state);

        // calculate new layer of probabilities from which an intermediate (or final) output state is sampled
        matrix_real pmf_local = compute_pmf( interferometer_matrix, sample, current_input );
        
#ifdef __DFE__
        if (out_of_memory) {
            return;
        }
#endif

        // pick a new sample from the possible output states according to the calculated probability distribution stored in pmfs
        sample_from_pmf(sample, pmf_local);
//tbb::tick_count t1 = tbb::tick_count::now();
//std::cout << sample.number_of_photons << " photons from " << number_of_output_photons << " in " << (t1-t0).seconds() << " seconds" << std::endl;

    }

     


}


int64_t
CGeneralizedCliffordsBUniformLossesSimulationStrategy::calculate_current_photon_number() {
    double rand_num = (double)rand()/RAND_MAX;

    int64_t photon_number = 0;

    double weight = particle_number_probabilities[photon_number];

    while (weight < rand_num && weight < 1.0){
        photon_number++;
        weight += particle_number_probabilities[photon_number];
    }

    return photon_number;
}


void CGeneralizedCliffordsBUniformLossesSimulationStrategy::calculate_particle_number_probabilities(){
    int64_t n = sum(input_state);
    
    size_t number_of_possible_particle_number = n + 1;
    particle_number_probabilities = matrix_real(1, number_of_possible_particle_number);

    // probability of a particle remaining in the circuit
    double eta = transmittance * transmittance;

    for (size_t i = 0; i < number_of_possible_particle_number; i++){
        particle_number_probabilities[i] =
            binomialCoeffInt64(n, i) * pow(eta, i) * pow(1.0 - eta, n - i);

    }


#ifdef DEBUG
    double weight_sum = 0;
    for (size_t i = 0; i < number_of_possible_particle_number; i++){
        weight_sum += particle_number_probabilities[i];
    }

    assert(std::abs(weight_sum - 1.0) < 0.0000001);
#endif

} 


}// PIC
