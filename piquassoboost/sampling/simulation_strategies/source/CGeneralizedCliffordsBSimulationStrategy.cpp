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
#include "CGeneralizedCliffordsBSimulationStrategy.h"
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

#ifdef __MPI__
#include <mpi.h>
#endif // MPI

namespace pic {
/*
    double rand_nums[40] = {0.929965, 0.961441, 0.46097, 0.090787, 0.137104, 0.499059, 0.951187, 0.373533, 0.634074, 0.0886671, 0.0856861, 0.999702, 0.419755, 0.376557, 0.947568, 0.705106, 0.0520666, 0.45318,
            0.874288, 0.656594, 0.287817, 0.484918, 0.854716, 0.31408, 0.516911, 0.374158, 0.0124914, 0.878496, 0.322593, 0.699271, 0.0583747, 0.56629, 0.195314, 0.00059639, 0.443711, 0.652659, 0.350379, 0.839752, 0.710161, 0.28553};
    int rand_num_idx = 0;
*/


static double t_perm_accumulator=0.0;
static double t_DFE=0.0;
static double t_DFE_pure=0.0;
static double t_DFE_prepare=0.0;
static double t_CPU_permanent=0.0;
static double t_CPU=0.0;
static double t_CPU_permanent_Glynn=0.0;



/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
CGeneralizedCliffordsBSimulationStrategy::CGeneralizedCliffordsBSimulationStrategy() {
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
CGeneralizedCliffordsBSimulationStrategy::CGeneralizedCliffordsBSimulationStrategy( matrix &interferometer_matrix_in, int lib ) {
    this->lib = lib;
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
CGeneralizedCliffordsBSimulationStrategy::~CGeneralizedCliffordsBSimulationStrategy() {

}

/**
@brief Seeds the simulation with a specified value
@param value The value to seed with
*/
void
CGeneralizedCliffordsBSimulationStrategy::seed(unsigned long long int value) {
    srand(value);
}

/**
@brief Call to update the memor address of the stored matrix iinterferometer_matrix
@param interferometer_matrix_in The matrix describing the interferometer
*/
void
CGeneralizedCliffordsBSimulationStrategy::Update_interferometer_matrix( matrix &interferometer_matrix_in ) {

    interferometer_matrix = interferometer_matrix_in;
    //perm_accumulator = BatchednPermanentCalculator( interferometer_matrix );

}



/**
@brief Call to determine the resultant state after traversing through linear interferometer.
@param input_state_in The input state.
@return Returns with the resultant state after traversing through linear interferometer.
*/
std::vector<PicState_int64>
CGeneralizedCliffordsBSimulationStrategy::simulate( PicState_int64 &input_state_in, int samples_number ) {

#ifdef __DFE__
    lock_lib();
    init_dfe_lib(DFE_REP, useDual); 
    out_of_memory = false;  
#endif

    input_state = input_state_in;
    int64_t number_of_input_photons = sum(input_state);

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

        fill_r_sample( sample, number_of_input_photons );
              
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

            fill_r_sample( sample_new, number_of_input_photons );
            
#ifdef __DFE__
            if (out_of_memory) {
                out_of_memory = false;
                continue;
            }
#endif            
   


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
tbb::tick_count t0cpu = tbb::tick_count::now();
            PicState_int64 sample(input_state_in.cols, 0);
            sample.number_of_photons = 0;

            current_input = PicState_int64(sample.size(), 0);
            current_input.number_of_photons = 0;

            working_input_state = particle_input_state.copy();

            fill_r_sample( sample, number_of_input_photons );
            
#ifdef __DFE__
            if (out_of_memory) {
                out_of_memory = false;
                continue;
            }
#endif

            samples.push_back( sample );
//std::cout << "sample: " << idx+1 << std::endl;
//sample.print_matrix();
tbb::tick_count t1cpu = tbb::tick_count::now();
t_CPU += (t1cpu-t0cpu).seconds();            
//std::cout << "DFE all time: " << t_DFE << ", cpu permanent: " << t_CPU_permanent << " " << t_CPU_permanent_Glynn << std::endl;
//std::cout << "DFE_pure time: " << t_DFE_pure << std::endl;
//std::cout << "DFE_prepare time: " << t_DFE_prepare << std::endl;
//std::cout << idx << " total sampling time: " << t_CPU << std::endl;

        }


#endif
    }

#ifdef __DFE__
    unlock_lib();  
#endif      

    return samples;
}



/**
@brief Call to calculate and fill the output states for the individual shots.
@param sample The current sample state represented by a PicState_int64 class
*/
void
CGeneralizedCliffordsBSimulationStrategy::fill_r_sample( PicState_int64& sample, int64_t number_of_input_photons ) {



    while (number_of_input_photons > sample.number_of_photons) {
//tbb::tick_count t0 = tbb::tick_count::now();
       
        // randomly pick up an incomming photon and add it to current input state
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
//std::cout << sample.number_of_photons << " photons from " << number_of_input_photons << " in " << (t1-t0).seconds() << " seconds" << std::endl;

    }

     


}


/**
@brief Call to give mode-basis state in particle basis
*/
PicState_int64 modes_state_to_particle_state( const PicState_int64& mode_state ) {


    int64_t particles_number = sum(mode_state);
    PicState_int64 particles_state(particles_number);

    size_t index = 0;
    for ( size_t mode_idx=0; mode_idx<mode_state.size(); mode_idx++ ) {
    
        for( int64_t photon_idx=0; photon_idx<mode_state[mode_idx]; photon_idx++ ) {
            particles_state[index] = mode_idx;
            index++;
        }
    }

    return particles_state;
}

















} // PIC
