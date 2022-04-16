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
#include "GlynnPermanentCalculator.h"
#include "GlynnPermanentCalculatorDFE.h"
#include "GlynnPermanentCalculatorRepeated.h"
#ifdef __DFE__
#include "GlynnPermanentCalculatorRepeatedDFE.h"
#endif
#include "CChinHuhPermanentCalculator.h"
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


static double t_perm_accumulator=0.0;
static double t_DFE=0.0;
static double t_CPU_permanent;
static double t_CPU=0.0;



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
    perm_accumulator = BatchednPermanentCalculator( interferometer_matrix );

}



/**
@brief Call to determine the resultant state after traversing through linear interferometer.
@param input_state_in The input state.
@return Returns with the resultant state after traversing through linear interferometer.
*/
std::vector<PicState_int64>
CGeneralizedCliffordsBSimulationStrategy::simulate( PicState_int64 &input_state_in, int samples_number ) {

    input_state = input_state_in;
    number_of_input_photons = sum(input_state);

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
tbb::tick_count t0cpu = tbb::tick_count::now();
            PicState_int64 sample(input_state_in.cols, 0);
            sample.number_of_photons = 0;

            current_input = PicState_int64(sample.size(), 0);
            current_input.number_of_photons = 0;

            working_input_state = particle_input_state.copy();

            fill_r_sample( sample );

            samples.push_back( sample );
tbb::tick_count t1cpu = tbb::tick_count::now();
t_CPU += (t1cpu-t0cpu).seconds();            
std::cout << "DFE time: " << t_DFE << ", cpu permanent: " << t_CPU_permanent << std::endl;
std::cout << "CPU time: " << t_CPU << std::endl;
        }


#endif
    }




    return samples;
}



/**
@brief Call to calculate and fill the output states for the individual shots.
@param sample The current sample state represented by a PicState_int64 class
*/
void
CGeneralizedCliffordsBSimulationStrategy::fill_r_sample( PicState_int64& sample ) {





    while (number_of_input_photons > sample.number_of_photons) {
//tbb::tick_count t0 = tbb::tick_count::now();
       
        // randomly pick up an incomming photon and att it to current input state
        update_current_input();

        // calculate new layer of probabilities from which an intermediate (or final) output state is sampled
        compute_pmf( sample );

        // pick a new sample from the possible output states according to the calculated probability distribution stored in pmfs
        sample_from_pmf(sample);
//tbb::tick_count t1 = tbb::tick_count::now();
//std::cout << sample.number_of_photons << " photons in " << (t1-t0).seconds() << std::endl;

    }



}


/**
@brief Call to calculate new layer of probabilities from which an intermediate (or final) output state is sampled
*/
void 
CGeneralizedCliffordsBSimulationStrategy::compute_pmf( PicState_int64& sample ) {


    // reset the previously calculated pmf layer
    pmf = matrix_real(1, sample.size());


    // define function to filter out nonzero elements
    std::function<bool(int64_t)> filterNonZero = [](int64_t elem) { 
        return elem > 0;
    };

    // total sum of probabilities
    double probability_sum = 0.0;
/*
interferometer_matrix.print_matrix();
sample.print_matrix();
current_input.print_matrix();
*/
/*
    std::vector<PicState_int64> input_states;
    input_states.reserve(current_input.size());



    // clear the permanent accumulator
    perm_accumulator.clear();
    perm_accumulator.reserve_space( current_input.size());

    

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, current_input.size()), [&](tbb::blocked_range<size_t> r ) {
        for( size_t jdx=r.begin(); jdx!=r.end(); jdx++) {

            // add a photon to the current output state
            PicState_int64&& input_state_loc = current_input.copy();
            if (input_state_loc[idx]>0) {
                input_state_loc[idx]--;
         
                // accumulate input/output states for permanent calculations
                perm_accumulator.add(&input_state_loc, &sample, index_offset+jdx);

        }

    });
*/

    std::vector<matrix> matrices;
    matrices.reserve( current_input.size() );

    // calculate permanents of submatrices
    matrix permanent_addends(1, current_input.size());

#ifdef __DFE__
    // determine the number of nonzero elements in the sample
    size_t nonzero_elements = 0;
    for (size_t jdx=0; jdx<sample.size(); jdx++) {
        nonzero_elements = sample[jdx] > 0 ? nonzero_elements : nonzero_elements+1;
    }
#endif


    for (size_t idx=0; idx<current_input.size(); idx++) {
        //GlynnPermanentCalculator permanentCalculator;  
        GlynnPermanentCalculatorRepeated permanentCalculator;  

 
        // add a photon to the current output state
        PicState_int64&& input_state_loc = current_input.copy();
        if (input_state_loc[idx]>0) {
            input_state_loc[idx]--;
            input_state_loc.number_of_photons--;

#ifdef __DFE__
            if ( nonzero_elements < 7 ) {
#endif
/*
                matrix&& modifiedInterferometerMatrix = adaptInterferometerGlynnMultiplied(interferometer_matrix, &input_state_loc, &sample );
                permanent_addends[idx] = permanentCalculator.calculate( modifiedInterferometerMatrix  );
*/               
            
                matrix&& modifiedInterferometerMatrix = adaptInterferometer( interferometer_matrix, input_state_loc, sample );
                PicState_int64 adapted_input_state = input_state_loc.filter(filterNonZero);
                PicState_int64 adapted_output_state = sample.filter(filterNonZero);
                permanent_addends[idx] = permanentCalculator.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state);  
#ifdef __DFE__
           }
           else { 
tbb::tick_count t0 = tbb::tick_count::now();
                int useDual = 0;
                //GlynnPermanentCalculatorRepeatedMulti_DFE(interferometer_matrix, input_state_loc, sample, permanent_addends[idx], useDual);

                int useFloat = 0; 

                bool doCPU = false;
                const size_t numinits = 4;
                matrix_base<ComplexFix16> mtxfix[numinits] = {};
                matrix_base<long double> renormalize_data_all;
                uint8_t onerows = 0;
                size_t photons = 0;
                uint64_t totalPerms = 0;
                std::vector<uint64_t> mplicity;
                uint8_t mulsum = 0;

                prepareDataForRepeatedMulti_DFE(interferometer_matrix, input_state_loc, sample, useFloat, &mtxfix[0], renormalize_data_all, mplicity, onerows, photons, totalPerms, mulsum, doCPU);
                GlynnPermanentCalculatorRepeatedMulti_DFE(interferometer_matrix, input_state_loc, sample, useFloat, mtxfix, renormalize_data_all, mplicity, onerows, photons, totalPerms, mulsum, doCPU, useDual, permanent_addends[idx]);

tbb::tick_count t1 = tbb::tick_count::now();
t_DFE += (t1-t0).seconds();          

tbb::tick_count t0cpu = tbb::tick_count::now();
                matrix&& modifiedInterferometerMatrix = adaptInterferometer( interferometer_matrix, input_state_loc, sample );
                PicState_int64 adapted_input_state = input_state_loc.filter(filterNonZero);
                PicState_int64 adapted_output_state = sample.filter(filterNonZero);
                Complex16 permanent_tmp = permanentCalculator.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state);  
tbb::tick_count t1cpu = tbb::tick_count::now();
t_CPU_permanent += (t1cpu-t0cpu).seconds();
if ( std::norm( permanent_addends[idx] - permanent_tmp ) > 1e-3 ) {
 std::cout << permanent_addends[idx] << " " << permanent_tmp << std::endl;
}
permanent_addends[idx] = permanent_tmp;
//std::cout << "DFE permanent " << std::endl;
                //matrix&& modifiedInterferometerMatrix = adaptInterferometerGlynnMultiplied(interferometer_matrix, &input_state_loc, &sample );
                //GlynnPermanentCalculator_DFE(modifiedInterferometerMatrix, permanent_addends[idx], 1, 0);

           }
#endif

        }
        else {
            permanent_addends[idx] = Complex16(0.0,0.0);
        }
    }


   


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

        GlynnPermanentCalculatorRepeated permanentCalculator;

        matrix&& modifiedInterferometerMatrix = adaptInterferometer( interferometer_matrix, current_input, output_state );
        PicState_int64 adapted_input_state = current_input.filter(filterNonZero);
        PicState_int64 adapted_output_state = output_state.filter(filterNonZero);    
       
        Complex16 permanent = permanentCalculator.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state);
*/
/*
modifiedInterferometerMatrix.print_matrix();
std::cout << permanent_b << " "  << permanent << std::endl;
if ( std::norm( permanent-permanent_b)> 1e-3 ) abort();
*/

        pmf[mdx] = permanent.real()*permanent.real() + permanent.imag()*permanent.imag();
        probability_sum += pmf[mdx];

    }


    // renormalize the probability layer
    for (size_t mdx=0; mdx<sample.size(); mdx++ ) {
        pmf[mdx] = pmf[mdx]/probability_sum;
    }

}




/**
@brief Call to randomly increase the current input state by a single photon
*/
void 
CGeneralizedCliffordsBSimulationStrategy::update_current_input() {

    if ( working_input_state.size() == 0 ) {
        std::string error("CGeneralizedCliffordsBSimulationStrategy::update_current_input:  the size of working_input_state is zero");
        throw error;
    }

    // determine a random index
    int rand_index = rand() % working_input_state.size();

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
/*
std::cout << rand_index << std::endl;
working_input_state.print_matrix();
working_input_state_reduced.print_matrix();
*/
    // replace the working input state with the reduced one  
    working_input_state = working_input_state_reduced;
    

    

/*9
    current_input[self._working_input_state.pop(
            randint(0, len(self._working_input_state)))] += 1
*/
}




/**
@brief Call to calculate a new layer of probabilities of the possible output states
@param sample A preallocated PicState_int64 for one output
@param possible_outputs A preallocated vector of possible output states
*/
/*
void
CGeneralizedCliffordsBSimulationStrategy::calculate_new_layer_of_pmfs( PicState_int64& sample, PicStates &possible_outputs ) {

    int64_t number_of_particle_to_sample = sample.number_of_photons + 1;
    concurrent_PicStates &possible_input_states = labeled_states[number_of_particle_to_sample];

    // parallel loop to calculate the weights from possible input states and their weight
    matrix_base<double> multinomial_coefficients = matrix_base<double>(1,possible_input_states.size());
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

    // clear the permanent accumulator
    perm_accumulator.clear();
    perm_accumulator.reserve_space( possible_outputs.size() * possible_input_states.size());


tbb::tick_count t0cpu = tbb::tick_count::now();
    // Accumulate matrices and other metada for which output probabilites are then calculated
    //for ( int idx=0; idx<possible_outputs.size(); idx++) {
    tbb::parallel_for( (size_t)0, possible_outputs.size(), (size_t)1, [&](size_t idx ) {

        size_t index_offset = idx*possible_input_states.size();

        // calculate the individual probabilities
        //for ( size_t jdx=0; jdx<possible_input_states.size(); jdx++ ) {
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, possible_input_states.size()), [&](tbb::blocked_range<size_t> r ) {
            for( size_t jdx=r.begin(); jdx!=r.end(); jdx++) {
         
                // accumulate input/output states for permanent calculations
                perm_accumulator.add(&possible_input_states[jdx], &possible_outputs[idx], index_offset+jdx);

            }

        });

    });
tbb::tick_count t1cpu = tbb::tick_count::now();
t_CPU += (t1cpu-t0cpu).seconds();

tbb::tick_count t0 = tbb::tick_count::now();
    // calculate the permanents
    matrix&& permanents = perm_accumulator.calculate(lib);
tbb::tick_count t1 = tbb::tick_count::now();
t_perm_accumulator += (t1-t0).seconds();

t0cpu = tbb::tick_count::now();
    // calculate the individual probabilities associated with the output states
    tbb::parallel_for( (size_t)0, possible_outputs.size(), (size_t)1, [&](size_t idx ) {

        pmf[idx] = 0;

        size_t offset = idx*possible_input_states.size();

        // calculate the individual probabilities

        tbb::combinable<double> pmf_addend{[](){return 0.0;}};

        //for ( size_t jdx=0; jdx<possible_input_states.size(); jdx++ ) {
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, possible_input_states.size()), [&](tbb::blocked_range<size_t> r ) {
            for( size_t jdx=r.begin(); jdx!=r.end(); jdx++) {


                //double probability_tmp = calculate_outputs_probability(interferometer_matrix, possible_input_states[jdx], possible_outputs[idx], lib);
                //probability_tmp = probability_tmp*multinomial_coefficients[jdx]*multinomial_coefficients[jdx];

                double probability = calculate_outputs_probability(permanents[offset+jdx], possible_input_states[jdx], possible_outputs[idx]);
                probability = probability*multinomial_coefficients[jdx]*multinomial_coefficients[jdx];

                // thread local probability
                double& pmf_priv = pmf_addend.local();

                pmf_priv = pmf_priv + probability;
//std::cout << "probabilities: " << probability << " " << probability_tmp << " " << permanents[offset] << std::endl;


            }

        });

        pmf_addend.combine_each([&](double &a) {
             pmf[idx] = pmf[idx] + a;
        });

        probability_sum_total = probability_sum_total + pmf[idx];           


    });
    

    for (size_t idx=0;idx<pmf.size(); idx++) {
        pmf[idx] = pmf[idx]/probability_sum_total;
    }

    // store the calculated probability layer
    PicState_int64 key = sample.copy();
    pmfs[key] = pmf;
t1cpu = tbb::tick_count::now();
t_CPU += (t1cpu-t0cpu).seconds();

}
*/

/**
@brief Call to pick a new sample from the possible output states according to the calculated probability distribution stored in pmfs.
@param sample The current sample represanted by a PicState_int64 class that would be replaced by the new sample.
*/
void
CGeneralizedCliffordsBSimulationStrategy::sample_from_pmf( PicState_int64& sample ) {


    // create a random double
    double rand_num = (double)rand()/RAND_MAX;
   //double rand_num = rand_nums[rand_num_idx];//distribution(generator);
    //rand_num_idx = rand_num_idx + 1;

  

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

    sample[sampled_index]++;
    sample.number_of_photons++;
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

/*
mode_state.print_matrix();
particles_state.print_matrix();
*/
    return particles_state;
}


/**
@brief Call to determine the output probability of associated with the input and output states ---- OBSOLETE, will be deleted
@param interferometer_mtx The matrix of the interferometer.
@param input_state The input state.
@param output_state The output state.
*/
double
calculate_outputs_probability( matrix &interferometer_mtx, PicState_int64 &input_state, PicState_int64 &output_state, int lib ) {


    Complex16 permanent;

    matrix modifiedInterferometerMatrix = adaptInterferometer( interferometer_mtx, input_state, output_state );

    std::function<bool(int64_t)> filterNonZero = [](int64_t elem) { 
        return elem > 0;
    };

    PicState_int64 adapted_input_state = input_state.filter(filterNonZero);
    PicState_int64 adapted_output_state = output_state.filter(filterNonZero);
    
        
    GlynnPermanentCalculatorRepeated permanentCalculatorRecursive;
    permanent = permanentCalculatorRecursive.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state );

    double probability = permanent.real()*permanent.real() + permanent.imag()*permanent.imag();
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



/**
@brief Call to determine the output probability of associated with the input and output states
@param permanent The previously calculated permanent
@param input_state The input state.
@param output_state The output state.
*/
double
calculate_outputs_probability( Complex16& permanent, PicState_int64 &input_state, PicState_int64 &output_state ) {



    double probability = permanent.real()*permanent.real() + permanent.imag()*permanent.imag();
        // squared magnitude norm(a+ib) = a^2 + b^2 !!!

    int64_t photon_num = sum(input_state);
    probability = probability/factorial( photon_num );


    for (size_t idx=0; idx<input_state.size(); idx++) {
        probability = probability/factorial( input_state[idx] );
    }

    return probability;
}


















} // PIC
