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


#include "samplingHelperFunctions.h"

#include <iostream>
#include "CGeneralizedCliffordsBSimulationStrategy.h"
#include "CChinHuhPermanentCalculator.h"
#include "GlynnPermanentCalculatorRepeated.h"
#include "BBFGPermanentCalculatorRepeated.h"
#include "BatchedPermanentCalculator.h"
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


namespace pic{

    


matrix_real
compute_pmf( matrix &interferometer_matrix, PicState_int64& sample, PicState_int64 &current_input ) {

    // reset the previously calculated pmf layer
    matrix_real pmf = matrix_real(1, sample.size());

    // define function to filtecompute_pmfr out nonzero elements
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

    if ( nonzero_output_elements < nonzero_output_elements_threshold || current_input.number_of_photons > 40 ) {
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

            //tbb::tick_count t0 = tbb::tick_count::now();//////////////////////////
            //matrix&& modifiedInterferometerMatrix = adaptInterferometerGlynnMultiplied(interferometer_matrix, &input_state_loc, &sample );
            //permanent_addends[idx] = permanentCalculator.calculate( modifiedInterferometerMatrix  );
               
            
            matrix&& modifiedInterferometerMatrix = adaptInterferometer( interferometer_matrix, input_state_loc, sample );
            PicState_int64 adapted_input_state = input_state_loc.filter(filterNonZero);
            PicState_int64 adapted_output_state = sample.filter(filterNonZero);
            permanent_addends[colIndices[idx]] = BBFGpermanentCalculator.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state, false);

            //tbb::tick_count t1 = tbb::tick_count::now();////////////////////////// 
            //t_CPU_permanent += (t1-t0).seconds();    //////////////////////////             

/*
            tbb::tick_count t0b = tbb::tick_count::now();////////////////////////// 
            Complex16 perm = permanentCalculator.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state);
            tbb::tick_count t1b = tbb::tick_count::now();////////////////////////// 
            t_CPU_permanent_Glynn += (t1b-t0b).seconds();    //////////////////////////             
*/
/*
            if ( std::norm( permanent_addends[colIndices[idx]] - perm )/std::norm( permanent_addends[colIndices[idx]]) > 1e-3 ) {
                std::cout << "difference in idx=" << idx << " " << permanent_addends[colIndices[idx]] << " " << perm << std::endl;
            }  
*/


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
                    reduced_input_state[idx] = current_input[colIndices[idx]];
                }
                reduced_input_state.number_of_photons = current_input.number_of_photons;


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
                    //t_DFE_prepare += (t1c-t0c).seconds(); //////////////////////////   
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
                //t_DFE += (t1b-t0b).seconds(); ////////////////////////// 

        });

    }

#endif
     

//std::cout << "iteration done" << std::endl;

   


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



/**
@brief Call to pick a new sample from the possible output states according to the calculated probability distribution stored in pmfs.
@param sample The current sample represanted by a PicState_int64 class that would be replaced by the new sample.
*/
void
sample_from_pmf( PicState_int64& sample, matrix_real &pmf ) {


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


void 
update_input_by_single_photon(
    PicState_int64& current_input,
    PicState_int64& working_input_state
) {

    if ( working_input_state.size() == 0 ) {
        std::string error("Generalized Cliffords B Simulation Strategy::update_input_by_single_photon: the size of working_input_state is zero");
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


} // namespace pic
