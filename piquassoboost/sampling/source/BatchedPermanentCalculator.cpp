#include <iostream>
#include "BatchedPermanentCalculator.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include "GlynnPermanentCalculator.h"
#include "GlynnPermanentCalculatorRepeated.h"
#include "GlynnPermanentCalculatorDFE.h"
#include "CChinHuhPermanentCalculator.h"
#include <math.h>

namespace pic {


static double t_DFE_prep=0.0;
static double t_CPU_time=0.0;
static double t_DFE_time=0.0;
static double t_tot=0.0;


/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
BatchednPermanentCalculator::BatchednPermanentCalculator() {

    interferometer_matrix = matrix(0,0);

    batch_size_max = 100000;



}


/**
@brief Constructor of the class.
@return Returns with the instance of the class.
*/
BatchednPermanentCalculator::BatchednPermanentCalculator(matrix& interferometer_matrix_in) {

    interferometer_matrix = interferometer_matrix_in;

    batch_size_max = 100000;


}


/**
@brief Destructor of the class
*/
BatchednPermanentCalculator::~BatchednPermanentCalculator() {

std::cout << "DFE preparation time: " << t_DFE_prep << std::endl;
std::cout << "DFE time: " << t_DFE_time << std::endl;
std::cout << "CPU time in accumulator: " << t_CPU_time << std::endl;
std::cout << "total time in accumulator: " << t_tot << std::endl;

    // clear the contained inputs and metadata
    clear();
}




/**
@brief Call to calculate the permanent via Glynn formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion)
@param mtx The effective scattering matrix of a boson sampling instance
@param input_state The input state
@param output_state The output state
@param index The index of the new element. (For performance reasons the vontainer must be preallocated with function reserve)
@return Returns with the calculated permanent
*/
void BatchednPermanentCalculator::add( PicState_int64* input_state, PicState_int64* output_state, size_t index ) {


    int sum_input_states = input_state->number_of_photons;
    int sum_output_states = output_state->number_of_photons;
    if ( sum_input_states != sum_output_states) {
        std::string error("BatchednPermanentCalculator::add:  Number of input and output states should be equal");
        throw error;
    }

    if ( index >= input_states.size() || index >= output_states.size() ) {
        std::string error("BatchednPermanentCalculator::add:  Containers must be resized");
        throw error;
    }

    
    input_states[index] = input_state;
    output_states[index] = output_state;
    
}




/**
@brief Call to calculate the permanents of the accumulated matrices. The list of matrices is cleared after the calculation.
@return Returns with the vector of the calculated permanents
*/
matrix BatchednPermanentCalculator::calculate(int lib) {

    if (interferometer_matrix.size() == 0) {
        std::string error("BatchednPermanentCalculator::calculate:  interferometer matrix is zero");
        throw error;
    }

    matrix ret(1, input_states.size());

    if ( lib == GlynnRep ) {

        //GlynnPermanentCalculator permanentCalculator;
        GlynnPermanentCalculatorRepeated permanentCalculator;

        // define function to filter out nonzero elements
        std::function<bool(int64_t)> filterNonZero = [](int64_t elem) { 
            return elem > 0;
        };

 
        // calculate the permanents on CPU
        for ( int idx=0; idx<input_states.size(); idx++) {
/*
            // create the matrix for which permanent would be calculated including row/col multiplicities
            matrix&& modifiedInterferometerMatrix = adaptInterferometerGlynnMultiplied(interferometer_matrix, input_states[idx], output_states[idx] );
            ret[idx] = permanentCalculator.calculate( modifiedInterferometerMatrix  );
*/
            // create the matrix for which permanent would be calculated
            matrix&& modifiedInterferometerMatrix = adaptInterferometer( interferometer_matrix, *(input_states[idx]), *(output_states[idx]) );
            PicState_int64 adapted_input_state = input_states[idx]->filter(filterNonZero);
            PicState_int64 adapted_output_state = output_states[idx]->filter(filterNonZero);    
        
            ret[idx] = permanentCalculator.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state);

        }
    
    } 
    else if (lib == ChinHuh) {

        CChinHuhPermanentCalculator permanentCalculator;

        // calculate the permanents on CPU
        for ( int idx=0; idx<input_states.size(); idx++) {       
            ret[idx] = permanentCalculator.calculate( interferometer_matrix, *input_states[idx], *output_states[idx]);
        }

    }
    else if (lib == GlynnRepSingleDFE || lib == GlynnRepDualDFE) {   
        std::string error("BatchednPermanentCalculator::calculate:  repeated DFE not implemented yet");
        throw error;
    }
    else if (lib == GlynnRepMultiSingleDFE || lib == GlynnRepMultiDualDFE) {   


        // calculate the permanents on DFE
        int useDual = lib == GlynnRepMultiSingleDFE ? 0 : 1;
        int useFloat = 0;

tbb::tick_count t0 = tbb::tick_count::now();
        init_dfe_lib(DFE_MAIN, useDual);
tbb::tick_count t1 = tbb::tick_count::now();
t_DFE_prep += (t1-t0).seconds();
        inc_dfe_lib_count();

        // create the first batch
        int start_index = 0;
tbb::tick_count t0a = tbb::tick_count::now();


        std::vector<matrix>* matrices = create_batch( start_index );
        std::vector<matrix>* matrices_old = NULL;
        // get data for matrix renormalization
        matrix_real16* renormalize_data = get_renormalization_data(matrices);
        matrix_real16* renormalize_data_old = NULL;
        // renormalize the matrices for DFE calculation
        std::vector<matrix_base<ComplexFix16>>* mtxfix = renormalize_matrices( matrices, renormalize_data, useFloat );
        std::vector<matrix_base<ComplexFix16>>* mtxfix_old = NULL;


tbb::tick_count t1a = tbb::tick_count::now();
t_CPU_time += (t1a-t0a).seconds();        


        while ( start_index < (int)input_states.size() ) {
//std::cout << start_index << " " << input_states.size() << std::endl;

            std::vector<matrix>* matrices_new;
            matrix_real16* renormalize_data_new;
            std::vector<matrix_base<ComplexFix16>>* mtxfix_new;

            tbb::parallel_invoke(    
                [&]{
tbb::tick_count t0 = tbb::tick_count::now();
                    // calculate the permanents
                    matrix ret_batched(ret.get_data()+start_index, 1, matrices->size());

                    if (matrices->begin()->rows < 4) { //compute with other method
                        GlynnPermanentCalculator gpc;
                        for (size_t i = 0; i < matrices->size(); i++) {
                            ret_batched[i] = gpc.calculate((*matrices)[i]);
                        }
                        return;
                    }
  

                    // calculate the permanent on DFE
                    GlynnPermanentCalculatorBatch_DFE(mtxfix, renormalize_data, matrices->begin()->rows, matrices->begin()->cols, matrices->size(), ret_batched, useDual, useFloat);

                    //GlynnPermanentCalculatorBatch_DFE(*matrices, ret_batched, useDual, useFloat);
tbb::tick_count t1 = tbb::tick_count::now();
t_DFE_time += (t1-t0).seconds();
                },
                [&]{
//tbb::tick_count t0 = tbb::tick_count::now();

                    // in parallel create the new batch
                    matrices_new = create_batch( start_index+matrices->size() );    

                    // get data for matrix renormalization
                    renormalize_data_new = get_renormalization_data(matrices_new);

                    // renormalize the matrices for DFE calculation
                    mtxfix_new = renormalize_matrices( matrices_new, renormalize_data_new, useFloat );
//tbb::tick_count t1 = tbb::tick_count::now();
//t_CPU_time += (t1-t0).seconds();
        
                },
                [&]{
                    if (matrices_old) {
                        delete(matrices_old);
                        matrices_old = NULL;
                    } 
                },
                [&]{
                    if (mtxfix_old) {
                        delete(mtxfix_old);
                        mtxfix_old = NULL;
                    } 
                },
                [&]{
                    if (renormalize_data_old) {
                        delete(renormalize_data_old);
                        renormalize_data_old = NULL;
                    } 
                }
            );      


            start_index += (int)matrices->size();
tbb::tick_count t0cpu = tbb::tick_count::now();

            mtxfix_old = mtxfix;
            matrices_old = matrices;
            renormalize_data_old = renormalize_data;
            mtxfix = mtxfix_new;
            matrices = matrices_new;
            renormalize_data = renormalize_data_new;

tbb::tick_count t1cpu = tbb::tick_count::now();
t_CPU_time += (t1cpu-t0cpu).seconds();

        }

        if ( mtxfix->size() > 0 ) {
            // calculate the last batch if any left
            matrix ret_batched(ret.get_data()+start_index, 1, matrices->size());
            GlynnPermanentCalculatorBatch_DFE(mtxfix, renormalize_data, matrices->begin()->rows, matrices->begin()->cols, matrices->size(), ret_batched, useDual, useFloat);
            //GlynnPermanentCalculatorBatch_DFE(*matrices, ret_batched, useDual, useFloat);
        }

        if (matrices_old) {
            delete(matrices_old);
            matrices_old = NULL;
        } 

        if (mtxfix_old) {
            delete(mtxfix_old);
            mtxfix_old = NULL;
        } 

        if (renormalize_data_old) {
            delete(renormalize_data_old);
            renormalize_data_old = NULL;
        } 

        delete(mtxfix);
        mtxfix = NULL;
        delete(matrices);
        matrices = NULL;
        delete(renormalize_data);
        renormalize_data = NULL;

        
dec_dfe_lib_count();
tbb::tick_count t2 = tbb::tick_count::now();
t_tot += (t2-t0).seconds();
    }



/*
for ( int idx=0; idx<ret.size(); idx++ ) {

    Complex16 diff = ret[idx] - perms[idx];
    if ( diff.real()*diff.real() + diff.imag()*diff.imag() > 1e-8 ) {
        std::cout << ret[idx] << " " << perms[idx] << std::endl;
    }
 
}
*/

    return ret;
}




/**
@brief Call toclear the list of matrices and other metadata.
*/
void BatchednPermanentCalculator::clear() {

    input_states.clear();
    output_states.clear();


}


/**
@brief reservememory space to store data
@param num_of_permanents The number of permanents to be calculated
*/
void BatchednPermanentCalculator::reserve_space( size_t num_of_permanents) {


    input_states.resize(num_of_permanents);
    output_states.resize(num_of_permanents);

}




/**
@brief ????????
@param num_of_permanents The number of permanents to be calculated
*/
std::vector<matrix>* BatchednPermanentCalculator::create_batch( int start_index) {

    
    std::vector<matrix>* matrices = new std::vector<matrix>;

    // determine the size of the batch
    int batch_size = (int)input_states.size() - start_index < batch_size_max ? input_states.size() - start_index : batch_size_max;

    matrices->resize( batch_size);

    // create matrices for DFE
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, batch_size), [&](tbb::blocked_range<size_t> r ) {
        for( size_t idx=r.begin(); idx!=r.end(); idx++) {

            // create the matrix for which permanent would be calculated including row/col multiplicities
            matrix&& modifiedInterferometerMatrix = adaptInterferometerGlynnMultiplied(interferometer_matrix, input_states[idx+start_index], output_states[idx+start_index] );
            (*matrices)[idx] = modifiedInterferometerMatrix;
        }

    });

    return matrices;

}




/** @brief Creates a matrix from the `interferometerMatrix` corresponding to the parameters `input_state` and `output_state`.
 *         Corresponding rows and columns are multipled based on output and input states.
 *  @param interferometerMatrix Unitary matrix describing a quantum circuit
 *  @param input_state_in The input state
 *  @param output_state_in The output state
 *  @return Returns with the created matrix
 */
matrix
adaptInterferometerGlynnMultiplied( const matrix& interferometerMatrix, PicState_int64* input_state,  PicState_int64* output_state) {

    int n = interferometerMatrix.rows;

    int64_t sum = input_state->number_of_photons;
    matrix mtx(sum, sum);

    int row_idx = 0;
    int row_offset_orig = 0;
    int row_offset = 0;
    for (int i = 0; i < n; i++){

        for (int db_row = 0; db_row < (*output_state)[i]; db_row++){

            int col_idx = 0;

            for (int j = 0; j < n; j++){
                for (int db_col = 0; db_col < (*input_state)[j]; db_col++){
                    mtx[row_offset + col_idx] =  interferometerMatrix[row_offset_orig + j];
                    col_idx++;
                }
            }

            row_offset += mtx.stride;
        }

        row_offset_orig += interferometerMatrix.stride;

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
adaptInterferometer( const matrix& interferometerMatrix, PicState_int64 &input_state, PicState_int64 &output_state) {

    int nonZeroInput = 0;
    for (int i = 0; i < input_state.size(); i++){
        if (input_state[i] > 0){
            nonZeroInput++;
        }
    }
    int nonZeroOutput = 0;
    for (int i = 0; i < output_state.size(); i++){
        if (output_state[i] > 0){
            nonZeroOutput++;
        }
    }

    matrix new_mtx(nonZeroOutput, nonZeroInput);    

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
