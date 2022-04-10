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


/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
BatchednPermanentCalculator::BatchednPermanentCalculator() {

    interferometer_matrix = matrix(0,0);

    // reserve space for containers
    input_states.reserve(30);
    output_states.reserve(30);



}


/**
@brief Constructor of the class.
@return Returns with the instance of the class.
*/
BatchednPermanentCalculator::BatchednPermanentCalculator(matrix& interferometer_matrix_in) {

    interferometer_matrix = interferometer_matrix_in;


    // reserve space for containers
    input_states.reserve(30);
    output_states.reserve(30);



}


/**
@brief Destructor of the class
*/
BatchednPermanentCalculator::~BatchednPermanentCalculator() {

    // clear the contained inputs and metadata
    clear();
}




/**
@brief Call to calculate the permanent via Glynn formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion)
@param mtx The effective scattering matrix of a boson sampling instance
@param input_state The input state
@param output_state The output state
@return Returns with the calculated permanent
*/
void BatchednPermanentCalculator::add( PicState_int64& input_state, PicState_int64& output_state ) {


    int sum_input_states = sum(input_state);
    int sum_output_states = sum(output_state);
    if ( sum_input_states != sum_output_states) {
        std::string error("BatchednPermanentCalculator::add:  Number of input and output states should be equal");
        throw error;
    }

    if (sum_input_states == 0 || sum_output_states == 0)
        return;


    input_states.push_back(input_state);
    output_states.push_back(output_state);


    
}




/**
@brief Call to calculate the permanents of the accumulated matrices. The list of matrices is cleared after the calculation.
@return Returns with the vector of the calculated permanents
*/
std::vector<Complex16> BatchednPermanentCalculator::calculate(int lib) {

    if (interferometer_matrix.size() == 0) {
        std::string error("BatchednPermanentCalculator::calculate:  interferometer matrix is zero");
        throw error;
    }

    std::vector<Complex16> ret;
    ret.resize( input_states.size());

    if ( lib == GlynnRep ) {

        //GlynnPermanentCalculator permanentCalculator;
        GlynnPermanentCalculatorRepeated permanentCalculator;

        // define function to filter out nonzero elements
        std::function<bool(int64_t)> filterNonZero = [](int64_t elem) { 
            return elem > 0;
        };

 
        // calculate the permanents on CPU
        for ( int idx=0; idx<input_states.size(); idx++) {

            // create the matrix for which permanent would be calculated including row/col multiplicities
            //matrix&& modifiedInterferometerMatrix = adaptInterferometerGlynnMultiplied(matrices[idx], input_states[idx], output_states[idx] );
            //ret[idx] = permanentCalculator.calculate( modifiedInterferometerMatrix  );

            // create the matrix for which permanent would be calculated
            matrix&& modifiedInterferometerMatrix = adaptInterferometer( interferometer_matrix, input_states[idx], output_states[idx] );
            PicState_int64 adapted_input_state = input_states[idx].filter(filterNonZero);
            PicState_int64 adapted_output_state = output_states[idx].filter(filterNonZero);    
        
            ret[idx] = permanentCalculator.calculate( modifiedInterferometerMatrix, adapted_input_state, adapted_output_state);
        }
    
    } 
    else if (lib == ChinHuh) {

        CChinHuhPermanentCalculator permanentCalculator;

        // calculate the permanents on CPU
        for ( int idx=0; idx<input_states.size(); idx++) {       
            ret[idx] = permanentCalculator.calculate( interferometer_matrix, input_states[idx], output_states[idx]);
        }

    }
    else if (lib == GlynnRepSingleDFE || lib == GlynnRepDualDFE) {   
        std::string error("BatchednPermanentCalculator::calculate:  repeated DFE not implemented yet");
        throw error;
    }
    else if (lib == GlynnRepMultiSingleDFE || lib == GlynnRepMultiDualDFE) {   


        // calculate the permanents on DFE
        int useDual = 1;
        int useFloat = 0;

        std::vector<matrix> matrices;
        matrices.reserve( input_states.size());

        // calculate the permanents on CPU
        for ( int idx=0; idx<input_states.size(); idx++) {

            // create the matrix for which permanent would be calculated including row/col multiplicities
            matrix&& modifiedInterferometerMatrix = adaptInterferometerGlynnMultiplied(interferometer_matrix, input_states[idx], output_states[idx] );

            matrices.push_back( modifiedInterferometerMatrix );
        }


        GlynnPermanentCalculatorBatch_DFE(matrices, ret, useDual, useFloat);

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
