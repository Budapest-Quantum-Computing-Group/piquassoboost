
#ifndef GlynnPermanentCalculatorRepeatedDFE_H
#define GlynnPermanentCalculatorRepeatedDFE_H

#include "matrix.h"
#include "GlynnPermanentCalculatorDFE.h"
#include "PicState.h"
#include "PicVector.hpp"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

namespace pic {

/**
@brief Class representing a generalized Cliffords simulation strategy
*/
class cGlynnPermanentCalculatorRepeatedMulti_DFE {

public: 

    ///
    uint64_t totalPerms;

    
    /// set 1 to use floating point number representation in DFE (not supported yet) or 0 to use fixed points
    int useFloat;  
    ///
    int useDual;
    ///
    bool doCPU;
    ///
    const size_t numinits=4;
    ///
    matrix_base<ComplexFix16>* mtxfix_batched;
    ///
    matrix_base<long double> renormalize_data;
    ///
    matrix_base<long double> renormalize_data_batched;
    ///
    PicVector<PicState_int64> input_states;
    ///
    uint8_t onerows;
    /// The number of photons
    size_t photons;
    ///
    std::vector<uint8_t> mrows;
    ///
    std::vector<uint8_t> row_indices;
    ///
    PicState_int64 output_state_loc;
    ///
    std::vector<unsigned char> colIndices; 
    ///
    PicVector<std::vector<unsigned char>> colIndices_batched;
    ///
    const size_t max_dim = dfe_mtx_size;
    ///
    std::vector<uint64_t> curmp;
    ///
    std::vector<uint64_t> inp;
    ///
    PicVector<size_t> batch_iterations;
    ///
    PicVector<uint64_t> mplicity;
    ///
    uint8_t mulsum;
    ///
    matrix matrix_init;
    ///
    PicState_int64 input_state;
    ///
    PicState_int64 output_state;


public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
cGlynnPermanentCalculatorRepeatedMulti_DFE();

/**
@brief Constructor of the class.
@param ???????????????????
@return Returns with the instance of the class.
*/
cGlynnPermanentCalculatorRepeatedMulti_DFE( matrix& matrix_mtx_in, PicState_int64& input_state_in, PicState_int64& output_state_in, int useDual  );

/**
@brief Destructor of the class
*/
~cGlynnPermanentCalculatorRepeatedMulti_DFE();


/**
@brief Call to convert multiplicities of columns to indices
*/
void determineColIndices( PicState_int64& input_state );


/**
@brief Call to convert multiplicities of columns to indices
*/
void determineColIndices();


/**
@brief ????????????
*/
int reserveSpace( size_t jdx );


/**
@brief ???????
*/
void addInputState(size_t idx, PicState_int64& input_state_in);

/**
@brief ???????
*/
void initiateBatch(size_t batch_idx);


/**
@brief ???????
*/
void determineMultiplicities();


/**
@brief ???????
*/
void determineNormalization();

/**
@brief ???????
*/
void prepareDataForRepeatedMulti_DFE(size_t insideBatch_idx, size_t batch_idx_offset);



/**
@brief ???????
*/
matrix calculate(size_t batch_idx, size_t batch_index_offset);





/**
@brief ???????
*/
int determineBatchIterations();


}; // GlynnPermanentCalculatorRepeatedMulti_DFE



/**
@brief ??????
*/
void
GlynnPermanentCalculatorRepeatedMulti_DFE(matrix& matrix_init, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual);

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void GlynnPermanentCalculatorRepeated_DFE(matrix& matrix_mtx, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual, int useFloat);

void
GlynnPermanentCalculatorRepeatedInputBatch_DFE(matrix& matrix_init, std::vector<std::vector<PicState_int64>>& input_states,
    std::vector<PicState_int64>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual, int useFloat);

void
GlynnPermanentCalculatorRepeatedOutputBatch_DFE(matrix& matrix_init, std::vector<PicState_int64>& input_states,
    std::vector<std::vector<PicState_int64>>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual, int useFloat);

}


#endif
