
#ifndef GlynnPermanentCalculatorRepeatedDFE_H
#define GlynnPermanentCalculatorRepeatedDFE_H

#include "matrix.h"
#include "GlynnPermanentCalculatorDFE.h"
#include "PicState.h"

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
protected:

    
    /// set 1 to use floating point number representation in DFE (not supported yet) or 0 to use fixed points
    int useFloat;  
    ///
    int useDual;
    ///
    bool doCPU;
    ///
    const size_t numinits=4;
    ///
    matrix_base<ComplexFix16>* mtxfix;
    ///
    matrix_base<long double> renormalize_data_all;
    ///
    uint8_t onerows;
    /// The number of photons
    size_t photons;

    ///
    std::vector<uint64_t> mplicity;
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
@brief ???????
*/
void prepareDataForRepeatedMulti_DFE();

/**
@brief ???????
*/
Complex16 calculate();


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
    PicState_int64& output_state, Complex16& perm, int useDual);

}


#endif
