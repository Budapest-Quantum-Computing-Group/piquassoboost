
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
@brief ??????
*/
void
prepareDataForRepeatedMulti_DFE(matrix& matrix_init, PicState_int64& input_state, PicState_int64& output_state, int useFloat,
matrix_base<ComplexFix16>* mtxfix, matrix_base<long double>& renormalize_data_all, std::vector<uint64_t>& mplicity, uint8_t& onerows, size_t& photons, uint64_t& totalPerms, uint8_t& mulsum, bool& doCPU);

/**
@brief ??????
*/
void
GlynnPermanentCalculatorRepeatedMulti_DFE(matrix& matrix_init, PicState_int64& input_state, PicState_int64& output_state, const int useFloat,
const matrix_base<ComplexFix16>* mtxfix, const matrix_base<long double>& renormalize_data_all, const std::vector<uint64_t>& mplicity, const uint8_t& onerows, const size_t& photons, const uint64_t& totalPerms, const uint8_t& mulsum, const bool& doCPU, const int& useDual, Complex16& perm);

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
