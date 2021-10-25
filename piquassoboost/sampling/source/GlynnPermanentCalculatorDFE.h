
#ifndef GlynnPermanentCalculatorDFE_H
#define GlynnPermanentCalculatorDFE_H

#include "matrix.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

// the maximal dimension of matrix to be ported to FPGA for permanent calculation
#define MAX_FPGA_DIM 40
#define MAX_SINGLE_FPGA_DIM 28

extern "C"
{
    void calcPermanentGlynn_DualDFE(const pic::Complex16* mtx_data[8], const double* renormalize_data, const uint64_t rows, const uint64_t cols, pic::Complex16* perm);
    void calcPermanentGlynn_SingleDFE(const pic::Complex16* mtx_data[4], const double* renormalize_data, const uint64_t rows, const uint64_t cols, pic::Complex16* perm);
    void initialize_DFE();
    void releive_DFE();
}



namespace pic {

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void GlynnPermanentCalculator_DFEDualCard(matrix& matrix_mtx, Complex16& perm);


/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void GlynnPermanentCalculator_DFESingleCard(matrix& matrix_mtx, Complex16& perm);


}



#endif
