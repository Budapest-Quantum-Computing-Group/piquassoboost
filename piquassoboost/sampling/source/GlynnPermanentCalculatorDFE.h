
#ifndef GlynnPermanentCalculatorDFE_H
#define GlynnPermanentCalculatorDFE_H

#include "matrix.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

// the maximal dimension of matrix to be ported to FPGA for permanent calculation
#define MAX_FPGA_DIM 8*5
#define MAX_SINGLE_FPGA_DIM 4*10
#define BASEKERNPOW2 2

namespace pic {

/// @brief Structure type representing 16 byte complex numbers
typedef struct ComplexFix16 {
  /// the real part of a complex number
  __int64_t real;
  /// the imaginary part of a complex number
  __int64_t imag;
} ComplexFix16;

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void GlynnPermanentCalculator_DFE(matrix& matrix_mtx, Complex16& perm, int useDual);

}

typedef void(*CALCPERMGLYNNDFE)(const pic::ComplexFix16**, const long double*, const uint64_t, const uint64_t, pic::Complex16*);
typedef void(*INITPERMGLYNNDFE)(int);
typedef void(*FREEPERMGLYNNDFE)(void);
extern "C" CALCPERMGLYNNDFE calcPermanentGlynnDFE; 
extern "C" INITPERMGLYNNDFE initialize_DFE; 
extern "C" FREEPERMGLYNNDFE releive_DFE; 


#endif
