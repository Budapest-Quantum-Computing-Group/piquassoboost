
#ifndef GlynnPermanentCalculatorDFE_H
#define GlynnPermanentCalculatorDFE_H

#include <mutex>
#include "matrix.h"
#include "matrix_real16.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

namespace pic {

/// @brief Structure type representing 16 byte complex numbers
typedef struct ComplexFix16 {
  /// the real part of a complex number
  __int64_t real;
  /// the imaginary part of a complex number
  __int64_t imag;
} ComplexFix16;


/**
@brief ???????
*/
matrix_real16* get_renormalization_data( std::vector<matrix>* matrices );

/**
@brief ???????
*/
std::vector<matrix_base<ComplexFix16>>* renormalize_matrices( std::vector<matrix>* matrices, matrix_real16* renormalize_data, int useFloat );

/**
@brief ???????
*/
void GlynnPermanentCalculatorBatch_DFE(std::vector<matrix_base<ComplexFix16>>* mtxfix, matrix_real16* renormalize_data, int row_num, int col_num, int perm_num, matrix& perm, int useDual, int useFloat);

/**
@brief ???????
*/
void GlynnPermanentCalculatorBatch_DFE(std::vector<matrix>& matrices, matrix& perm, int useDual, int useFloat);

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void GlynnPermanentCalculator_DFE(matrix& matrix_mtx, Complex16& perm, int useDual, int useFloat);

}

#define DFE_MAIN 0
#define DFE_FLOAT 1
#define DFE_REP 2

void inc_dfe_lib_count();
void dec_dfe_lib_count();
int init_dfe_lib(int choice, int dual);
void lock_lib();
void unlock_lib();
extern "C" size_t dfe_mtx_size;
extern "C" size_t dfe_basekernpow2;

#endif
