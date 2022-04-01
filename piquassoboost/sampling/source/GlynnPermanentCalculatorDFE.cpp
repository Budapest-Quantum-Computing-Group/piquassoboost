#include "GlynnPermanentCalculatorDFE.h"
#include "GlynnPermanentCalculator.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

CALCPERMGLYNNDFE calcPermanentGlynnDFE = NULL;
INITPERMGLYNNDFE initialize_DFE = NULL;
FREEPERMGLYNNDFE releive_DFE = NULL;

#define ROWCOL(m, r, c) ToComplex32(m[ r*m.stride + c])

namespace pic {

inline Complex16 ToComplex16(Complex32 v) {
  return Complex16((double)v.real(), (double)v.imag());
}

inline Complex32 ToComplex32(Complex16 v) {
  return Complex32((long double)v.real(), (long double)v.imag());
}

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculator_DFE(matrix& matrix_mtx, Complex16& perm, int useDual)
{
    if (matrix_mtx.rows < 1+BASEKERNPOW2 || (matrix_mtx.rows < 1+1+BASEKERNPOW2 && useDual)) { //compute with other method
      if (matrix_mtx.rows == 0) perm = Complex16(1.0,0.0);
      else if (matrix_mtx.rows == 1) perm = matrix_mtx[0];
      else if (matrix_mtx.rows == 2) perm = ToComplex16(ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) + ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 0));
      else if (matrix_mtx.rows == 3)
        perm = ToComplex16(ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 2) +
               ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 0) +
               ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 1) +
               ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 0) +
               ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 2) +
               ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 1));
      else if (matrix_mtx.rows == 4)
        perm = ToComplex16(ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 0) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 0) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 1) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 0) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 3) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 3) * ROWCOL(matrix_mtx, 3, 0) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 2) * ROWCOL(matrix_mtx, 1, 3) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 0) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 0) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 2) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 1) * ROWCOL(matrix_mtx, 2, 2) * ROWCOL(matrix_mtx, 3, 0) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 0) * ROWCOL(matrix_mtx, 3, 1) +
                ROWCOL(matrix_mtx, 0, 3) * ROWCOL(matrix_mtx, 1, 2) * ROWCOL(matrix_mtx, 2, 1) * ROWCOL(matrix_mtx, 3, 0));
      else {
        GlynnPermanentCalculator gpc;
        perm = gpc.calculate(matrix_mtx);
      }        
      return;
    }
    // calulate the maximal sum of the columns to normalize the matrix
    matrix_base<Complex32> colSumMax( matrix_mtx.cols, 1);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
    for (size_t idx=0; idx<matrix_mtx.rows; idx++) {
        for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
            Complex32 value1 = colSumMax[jdx] + matrix_mtx[ idx*matrix_mtx.stride + jdx];
            Complex32 value2 = colSumMax[jdx] - matrix_mtx[ idx*matrix_mtx.stride + jdx];
            if ( std::abs( value1 ) < std::abs( value2 ) ) {
                colSumMax[jdx] = value2;
            }
            else {
                colSumMax[jdx] = value1;
            }

        }

    }

    // calculate the renormalization coefficients
    matrix_base<long double> renormalize_data(matrix_mtx.cols, 1);
    for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(colSumMax[jdx]); 
        //printf("%d %.21Lf\n", jdx, renormalize_data[jdx]);
    }

    // renormalize the input matrix and convert to fixed point maximizing precision via long doubles
    // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
    const size_t max_dim = useDual ? MAX_FPGA_DIM : MAX_SINGLE_FPGA_DIM;
    const size_t rows = matrix_mtx.rows;
    const size_t max_fpga_cols = max_dim >> BASEKERNPOW2;
    const size_t numinits = 1 << BASEKERNPOW2;
    const size_t actualinits = (matrix_mtx.cols + 9) / 10;
    matrix_base<ComplexFix16> mtxfix[numinits] = {};
    const long double fixpow = 1L << 62;
    for (size_t i = 0; i < actualinits; i++) {
      mtxfix[i] = matrix_base<ComplexFix16>(rows, max_fpga_cols);
      size_t basecol = max_fpga_cols * i;
      size_t lastcol = matrix_mtx.cols<=basecol ? 0 : std::min(max_fpga_cols, matrix_mtx.cols-basecol);
      for (size_t idx=0; idx < rows; idx++) {
        size_t offset = idx * matrix_mtx.stride + basecol;
        size_t offset_small = idx*mtxfix[i].stride;
        for (size_t jdx = 0; jdx < lastcol; jdx++) {
          mtxfix[i][offset_small+jdx].real = llrint((long double)matrix_mtx[offset+jdx].real() * fixpow / renormalize_data[basecol+jdx]);
          mtxfix[i][offset_small+jdx].imag = llrint((long double)matrix_mtx[offset+jdx].imag() * fixpow / renormalize_data[basecol+jdx]);
          //printf("%d %d %d %llX %llX\n", i, idx, jdx, mtxfix[i][offset_small+jdx].real, mtxfix[i][offset_small+jdx].imag); 
        }
        memset(&mtxfix[i][offset_small+lastcol], 0, sizeof(ComplexFix16)*(max_fpga_cols-lastcol));
      }
      for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxfix[i][jdx].real = fixpow; 
    }

    //note: stride must equal number of columns, or this will not work as the C call expects contiguous data
    ComplexFix16* mtx_fix_data[numinits];
    //assert(mtxfix[i].stride == mtxfix[i].cols);
    //assert(matrix_mtx.rows == matrix_mtx.cols && matrix_mtx.rows <= (dual ? MAX_FPGA_DIM : MAX_SINGLE_FPGA_DIM));
    for (size_t i = 0; i < numinits; i++) mtx_fix_data[i] = mtxfix[i].get_data();
    calcPermanentGlynnDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, &perm);


    return;
}

}
