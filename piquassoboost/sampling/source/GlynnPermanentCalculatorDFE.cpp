#include "GlynnPermanentCalculatorDFE.h"
#include "GlynnPermanentCalculator.hpp"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

#include <atomic>
#include <dlfcn.h>
#include <unistd.h>

//DFE management library interface follows:
#define DFE_LIB_SIM "libPermanentGlynnSIM.so"
#define DFE_LIB_SIMDUAL "libPermanentGlynnDualSIM.so"
#define DFE_LIB "libPermanentGlynnDFE.so"
#define DFE_LIBDUAL "libPermanentGlynnDualDFE.so"
#define DFE_LIB_SIMF "libPermanentGlynnSIMF.so"
#define DFE_LIB_SIMFDUAL "libPermanentGlynnDualSIMF.so"
#define DFE_LIBF "libPermanentGlynnDFEF.so"
#define DFE_LIBFDUAL "libPermanentGlynnDualDFEF.so"
#define DFE_REP_LIB_SIM "libPermRepGlynnSIM.so"
#define DFE_REP_LIB_SIMDUAL "libPermRepGlynnDualSIM.so"
#define DFE_REP_LIB "libPermRepGlynnDFE.so"
#define DFE_REP_LIBDUAL "libPermRepGlynnDualDFE.so"


typedef void(*CALCPERMGLYNNDFE)(const pic::ComplexFix16**, const long double*, const uint64_t, const uint64_t, const uint64_t, pic::Complex16*);
typedef int(*INITPERMGLYNNDFE)(int, size_t*, size_t*);
typedef void(*FREEPERMGLYNNDFE)(void);

CALCPERMGLYNNDFE calcPermanentGlynnDFE = NULL;
INITPERMGLYNNDFE initialize_DFE = NULL;
FREEPERMGLYNNDFE releive_DFE = NULL;
CALCPERMGLYNNDFE calcPermanentGlynnDFEF = NULL;
INITPERMGLYNNDFE initialize_DFEF = NULL;
FREEPERMGLYNNDFE releive_DFEF = NULL;

typedef void(*CALCPERMGLYNNREPDFE)(const pic::ComplexFix16**, const long double*, const uint64_t, const uint64_t, const unsigned char*,
  const uint8_t*, const uint8_t, const uint8_t, const uint64_t*, const uint64_t, const uint8_t, pic::Complex16*);
typedef int(*INITPERMGLYNNREPDFE)(int, size_t*, size_t*);
typedef void(*FREEPERMGLYNNREPDFE)(void);

extern "C" CALCPERMGLYNNREPDFE calcPermanentGlynnRepDFE;
extern "C" INITPERMGLYNNREPDFE initializeRep_DFE;
extern "C" FREEPERMGLYNNREPDFE releiveRep_DFE;

size_t dfe_mtx_size;
size_t dfe_basekernpow2;

void* handle = NULL;
int isLastDual = 0;
std::atomic_size_t refcount(0);
std::atomic_size_t read_count(0); //readers-writer problem semaphore
std::recursive_mutex libmutex; //writing mutex
std::mutex libreadmutex; //reader mutex

/**
@brief ???????
*/
void unload_dfe_lib()
{
    const std::lock_guard<std::recursive_mutex> lock(libmutex);
    if (handle) {
        if (releive_DFE) {
            releive_DFE();
            initialize_DFE = NULL, releive_DFE = NULL, calcPermanentGlynnDFE = NULL;
        }
        if (releive_DFEF) {
            releive_DFEF();
            initialize_DFEF = NULL, releive_DFEF = NULL, calcPermanentGlynnDFEF = NULL;
        }
        if (releiveRep_DFE) {
            releiveRep_DFE();
            initializeRep_DFE = NULL, releiveRep_DFE = NULL, calcPermanentGlynnRepDFE = NULL;
        }
        dlclose(handle);
        handle = NULL;
    }
}

/**
@brief ???????
*/
int init_dfe_lib(int choice, int dual) {
    const std::lock_guard<std::recursive_mutex> lock(libmutex);
    if (choice == DFE_MAIN && initialize_DFE && dual == isLastDual) return initialize_DFE(0, &dfe_mtx_size, &dfe_basekernpow2);
    if (choice == DFE_FLOAT && initialize_DFEF && dual == isLastDual) return initialize_DFEF(0, &dfe_mtx_size, &dfe_basekernpow2);
    if (choice == DFE_REP && initializeRep_DFE && dual == isLastDual) return initializeRep_DFE(0, &dfe_mtx_size, &dfe_basekernpow2);
    isLastDual = dual;
    unload_dfe_lib();
    const char* simLib = NULL, *lib = NULL;
    if (choice == DFE_MAIN) {
        simLib = dual ? DFE_LIB_SIMDUAL : DFE_LIB_SIM;
        lib = dual ? DFE_LIBDUAL : DFE_LIB;
    } else if (choice == DFE_FLOAT) {
        simLib = dual ? DFE_LIB_SIMFDUAL : DFE_LIB_SIMF;
        lib = dual ? DFE_LIBFDUAL : DFE_LIBF;
    } else if (choice == DFE_REP) {
        simLib = dual ? DFE_REP_LIB_SIMDUAL : DFE_REP_LIB_SIM;
        lib = dual ? DFE_REP_LIBDUAL : DFE_REP_LIB;
    }
    // dynamic-loading the correct DFE permanent calculator (Simulator/DFE/single or dual) from shared libararies
    handle = dlopen(getenv("USE_DFE_SIMULATOR") ? simLib : lib, RTLD_NOW); //"MAXELEROSDIR"
    if (handle == NULL) {
        char* pwd = getcwd(NULL, 0);
        fprintf(stderr, "%s\n'%s' (in %s mode) failed to load from working directory '%s' use export LD_LIBRARY_PATH\n", dlerror(), getenv("USE_DFE_SIMULATOR") ? simLib : lib, getenv("USE_DFE_SIMULATOR") ? "simulator" : "DFE", pwd);
        free(pwd);
    } else {
      // in case the DFE libraries were loaded successfully the function pointers are set to initialize/releive DFE engine and run DFE calculations
      if (choice == DFE_MAIN) {
          calcPermanentGlynnDFE = (CALCPERMGLYNNDFE)dlsym(handle, "calcPermanentGlynnDFE");
          initialize_DFE = (INITPERMGLYNNDFE)dlsym(handle, "initialize_DFE");
          releive_DFE = (FREEPERMGLYNNDFE)dlsym(handle, "releive_DFE");
          if (initialize_DFE) return initialize_DFE(0, &dfe_mtx_size, &dfe_basekernpow2);
      } else if (choice == DFE_FLOAT) {
          calcPermanentGlynnDFEF = (CALCPERMGLYNNDFE)dlsym(handle, "calcPermanentGlynnDFEF");
          initialize_DFEF = (INITPERMGLYNNDFE)dlsym(handle, "initialize_DFEF");
          releive_DFEF = (FREEPERMGLYNNDFE)dlsym(handle, "releive_DFEF");
          if (initialize_DFEF) return initialize_DFEF(0, &dfe_mtx_size, &dfe_basekernpow2);
      } else if (choice == DFE_REP) {
          calcPermanentGlynnRepDFE = (CALCPERMGLYNNREPDFE)dlsym(handle, "calcPermanentGlynnRepDFE");
          initializeRep_DFE = (INITPERMGLYNNREPDFE)dlsym(handle, "initializeRep_DFE");
          releiveRep_DFE = (FREEPERMGLYNNREPDFE)dlsym(handle, "releiveRep_DFE");
          if (initializeRep_DFE) return initializeRep_DFE(0, &dfe_mtx_size, &dfe_basekernpow2);
      }
    }
}

/**
@brief ???????
*/
void inc_dfe_lib_count() { refcount++; }

/**
@brief ???????
*/
void dec_dfe_lib_count()
{
    if (--refcount == 0) unload_dfe_lib();
}

/**
@brief ???????
*/
void lock_lib()
{
    const std::lock_guard<std::mutex> lock(libreadmutex);
    if (++read_count == 1) libmutex.lock();
}

/**
@brief ???????
*/
void unlock_lib()
{
    const std::lock_guard<std::mutex> lock(libreadmutex);
    if (--read_count == 0) libmutex.unlock();
}

#define ROWCOL(m, r, c) ToComplex32(m[ r*m.stride + c])

namespace pic {

/**
@brief ???????
*/
inline Complex16 ToComplex16(Complex32 v) {
  return Complex16((double)v.real(), (double)v.imag());
}

/**
@brief ???????
*/
inline Complex32 ToComplex32(Complex16 v) {
  return Complex32((long double)v.real(), (long double)v.imag());
}

/**
@brief ???????
*/
inline long long doubleToLLRaw(double d)
{
    double* pd = &d;
    long long* pll = (long long*)pd;
    return *pll;
}


/**
@brief ???????
*/
matrix_real16* get_renormalization_data( std::vector<matrix>* matrices ) {

    if (matrices->size() == 0 ) {
        return new matrix_real16(0,0);
    }

    matrix_real16* renormalize_data = new matrix_real16(matrices->size(), matrices->begin()->cols);

    //for (size_t i = 0; i < matrices.size(); i++) {
    tbb::parallel_for( tbb::blocked_range<size_t>(0, matrices->size()), [&](tbb::blocked_range<size_t> r) {
        for (size_t i=r.begin(); i<r.end(); ++i) {
            matrix& matrix_mtx = (*matrices)[i];

            // calulate the maximal sum of the columns to normalize the matrix
            matrix_base<Complex32> colSumMax( matrix_mtx.cols, 4);
            memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );

            //sum up vectors in first/upper-left and fourth/lower-right quadrants
            for (size_t idx=0; idx<matrix_mtx.rows; idx++) {
                for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
                    size_t offset = idx*matrix_mtx.stride + jdx;
                    int realPos = matrix_mtx[offset].real() > 0;
                    int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
                    if (realPos) colSumMax[2*jdx+slopeUpLeft] += matrix_mtx[offset];
                    else colSumMax[2*jdx+slopeUpLeft] -= matrix_mtx[offset];
                }
            
            }
            
            //now try to add/subtract neighbor quadrant values to the prior sum vector to see if it increase the absolute value 
            for (size_t idx=0; idx<matrix_mtx.rows; idx++) {
                for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
                    size_t offset = idx*matrix_mtx.stride + jdx;
                    int realPos = matrix_mtx[offset].real() > 0;
                    int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
                    Complex32 value1 = colSumMax[2*jdx+1-slopeUpLeft] + matrix_mtx[offset];
                    Complex32 value2 = colSumMax[2*jdx+1-slopeUpLeft] - matrix_mtx[offset];
                    colSumMax[2*jdx+1-slopeUpLeft] = std::norm(value1) > std::norm(value2) ? value1 : value2;
                } 
            }   
            
            // calculate the renormalization coefficients
            for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
                (*renormalize_data)[i*renormalize_data->stride+jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
                //printf("%d %.21Lf\n", jdx, renormalize_data[jdx]);
            }
            
        }
        
    });

    return renormalize_data;

}


/**
@brief ???????
*/
std::vector<matrix_base<ComplexFix16>>* renormalize_matrices( std::vector<matrix>* matrices, matrix_real16* renormalize_data, int useFloat ) {

    if ( matrices->size() == 0 ) {
        //std::vector<matrix_base<ComplexFix16>> mtxfix;
        return new std::vector<matrix_base<ComplexFix16>>;
    }

    // renormalize the input matrix and convert to fixed point maximizing precision via long doubles
    // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
    const size_t max_dim = dfe_mtx_size;
    const size_t rows = matrices->begin()->rows;
    const size_t numinits = 4;
    const size_t max_fpga_cols = max_dim / numinits;
    const size_t actualinits = (matrices->begin()->cols + max_fpga_cols-1) / max_fpga_cols;

    std::vector<matrix_base<ComplexFix16>>* mtxfix = new std::vector<matrix_base<ComplexFix16>>;//[numinits] = {};
    mtxfix->resize(numinits);

    const long double fixpow = 1ULL << 62;
    const double fOne = doubleToLLRaw(1.0);

    for (size_t i = 0; i < actualinits; i++) {
        (*mtxfix)[i] = matrix_base<ComplexFix16>(rows * matrices->size(), max_fpga_cols);
    }

    tbb::parallel_for( tbb::blocked_range<size_t>(0, matrices->size()), [&](tbb::blocked_range<size_t> r) {
        for (size_t i = 0; i < actualinits; i++) {
          size_t basecol = max_fpga_cols * i;
          size_t lastcol = matrices->begin()->cols<=basecol ? 0 : std::min(max_fpga_cols, matrices->begin()->cols-basecol);
          //for (size_t j = 0; j < matrices.size(); j++) {
          for (size_t j = r.begin(); j < r.end(); j++) {
              size_t rowbase = j * rows;
              matrix& matrix_mtx = (*matrices)[j];

              for (size_t idx=0; idx < rows; idx++) {
                  size_t offset = idx * matrix_mtx.stride + basecol;
                  size_t offset_small = (rowbase + idx)*(*mtxfix)[i].stride;

                  for (size_t jdx = 0; jdx < lastcol; jdx++) {
                      (*mtxfix)[i][offset_small+jdx].real = useFloat ? doubleToLLRaw(matrix_mtx[offset+jdx].real()) : llrint((long double)matrix_mtx[offset+jdx].real() * fixpow / (*renormalize_data)[j*renormalize_data->stride+basecol+jdx]);
                      (*mtxfix)[i][offset_small+jdx].imag = useFloat ? doubleToLLRaw(matrix_mtx[offset+jdx].imag()) : llrint((long double)matrix_mtx[offset+jdx].imag() * fixpow / (*renormalize_data)[j*renormalize_data->stride+basecol+jdx]);
                  //printf("%d %d %d %llX %llX\n", i, idx, jdx, mtxfix[i][offset_small+jdx].real, mtxfix[i][offset_small+jdx].imag); 
                  }

                  memset(&((*mtxfix)[i][offset_small+lastcol]), 0, sizeof(ComplexFix16)*(max_fpga_cols-lastcol));
              }

              for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) (*mtxfix)[i][rowbase*(*mtxfix)[i].stride+jdx].real = useFloat ? fOne : fixpow;
          } 
        }
    });

    return mtxfix;

}


/**
@brief ???????
*/
void
GlynnPermanentCalculatorBatch_DFE(std::vector<matrix_base<ComplexFix16>>* mtxfix, matrix_real16* renormalize_data, int row_num, int col_num, int perm_num, matrix& perm, int useDual, int useFloat)
{
    if ( mtxfix->size() == 0 ) return;

    lock_lib();
    if (!useFloat) init_dfe_lib(DFE_MAIN, useDual);
    else if (useFloat) init_dfe_lib(DFE_FLOAT, useDual);

    //note: stride must equal number of columns, or this will not work as the C call expects contiguous data
    ComplexFix16* mtx_fix_data[mtxfix->size()];
    for (size_t i = 0; i < mtxfix->size(); i++) {
        mtx_fix_data[i] =(*mtxfix)[i].get_data();
    }

    if (useFloat)
        calcPermanentGlynnDFEF( (const ComplexFix16**)mtx_fix_data, renormalize_data->get_data(), row_num, col_num, perm_num, perm.get_data());
    else
        calcPermanentGlynnDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data->get_data(), row_num, col_num, perm_num, perm.get_data());

    unlock_lib();

}

/**
@brief ???????
*/
void
GlynnPermanentCalculatorBatch_DFE(std::vector<matrix>& matrices, matrix& perm, int useDual, int useFloat)
{

    if ( matrices.size() == 0 ) return;

    lock_lib();
    if (!useFloat) init_dfe_lib(DFE_MAIN, useDual);
    else if (useFloat) init_dfe_lib(DFE_FLOAT, useDual);

    if (!((!useFloat && calcPermanentGlynnDFE) || (useFloat && calcPermanentGlynnDFEF)) ||
        matrices.begin()->rows < 1+dfe_basekernpow2) { //compute with other method
      GlynnPermanentCalculatorLongDouble gpc;
      for (size_t i = 0; i < matrices.size(); i++) {
          perm[i] = gpc.calculate(matrices[i]);
      }
      unlock_lib();
      dec_dfe_lib_count();
      return;
    }

    // get data for matrix renormalization
    matrix_real16* renormalize_data;

    if (!useFloat) {
        renormalize_data = get_renormalization_data(&matrices);
    }
    else {
        renormalize_data = new matrix_real16(matrices.size(), matrices.begin()->cols);
    }

    // renormalize the matrices for DFE calculation
    std::vector<matrix_base<ComplexFix16>>* mtxfix = renormalize_matrices( &matrices, renormalize_data, useFloat );

    // calculate the permanent on DFE
    GlynnPermanentCalculatorBatch_DFE(mtxfix, renormalize_data, matrices.begin()->rows, matrices.begin()->cols, matrices.size(), perm, useDual, useFloat);

    delete(mtxfix);
    delete(renormalize_data);

    
    unlock_lib();
    return;
}

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculator_DFE(matrix& matrix_mtx, Complex16& perm, int useDual, int useFloat)
{
    lock_lib();
    if (!useFloat) init_dfe_lib(DFE_MAIN, useDual);
    else if (useFloat) init_dfe_lib(DFE_FLOAT, useDual);

    if (!((!useFloat && calcPermanentGlynnDFE) || (useFloat && calcPermanentGlynnDFEF)) ||
        matrix_mtx.rows < 1+dfe_basekernpow2 || matrix_mtx.cols == 0 || matrix_mtx.rows >= matrix_mtx.cols + 2) { //compute with other method
      GlynnPermanentCalculatorLongDouble gpc;
      perm = gpc.calculate(matrix_mtx);
      unlock_lib();
      dec_dfe_lib_count();
      return;
    }
    matrix_base<long double> renormalize_data(matrix_mtx.cols, 1);
    if (!useFloat) {
        // calulate the maximal sum of the columns to normalize the matrix
        matrix_base<Complex32> colSumMax( matrix_mtx.cols, 2);
        memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
        //sum up vectors in first/upper-left and fourth/lower-right quadrants
        for (size_t idx=0; idx<matrix_mtx.rows; idx++) {            
            for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
                size_t offset = idx*matrix_mtx.stride + jdx;
                int realPos = matrix_mtx[offset].real() > 0;
                int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
                if (realPos) colSumMax[2*jdx+slopeUpLeft] += matrix_mtx[offset];
                else colSumMax[2*jdx+slopeUpLeft] -= matrix_mtx[offset];
            }    
        }
        //now try to add/subtract neighbor quadrant values to the prior sum vector to see if it increase the absolute value 
        for (size_t idx=0; idx<matrix_mtx.rows; idx++) {
            for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
                size_t offset = idx*matrix_mtx.stride + jdx;
                int realPos = matrix_mtx[offset].real() > 0;
                int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
                Complex32 value1 = colSumMax[2*jdx+1-slopeUpLeft] + matrix_mtx[offset];
                Complex32 value2 = colSumMax[2*jdx+1-slopeUpLeft] - matrix_mtx[offset];
                colSumMax[2*jdx+1-slopeUpLeft] = std::norm(value1) > std::norm(value2) ? value1 : value2;
            } 
        }
    
        // calculate the renormalization coefficients
        for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
            renormalize_data[jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
            //printf("%d %.21Lf\n", jdx, renormalize_data[jdx]);
        }
    }

    // renormalize the input matrix and convert to fixed point maximizing precision via long doubles
    // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
    const size_t max_dim = dfe_mtx_size;
    const size_t rows = matrix_mtx.rows;
    const size_t numinits = 4;
    const size_t max_fpga_cols = max_dim / numinits;
    const size_t actualinits = (matrix_mtx.cols + max_fpga_cols-1) / max_fpga_cols;
    matrix_base<ComplexFix16> mtxfix[numinits] = {};
    const long double fixpow = 1ULL << 62;
    const double fOne = doubleToLLRaw(1.0);
    for (size_t i = 0; i < actualinits; i++) {
      mtxfix[i] = matrix_base<ComplexFix16>(rows, max_fpga_cols);
      size_t basecol = max_fpga_cols * i;
      size_t lastcol = matrix_mtx.cols<=basecol ? 0 : std::min(max_fpga_cols, matrix_mtx.cols-basecol);
      for (size_t idx=0; idx < rows; idx++) {
        size_t offset = idx * matrix_mtx.stride + basecol;
        size_t offset_small = idx*mtxfix[i].stride;
        for (size_t jdx = 0; jdx < lastcol; jdx++) {
          mtxfix[i][offset_small+jdx].real = useFloat ? doubleToLLRaw(matrix_mtx[offset+jdx].real()) : llrint((long double)matrix_mtx[offset+jdx].real() * fixpow / renormalize_data[basecol+jdx]);
          mtxfix[i][offset_small+jdx].imag = useFloat ? doubleToLLRaw(matrix_mtx[offset+jdx].imag()) : llrint((long double)matrix_mtx[offset+jdx].imag() * fixpow / renormalize_data[basecol+jdx]);
          //printf("%d %d %d %llX %llX\n", i, idx, jdx, mtxfix[i][offset_small+jdx].real, mtxfix[i][offset_small+jdx].imag); 
        }
        memset(&mtxfix[i][offset_small+lastcol], 0, sizeof(ComplexFix16)*(max_fpga_cols-lastcol));
      }
      for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxfix[i][jdx].real = useFloat ? fOne : fixpow; 
    }

    //note: stride must equal number of columns, or this will not work as the C call expects contiguous data
    ComplexFix16* mtx_fix_data[numinits];
    //assert(mtxfix[i].stride == mtxfix[i].cols);
    //assert(matrix_mtx.rows == matrix_mtx.cols && matrix_mtx.rows <= dfe_mtx_size);
    for (size_t i = 0; i < numinits; i++) mtx_fix_data[i] = mtxfix[i].get_data();
   
    if (useFloat)
        calcPermanentGlynnDFEF( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, 1, &perm);
    else
        calcPermanentGlynnDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, 1, &perm);

    unlock_lib();

    return;
}

}
