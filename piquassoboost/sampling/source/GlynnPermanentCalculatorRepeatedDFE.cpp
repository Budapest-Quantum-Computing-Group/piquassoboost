#include "GlynnPermanentCalculatorRepeatedDFE.h"
#include "GlynnPermanentCalculatorRepeated.h"
#include "BatchedPermanentCalculator.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif
#include <vector>

typedef void(*CALCPERMGLYNNREPDFE)(const pic::ComplexFix16**, const long double*, const uint64_t, const uint64_t, const unsigned char*,
  const uint8_t*, const uint8_t, const uint8_t, const uint64_t*, const uint64_t, const uint8_t, pic::Complex16*);
typedef int(*INITPERMGLYNNREPDFE)(size_t*, size_t*);
typedef void(*FREEPERMGLYNNREPDFE)(void);

CALCPERMGLYNNREPDFE calcPermanentGlynnRepDFE = NULL;
INITPERMGLYNNREPDFE initializeRep_DFE = NULL;
FREEPERMGLYNNREPDFE releiveRep_DFE = NULL;

#define ROWCOL(m, r, c) ToComplex32(m[ r*m.stride + c])

namespace pic {

static double t_DFE = 0.0;
static double t_DFE_tot = 0.0;

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
uint64_t binomial_gcode(uint64_t bc, int parity, uint64_t n, uint64_t k)
{
  return parity ? bc*k/(n-k+1) : bc*(n-k)/(k+1);
}

/**
@brief ???????
*/
matrix transpose_reorder_rows_cols(matrix& matrix_mtx, std::vector<uint8_t> & rowchange_indices, std::vector<uint8_t> & colIndices, int transpose)
{
    matrix matrix_rows(rowchange_indices.size(), colIndices.size());
    if (transpose) {
        for (size_t i = 0; i < colIndices.size(); i++) {
            size_t offset = colIndices[i]*matrix_mtx.stride;
            for (size_t j = 0; j < rowchange_indices.size(); j++) {
                matrix_rows[j*matrix_rows.stride+i] = matrix_mtx[offset+rowchange_indices[j]];
            }
        }
    } else {
        for (size_t i = 0; i < rowchange_indices.size(); i++) {
            size_t offset = rowchange_indices[i]*matrix_mtx.stride;
            size_t newoffset = i*matrix_rows.stride;
            for (size_t j = 0; j < colIndices.size(); j++) {
                matrix_rows[newoffset+j] = matrix_mtx[offset+colIndices[j]];
            }
        }
    }
    return matrix_rows;
}

/**
@brief ???????
*/
inline void symmetricQuadrantNormalize(Complex32* sums, Complex16 val) {
    Complex32 value1 = sums[0] + val;
    Complex32 value2 = sums[0] - val;
    Complex32 value3 = sums[1] + val;
    Complex32 value4 = sums[1] - val;
    int symQuad1 = (value1.real() < 0) == (value1.imag() < 0);                  
    int symQuad2 = (value2.real() < 0) == (value2.imag() < 0);
    int symQuad3 = (value3.real() < 0) == (value3.imag() < 0);
    int symQuad4 = (value4.real() < 0) == (value4.imag() < 0);
    if (symQuad1 == symQuad2) { 
        sums[symQuad1] = std::norm(value1) > std::norm(value2) ? value1 : value2;
        if (symQuad3 == symQuad2) {
            sums[symQuad3] = std::norm(value3) > std::norm(sums[symQuad3]) ? value3 : sums[symQuad3];
            if (symQuad4 == symQuad3) {
                sums[symQuad4] = std::norm(value4) > std::norm(sums[symQuad4]) ? value4 : sums[symQuad4];
            } else sums[symQuad4] = value4;
        } else {
            sums[symQuad3] = value3;
            sums[symQuad4] = std::norm(value4) > std::norm(sums[symQuad4]) ? value4 : sums[symQuad4];
        }
    } else {
        sums[symQuad1] = value1;
        sums[symQuad2] = value2;
        sums[symQuad3] = std::norm(value3) > std::norm(sums[symQuad3]) ? value3 : sums[symQuad3];  
        sums[symQuad4] = std::norm(value4) > std::norm(sums[symQuad4]) ? value4 : sums[symQuad4];
    }
}

typedef void(*CALCPERMGLYNNDFE)(const pic::ComplexFix16**, const long double*, const uint64_t, const uint64_t, const uint64_t, pic::Complex16*);
extern "C" CALCPERMGLYNNDFE calcPermanentGlynnDFE;
extern "C" CALCPERMGLYNNDFE calcPermanentGlynnDFEF;

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
cGlynnPermanentCalculatorRepeatedMulti_DFE::cGlynnPermanentCalculatorRepeatedMulti_DFE() {


    mtxfix = NULL;
}

/**
@brief Constructor of the class.
@param ???????????????????
@return Returns with the instance of the class.
*/
cGlynnPermanentCalculatorRepeatedMulti_DFE::cGlynnPermanentCalculatorRepeatedMulti_DFE( matrix& matrix_mtx_in, PicState_int64& input_state_in, PicState_int64& output_state_in, int useDual_in  ) {


    // set 1 to use floating point number representation in DFE (not supported yet) or 0 to use fixed points
    useFloat = 0;  
    //
    doCPU = false;
    //
    mtxfix = new matrix_base<ComplexFix16>[numinits];
    //
    matrix_base<long double> renormalize_data_all;
    //
    onerows = 0;
    // The number of photons
    photons = 0;
    //
    totalPerms = 0;
    //
    mulsum = 0;
    //
    useDual = useDual_in;
    //
    matrix_init = matrix_mtx_in;
    //
    input_state = input_state_in;
    //
    output_state = output_state_in;
}

/**
@brief Destructor of the class
*/
cGlynnPermanentCalculatorRepeatedMulti_DFE::~cGlynnPermanentCalculatorRepeatedMulti_DFE() {


    if (mtxfix != NULL) {
        delete[](mtxfix);
    }


}


/**
@brief ???????
*/
void
cGlynnPermanentCalculatorRepeatedMulti_DFE::prepareDataForRepeatedMulti_DFE()
{
    // if nothing uneaxpected happens the permanent calculations would be done on DFE not on CPU
    doCPU = false;

    // determine the number of particles
    photons = 0;
    int transpose = 0;
    for (size_t i = 0; i < input_state.size(); i++) {
        photons += input_state[i];
        transpose += ((input_state[i] != 0) ? 1 : 0) - ((output_state[i] != 0) ? 1 : 0);  
    }

    if (photons < 1+dfe_basekernpow2) {
        doCPU = true;
        return;
    }

    transpose = transpose < 0; //transpose if needed to reduce complexity on rows direction
    const size_t max_dim = dfe_mtx_size;
    //convert multiplicities of rows and columns to indices
    std::vector<unsigned char> colIndices; colIndices.reserve(max_dim);
    for (size_t i = 0; i < output_state.size(); i++) {
      for (size_t j = transpose ? output_state[i] : input_state[i]; j != 0; j--) {
        colIndices.push_back(i);
      }
    }
 
    PicState_int64 adj_input_state = transpose ? input_state.copy() : output_state.copy();  
    std::vector<uint8_t> mrows;
    std::vector<uint8_t> row_indices;
    for (size_t i = 0; i < adj_input_state.size(); i++) {
        if (adj_input_state[i] == 1) row_indices.push_back(i);
        else if (adj_input_state[i] > 1) mrows.push_back(i);
    }
 
    //sort multiplicity >=2 row indices since we need anchor rows, and complexity reduction greatest by using smallest multiplicities
    sort(mrows.begin(), mrows.end(), [&adj_input_state](size_t i, size_t j) { return adj_input_state[i] < adj_input_state[j]; }); 
    if (row_indices.size() < 1) { //Glynn anchor row
        row_indices.push_back(mrows[0]);
        if (--adj_input_state[mrows[0]] == 1) {
          row_indices.push_back(mrows[0]);
          mrows.erase(mrows.begin());
        }
    }

    //construct multiplicity Gray code counters
    mulsum = 0;
    onerows = row_indices.size();
    std::vector<uint64_t> curmp, inp;
    
    // determine the number of smaller permanents
    totalPerms = 1;
    for (size_t i = 0; i < mrows.size(); i++) {
        //for (size_t j = 0; j < adj_input_state[mrows[i]]; j++)
        row_indices.push_back(mrows[i]);
        curmp.push_back(adj_input_state[mrows[i]]);
        inp.push_back(adj_input_state[mrows[i]]);
        mulsum += adj_input_state[mrows[i]];
        totalPerms *= (adj_input_state[mrows[i]] + 1);
    }


    if (onerows < 1+dfe_basekernpow2) { //pass the calculation to CPU
        doCPU = true;
        return;
    }


    // calulate the maximal sum of the columns to normalize the matrix
    matrix_base<Complex32> colSumMax( photons, 2);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
    for (size_t i=0; i<row_indices.size(); i++) {
        //size_t offset = (transpose ? colIndices[i] : row_indices[i]) * matrix_init.stride;
        for (int64_t idx = 0; idx < (i < onerows ? 1 : adj_input_state[row_indices[i]]); idx++) {
            for( size_t jdx=0; jdx<photons; jdx++) {
                size_t offset = transpose ? colIndices[jdx]*matrix_init.stride+row_indices[i] : row_indices[i]*matrix_init.stride+colIndices[jdx];
                symmetricQuadrantNormalize(&colSumMax[2*jdx], matrix_init[offset]);
            }
        }
    }

    // calculate the renormalization coefficients
    matrix_base<long double> renormalize_data(1, photons);
    for (size_t jdx=0; jdx<photons; jdx++ ) {
        renormalize_data[jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
    }
    
    // renormalize the input matrix and convert to fixed point maximizing precision via long doubles
    // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
    const size_t rows = row_indices.size();
    const size_t numinits = 4;
    const size_t max_fpga_cols = max_dim / numinits;
    const size_t actualinits = (photons + max_fpga_cols-1) / max_fpga_cols;
    matrix_base<ComplexFix16> mtxprefix[numinits] = {};
    const long double fixpow = 1ULL << 62;
    for (size_t i = 0; i < actualinits; i++) {
      mtxprefix[i] = matrix_base<ComplexFix16>(rows, max_fpga_cols);
      size_t basecol = max_fpga_cols * i;
      size_t lastcol = photons<=basecol ? 0 : std::min(max_fpga_cols, photons-basecol);
      for (size_t idx=0; idx < rows; idx++) {
        size_t offset_small = idx*mtxprefix[i].stride;
        for (size_t jdx = 0; jdx < lastcol; jdx++) {
          size_t offset = transpose ? colIndices[basecol+jdx]*matrix_init.stride+row_indices[idx] : row_indices[idx]*matrix_init.stride+colIndices[basecol+jdx];
          mtxprefix[i][offset_small+jdx].real = llrint((long double)matrix_init[offset].real() * fixpow / renormalize_data[basecol+jdx]);
          mtxprefix[i][offset_small+jdx].imag = llrint((long double)matrix_init[offset].imag() * fixpow / renormalize_data[basecol+jdx]);
          if (idx >= onerows) { //start with all positive Gray codes, so sum everything onto the adjust row
              for (int64_t j = 0; j < adj_input_state[row_indices[idx]]; j++) {
                  mtxprefix[i][jdx].real += mtxprefix[i][offset_small+jdx].real;
                  mtxprefix[i][jdx].imag += mtxprefix[i][offset_small+jdx].imag;
              }
          }
        }
        memset(&mtxprefix[i][offset_small+lastcol], 0, sizeof(ComplexFix16)*(max_fpga_cols-lastcol));
      }
      for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxprefix[i][jdx].real = fixpow;
    }
    
    //matrix_base<ComplexFix16> mtxfix[numinits] = {};
    for (size_t i = 0; i < actualinits; i++)
        mtxfix[i] = matrix_base<ComplexFix16>(onerows * totalPerms, max_fpga_cols);
  
    Complex32 res;
    uint64_t gcodeidx = 0, cur_multiplicity = 1, skipidx = (1ULL << curmp.size())-1; //gcodeidx is direction bit vector, skipidx set to not skip all indexes - technically "not skip index"
   
    while (true) {
  
        size_t offset_small = mplicity.size()*onerows*max_fpga_cols;
        for (size_t i = 0; i < actualinits; i++) {
            memcpy(mtxfix[i].get_data()+offset_small, mtxprefix[i].get_data(), sizeof(ComplexFix16) * onerows * max_fpga_cols);
        }
        mplicity.push_back(cur_multiplicity);   
        
             
        if (skipidx == 0) { 
            // at last step determine the normalization data
            renormalize_data_all = matrix_base<long double>(totalPerms, photons);
            for (size_t i = 0; i < totalPerms; i++) memcpy(renormalize_data_all.get_data()+photons*i, renormalize_data.get_data(), photons * sizeof(long double));  
  
            return;
        }
        
        
        size_t i = __builtin_ctzll(skipidx); //count of trailing zeros to compute next change index
        bool curdir = (gcodeidx & (1ULL << i)) == 0;
        cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[i], (curmp[i] + inp[i]) / 2);
        curmp[i] = curdir ? curmp[i] - 2 : curmp[i] + 2;
        size_t offset = (onerows+i)*max_fpga_cols;
        //add or subtract to the adjustment row
        for (size_t idx = 0; idx < actualinits; idx++) {
            for (size_t j = 0; j < max_fpga_cols; j++) { //Gray code adjustment by adding or subtracting 2 times the appropriate row, caring for overflow since we are (64, -62) fixed point cannot multiply by 2 which requires 65 bits
                if (curdir) {
                    mtxprefix[idx][j].real = (mtxprefix[idx][j].real - mtxprefix[idx][offset+j].real) - mtxprefix[idx][offset+j].real;
                    mtxprefix[idx][j].imag = (mtxprefix[idx][j].imag - mtxprefix[idx][offset+j].imag) - mtxprefix[idx][offset+j].imag;
                } else {
                    mtxprefix[idx][j].real = (mtxprefix[idx][j].real + mtxprefix[idx][offset+j].real) + mtxprefix[idx][offset+j].real;
                    mtxprefix[idx][j].imag = (mtxprefix[idx][j].imag + mtxprefix[idx][offset+j].imag) + mtxprefix[idx][offset+j].imag;
                }
            }            
        }
        if ((!curdir && curmp[i] == inp[i]) || (curdir && curmp[i] == -inp[i])) skipidx ^= ((1ULL << (i+1)) - 1); //set all skipping before and including current index
        else skipidx ^= ((1ULL << i) - 1); //flip all skipping which come before current index
        gcodeidx ^= (1ULL << i) - 1; //flip all directions which come before current index
    }



}

/**
@brief ???????
*/
Complex16
cGlynnPermanentCalculatorRepeatedMulti_DFE::calculate()
{

    Complex16 perm(0.0,0.0);

    lock_lib();
    if (!useFloat) init_dfe_lib(DFE_MAIN, useDual);
    else if (useFloat) init_dfe_lib(DFE_FLOAT, useDual);    

    const size_t numinits = 4;

    if (doCPU) { //compute with other method
      GlynnPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_init, input_state, output_state);
      unlock_lib();
      return perm;
    }

    std::vector<Complex16> perms;
    perms.resize(totalPerms);
    //GlynnPermanentCalculatorBatch_DFE(matrices, perms, useDual, false);
    //note: stride must equal number of columns, or this will not work as the C call expects contiguous data
    ComplexFix16* mtx_fix_data[numinits];

    //assert(mtxfix[i].stride == mtxfix[i].cols);
    for (size_t i = 0; i < numinits; i++) {
        mtx_fix_data[i] = mtxfix[i].get_data();
    }

    if (totalPerms == 1) {
        matrix&& modifiedInterferometerMatrix = adaptInterferometerGlynnMultiplied(matrix_init, &input_state, &output_state );
        GlynnPermanentCalculator_DFE(modifiedInterferometerMatrix, perm, useDual, useFloat);
        unlock_lib();
        return perm;
    }

    if (useFloat) {
        calcPermanentGlynnDFEF( (const ComplexFix16**)mtx_fix_data, renormalize_data_all.get_data(), onerows, photons, totalPerms, perms.data());
    }
    else {
tbb::tick_count t0 = tbb::tick_count::now();            
        calcPermanentGlynnDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data_all.get_data(), onerows, photons, totalPerms, perms.data());    
tbb::tick_count t1 = tbb::tick_count::now();
t_DFE += (t1-t0).seconds();                                       
    }

    Complex32 res_tmp(0.0,0.0);
    for (size_t i = 0; i < perms.size(); i++) {
        if (i & 1) {
            res_tmp -= ToComplex32(perms[i]) * (long double)mplicity[i]; 
        }
        else {
            res_tmp += ToComplex32(perms[i]) * (long double)mplicity[i];
        }
    }
    perm = ToComplex16(res_tmp / (long double)(1ULL << mulsum)); //2**mulsum is the effective number of permanents or sum of all multiplicities
    unlock_lib();

tbb::tick_count t1_tot = tbb::tick_count::now();
//t_DFE_tot += (t1_tot-t0_tot).seconds();
//std::cout << "clean DFE time: " << t_DFE << ", tot DFE time:" << t_DFE_tot << std::endl; 

    return perm;


}



/**
@brief ???????
*/
void
prepareDataForRepeatedMulti_DFE(matrix& matrix_init, PicState_int64& input_state, PicState_int64& output_state, int useFloat,
matrix_base<ComplexFix16>* mtxfix, matrix_base<long double>& renormalize_data_all, std::vector<uint64_t>& mplicity, uint8_t& onerows, size_t& photons, uint64_t& totalPerms, uint8_t& mulsum, bool& doCPU)
{
    // if nothing uneaxpected happens the permanent calculations would be done on DFE not on CPU
    doCPU = false;

    // determine the number of particles
    photons = 0;
    int transpose = 0;
    for (size_t i = 0; i < input_state.size(); i++) {
        photons += input_state[i];
        transpose += ((input_state[i] != 0) ? 1 : 0) - ((output_state[i] != 0) ? 1 : 0);  
    }

    if (photons < 1+dfe_basekernpow2) {
        doCPU = true;
        return;
    }

    transpose = transpose < 0; //transpose if needed to reduce complexity on rows direction
    const size_t max_dim = dfe_mtx_size;
    //convert multiplicities of rows and columns to indices
    std::vector<unsigned char> colIndices; colIndices.reserve(max_dim);
    for (size_t i = 0; i < output_state.size(); i++) {
      for (size_t j = transpose ? output_state[i] : input_state[i]; j != 0; j--) {
        colIndices.push_back(i);
      }
    }
 
    PicState_int64 adj_input_state = transpose ? input_state.copy() : output_state.copy();  
    std::vector<uint8_t> mrows;
    std::vector<uint8_t> row_indices;
    for (size_t i = 0; i < adj_input_state.size(); i++) {
        if (adj_input_state[i] == 1) row_indices.push_back(i);
        else if (adj_input_state[i] > 1) mrows.push_back(i);
    }
 
    //sort multiplicity >=2 row indices since we need anchor rows, and complexity reduction greatest by using smallest multiplicities
    sort(mrows.begin(), mrows.end(), [&adj_input_state](size_t i, size_t j) { return adj_input_state[i] < adj_input_state[j]; }); 
    if (row_indices.size() < 1) { //Glynn anchor row
        row_indices.push_back(mrows[0]);
        if (--adj_input_state[mrows[0]] == 1) {
          row_indices.push_back(mrows[0]);
          mrows.erase(mrows.begin());
        }
    }

    //construct multiplicity Gray code counters
    mulsum = 0;
    onerows = row_indices.size();
    std::vector<uint64_t> curmp, inp;
    
    // determine the number of smaller permanents
    totalPerms = 1;
    for (size_t i = 0; i < mrows.size(); i++) {
        //for (size_t j = 0; j < adj_input_state[mrows[i]]; j++)
        row_indices.push_back(mrows[i]);
        curmp.push_back(adj_input_state[mrows[i]]);
        inp.push_back(adj_input_state[mrows[i]]);
        mulsum += adj_input_state[mrows[i]];
        totalPerms *= (adj_input_state[mrows[i]] + 1);
    }


    if (onerows < 1+dfe_basekernpow2) { //pass the calculation to CPU
        doCPU = true;
        return;
    }


    // calulate the maximal sum of the columns to normalize the matrix
    matrix_base<Complex32> colSumMax( photons, 2);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
    for (size_t i=0; i<row_indices.size(); i++) {
        //size_t offset = (transpose ? colIndices[i] : row_indices[i]) * matrix_init.stride;
        for (int64_t idx = 0; idx < (i < onerows ? 1 : adj_input_state[row_indices[i]]); idx++) {
            for( size_t jdx=0; jdx<photons; jdx++) {
                size_t offset = transpose ? colIndices[jdx]*matrix_init.stride+row_indices[i] : row_indices[i]*matrix_init.stride+colIndices[jdx];
                symmetricQuadrantNormalize(&colSumMax[2*jdx], matrix_init[offset]);
            }
        }
    }

    // calculate the renormalization coefficients
    matrix_base<long double> renormalize_data(1, photons);
    for (size_t jdx=0; jdx<photons; jdx++ ) {
        renormalize_data[jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
    }
    
    // renormalize the input matrix and convert to fixed point maximizing precision via long doubles
    // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
    const size_t rows = row_indices.size();
    const size_t numinits = 4;
    const size_t max_fpga_cols = max_dim / numinits;
    const size_t actualinits = (photons + max_fpga_cols-1) / max_fpga_cols;
    matrix_base<ComplexFix16> mtxprefix[numinits] = {};
    const long double fixpow = 1ULL << 62;
    for (size_t i = 0; i < actualinits; i++) {
      mtxprefix[i] = matrix_base<ComplexFix16>(rows, max_fpga_cols);
      size_t basecol = max_fpga_cols * i;
      size_t lastcol = photons<=basecol ? 0 : std::min(max_fpga_cols, photons-basecol);
      for (size_t idx=0; idx < rows; idx++) {
        size_t offset_small = idx*mtxprefix[i].stride;
        for (size_t jdx = 0; jdx < lastcol; jdx++) {
          size_t offset = transpose ? colIndices[basecol+jdx]*matrix_init.stride+row_indices[idx] : row_indices[idx]*matrix_init.stride+colIndices[basecol+jdx];
          mtxprefix[i][offset_small+jdx].real = llrint((long double)matrix_init[offset].real() * fixpow / renormalize_data[basecol+jdx]);
          mtxprefix[i][offset_small+jdx].imag = llrint((long double)matrix_init[offset].imag() * fixpow / renormalize_data[basecol+jdx]);
          if (idx >= onerows) { //start with all positive Gray codes, so sum everything onto the adjust row
              for (int64_t j = 0; j < adj_input_state[row_indices[idx]]; j++) {
                  mtxprefix[i][jdx].real += mtxprefix[i][offset_small+jdx].real;
                  mtxprefix[i][jdx].imag += mtxprefix[i][offset_small+jdx].imag;
              }
          }
        }
        memset(&mtxprefix[i][offset_small+lastcol], 0, sizeof(ComplexFix16)*(max_fpga_cols-lastcol));
      }
      for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxprefix[i][jdx].real = fixpow;
    }
    
    //matrix_base<ComplexFix16> mtxfix[numinits] = {};
    for (size_t i = 0; i < actualinits; i++)
        mtxfix[i] = matrix_base<ComplexFix16>(onerows * totalPerms, max_fpga_cols);
  
    Complex32 res;
    uint64_t gcodeidx = 0, cur_multiplicity = 1, skipidx = (1ULL << curmp.size())-1; //gcodeidx is direction bit vector, skipidx set to not skip all indexes - technically "not skip index"
   
    while (true) {
  
        size_t offset_small = mplicity.size()*onerows*max_fpga_cols;
        for (size_t i = 0; i < actualinits; i++) {
            memcpy(mtxfix[i].get_data()+offset_small, mtxprefix[i].get_data(), sizeof(ComplexFix16) * onerows * max_fpga_cols);
        }
        mplicity.push_back(cur_multiplicity);   
        
             
        if (skipidx == 0) { 
            // at last step determine the normalization data
            renormalize_data_all = matrix_base<long double>(totalPerms, photons);
            for (size_t i = 0; i < totalPerms; i++) memcpy(renormalize_data_all.get_data()+photons*i, renormalize_data.get_data(), photons * sizeof(long double));  
  
            return;
        }
        
        
        size_t i = __builtin_ctzll(skipidx); //count of trailing zeros to compute next change index
        bool curdir = (gcodeidx & (1ULL << i)) == 0;
        cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[i], (curmp[i] + inp[i]) / 2);
        curmp[i] = curdir ? curmp[i] - 2 : curmp[i] + 2;
        size_t offset = (onerows+i)*max_fpga_cols;
        //add or subtract to the adjustment row
        for (size_t idx = 0; idx < actualinits; idx++) {
            for (size_t j = 0; j < max_fpga_cols; j++) { //Gray code adjustment by adding or subtracting 2 times the appropriate row, caring for overflow since we are (64, -62) fixed point cannot multiply by 2 which requires 65 bits
                if (curdir) {
                    mtxprefix[idx][j].real = (mtxprefix[idx][j].real - mtxprefix[idx][offset+j].real) - mtxprefix[idx][offset+j].real;
                    mtxprefix[idx][j].imag = (mtxprefix[idx][j].imag - mtxprefix[idx][offset+j].imag) - mtxprefix[idx][offset+j].imag;
                } else {
                    mtxprefix[idx][j].real = (mtxprefix[idx][j].real + mtxprefix[idx][offset+j].real) + mtxprefix[idx][offset+j].real;
                    mtxprefix[idx][j].imag = (mtxprefix[idx][j].imag + mtxprefix[idx][offset+j].imag) + mtxprefix[idx][offset+j].imag;
                }
            }            
        }
        if ((!curdir && curmp[i] == inp[i]) || (curdir && curmp[i] == -inp[i])) skipidx ^= ((1ULL << (i+1)) - 1); //set all skipping before and including current index
        else skipidx ^= ((1ULL << i) - 1); //flip all skipping which come before current index
        gcodeidx ^= (1ULL << i) - 1; //flip all directions which come before current index
    }



}


/**
@brief ???????
*/
void
GlynnPermanentCalculatorRepeatedMulti_DFE(matrix& matrix_init, PicState_int64& input_state, PicState_int64& output_state, const int useFloat,
const matrix_base<ComplexFix16>* mtxfix, const matrix_base<long double>& renormalize_data_all, const std::vector<uint64_t>& mplicity, const uint8_t& onerows, const size_t& photons, const uint64_t& totalPerms, const uint8_t& mulsum, const bool& doCPU, const int& useDual, Complex16& perm)
{
    lock_lib();
    if (!useFloat) init_dfe_lib(DFE_MAIN, useDual);
    else if (useFloat) init_dfe_lib(DFE_FLOAT, useDual);    

    const size_t numinits = 4;

    if (doCPU) { //compute with other method
      GlynnPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_init, input_state, output_state);
      unlock_lib();
      return;
    }

    std::vector<Complex16> perms;
    perms.resize(totalPerms);
    //GlynnPermanentCalculatorBatch_DFE(matrices, perms, useDual, false);
    //note: stride must equal number of columns, or this will not work as the C call expects contiguous data
    ComplexFix16* mtx_fix_data[numinits];

    //assert(mtxfix[i].stride == mtxfix[i].cols);
    for (size_t i = 0; i < numinits; i++) {
        mtx_fix_data[i] = mtxfix[i].get_data();
    }

    if (totalPerms == 1) {
        matrix&& modifiedInterferometerMatrix = adaptInterferometerGlynnMultiplied(matrix_init, &input_state, &output_state );
        GlynnPermanentCalculator_DFE(modifiedInterferometerMatrix, perm, useDual, useFloat);
        unlock_lib();
        return;
    }

    if (useFloat) {
        calcPermanentGlynnDFEF( (const ComplexFix16**)mtx_fix_data, renormalize_data_all.get_data(), onerows, photons, totalPerms, perms.data());
    }
    else {
tbb::tick_count t0 = tbb::tick_count::now();            
        calcPermanentGlynnDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data_all.get_data(), onerows, photons, totalPerms, perms.data());    
tbb::tick_count t1 = tbb::tick_count::now();
t_DFE += (t1-t0).seconds();                                       
    }

    Complex32 res_tmp(0.0,0.0);
    for (size_t i = 0; i < perms.size(); i++) {
        if (i & 1) {
            res_tmp -= ToComplex32(perms[i]) * (long double)mplicity[i]; 
        }
        else {
            res_tmp += ToComplex32(perms[i]) * (long double)mplicity[i];
        }
    }
    perm = ToComplex16(res_tmp / (long double)(1ULL << mulsum)); //2**mulsum is the effective number of permanents or sum of all multiplicities
    unlock_lib();

tbb::tick_count t1_tot = tbb::tick_count::now();
//t_DFE_tot += (t1_tot-t0_tot).seconds();
//std::cout << "clean DFE time: " << t_DFE << ", tot DFE time:" << t_DFE_tot << std::endl; 

    return;


}
/**
@brief ???????
*/
void
GlynnPermanentCalculatorRepeatedMulti_DFE(matrix& matrix_init, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual)
{
tbb::tick_count t0_tot = tbb::tick_count::now(); 
    lock_lib();
    int useFloat = 0;
    if (!useFloat) init_dfe_lib(DFE_MAIN, useDual);
    else if (useFloat) init_dfe_lib(DFE_FLOAT, useDual);  
    
    cGlynnPermanentCalculatorRepeatedMulti_DFE DFEcalculator(matrix_init, input_state, output_state, useDual );
    DFEcalculator.prepareDataForRepeatedMulti_DFE();
    perm = DFEcalculator.calculate();
    
    unlock_lib();
    return;

}


/**
@brief ???????
*/
matrix transpose_reorder_rows(matrix& matrix_mtx, std::vector<uint8_t> & rowchange_indices)
{
    matrix matrix_rows(rowchange_indices.size(), matrix_mtx.rows);
    for (size_t i = 0; i < matrix_mtx.rows; i++) {
        size_t offset = i*matrix_mtx.stride;
        for (size_t j = 0; j < rowchange_indices.size(); j++) {
            matrix_rows[j*matrix_rows.stride+i] = matrix_mtx[offset+rowchange_indices[j]];
        }
    }
    return matrix_rows;
}


/**
@brief ???????
*/
matrix input_to_bincoeff_indices(matrix& matrix_mtx, PicState_int64& input_state, int useDual, std::vector<uint8_t> & rowchange_indices, std::vector<uint64_t> & mplicity, uint8_t & onerows, uint64_t & changecount, uint8_t & mulsum)
{
  std::vector<uint8_t> mrows;
  std::vector<uint8_t> row_indices;
  for (size_t i = 0; i < input_state.size(); i++) {
    if (input_state[i] == 1) row_indices.push_back(i);
    else if (input_state[i] > 1) mrows.push_back(i);
  }
  sort(mrows.begin(), mrows.end(), [&input_state](size_t i, size_t j) { return input_state[i] < input_state[j]; }); 
  while (row_indices.size() < 1+dfe_basekernpow2) { //Glynn anchor row, plus 2/3 anchor rows needed for binary Gray code in kernel
    row_indices.push_back(mrows[0]);
    if (--input_state[mrows[0]] == 1) {
      row_indices.push_back(mrows[0]);
      mrows.erase(mrows.begin());
    }
  }
  onerows = row_indices.size(), mulsum = 0, changecount = 0;
  std::vector<uint64_t> curmp, inp;
  for (size_t i = 0; i < mrows.size(); i++) {
    row_indices.push_back(mrows[i]);
    curmp.push_back(input_state[mrows[i]]);
    inp.push_back(input_state[mrows[i]]);
    mulsum += input_state[mrows[i]];
  }
  matrix matrix_rows = transpose_reorder_rows(matrix_mtx, row_indices);
  for (size_t i = 0; i < row_indices.size(); i++) {
      for (size_t j = i < onerows ? 1 : input_state[row_indices[i]]; j != 0; j--) {
        rowchange_indices.push_back(i);
      }
  }
  if (mrows.size() == 0) { mplicity.push_back(1); return matrix_rows; }
  int parity = 0;
  uint64_t gcodeidx = 0, cur_multiplicity = 1, skipidx = (1ULL << curmp.size())-1;
  while (true) {
    mplicity.push_back(cur_multiplicity);
    if (skipidx == 0) {
        return matrix_rows;
    }
    parity = !parity;
    size_t i = __builtin_ctzll(skipidx);
    bool curdir = (gcodeidx & (1ULL << i)) == 0;
    cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[i], (curmp[i] + inp[i]) / 2);
    curmp[i] = curdir ? curmp[i] - 2 : curmp[i] + 2;
    rowchange_indices.push_back((onerows+i) | (curdir ? 0x80 : 0)); //high bit indicates subtraction
    changecount++;
    if ((!curdir && curmp[i] == inp[i]) || (curdir && curmp[i] == -inp[i])) skipidx ^= ((1ULL << (i+1)) - 1);
    else skipidx ^= ((1ULL << i) - 1);
    gcodeidx ^= (1ULL << i) - 1;
  }
}

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculatorRepeated_DFE(matrix& matrix_init, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual)
{
    lock_lib();
    init_dfe_lib(DFE_REP, useDual);    
    size_t photons = 0;
    for (size_t i = 0; i < input_state.size(); i++) {
        photons += input_state[i];
    }
    if (!calcPermanentGlynnRepDFE || photons < 1+dfe_basekernpow2) { //compute with other method
      GlynnPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_init, input_state, output_state);
      unlock_lib();
      return;
    }
    std::vector<uint8_t> rowchange_indices;
    std::vector<uint64_t> mplicity;
    uint8_t onerows, mulsum; uint64_t changecount;
    PicState_int64 adj_input_state = input_state.copy();
    matrix matrix_mtx = input_to_bincoeff_indices(matrix_init, adj_input_state, useDual, rowchange_indices, mplicity, onerows, changecount, mulsum); 
    
    // calulate the maximal sum of the columns to normalize the matrix
    matrix_base<Complex32> colSumMax( matrix_mtx.cols, 2);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
    for (size_t i=0; i<photons; i++) {
        size_t idx = rowchange_indices[i];
        for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
            size_t offset = idx*matrix_mtx.stride + jdx;
            symmetricQuadrantNormalize(&colSumMax[2*jdx], matrix_mtx[offset]);
        }
    }

    // calculate the renormalization coefficients
    matrix_base<long double> renormalize_data(matrix_mtx.cols, 1);
    for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
        //printf("%d %.21Lf\n", jdx, renormalize_data[jdx]);
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
    for (size_t i = 0; i < numinits; i++) mtx_fix_data[i] = mtxfix[i].get_data();
    
    std::vector<unsigned char> colIndices; colIndices.reserve(max_dim);
    for (size_t i = 0; i < output_state.size(); i++) {
      for (size_t j = output_state[i]; j != 0; j--) {
        colIndices.push_back(i);
      }
    }
    for (size_t i = (colIndices.size() % 16 == 0) ? 0 : (16 - colIndices.size() % 16); i != 0; i--) colIndices.push_back(0); //round up to nearest 16 bytes to allow streaming
    for (size_t i = (rowchange_indices.size() % 16 == 0) ? 0 : (16 - rowchange_indices.size() % 16); i != 0; i--) rowchange_indices.push_back(0); //round up to nearest 16 bytes to allow streaming
    calcPermanentGlynnRepDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, colIndices.data(),
      rowchange_indices.data(), photons, onerows, mplicity.data(), changecount, mulsum, &perm);

    unlock_lib();
    return;
}

}
