#include "GlynnPermanentCalculatorRepeatedDFE.h"
#include "GlynnPermanentCalculatorRepeated.h"
#include "BatchedPermanentCalculator.h"
#include <fstream>
#include <iostream>

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif
#include <vector>
#include "common_functionalities.h"

typedef void(*CALCPERMGLYNNREPDFE)(const pic::ComplexFix16**, const long double*, const uint64_t, const uint64_t, const unsigned char*,
  const uint8_t*, const uint8_t*, const uint8_t, const uint8_t, const uint64_t*, const uint64_t, const uint8_t, pic::Complex16*);
typedef int(*INITPERMGLYNNREPDFE)(int, size_t*, size_t*);
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
    mtxfix = NULL;//new matrix_base<ComplexFix16>[numinits];
    //
    //matrix_base<long double> renormalize_data_all;
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

    reset();

}


/**
@brief Call to convert multiplicities of columns to indices
*/
void 
cGlynnPermanentCalculatorRepeatedMulti_DFE::determineColIndices( PicState_int64& input_state ) {


    //convert multiplicities of columns to indices
    colIndices.clear();
    colIndices.reserve(max_dim);
    for (size_t i = 0; i < input_state.size(); i++) {
        for (size_t j = 0; j<input_state[i]; j++) {
            colIndices.push_back(i);
        }
    }

}



/**
@brief ????????????
*/
void 
cGlynnPermanentCalculatorRepeatedMulti_DFE::reserveSpace()  {

    

}

/**
@brief Call to convert multiplicities of columns to indices
*/
void 
cGlynnPermanentCalculatorRepeatedMulti_DFE::determineColIndices() {

    determineColIndices( input_state );
}


/**
@brief ???????
*/
void 
cGlynnPermanentCalculatorRepeatedMulti_DFE::determineMultiplicitiesForRepeatedMulti_DFE() {


    // if nothing uneaxpected happens the permanent calculations would be done on DFE not on CPU
    doCPU = false;

    // determine the number of particles
    photons = 0;
    for (size_t i = 0; i < output_state.size(); i++) {
        photons += output_state[i];
    }

    if (photons < 1+dfe_basekernpow2) {
        doCPU = true;
        return;
    }

    
    //convert multiplicities of columns to indices
    //determineColIndices();
 
    output_state_loc = output_state.copy();  
   
    // separate row indices of single and multiple occurances
    for (size_t i = 0; i < output_state_loc.size(); i++) {
        if (output_state_loc[i] == 1) {
            row_indices.push_back(i);
        }
        else if (output_state_loc[i] > 1) {
            mrows.push_back(i);
        }
    }
 
    //sort multiplicity >=2 row indices since we need anchor rows, and complexity reduction greatest by using smallest multiplicities
    int64_t* output_state_loc_data = output_state_loc.get_data();
    sort(mrows.begin(), mrows.end(), [&output_state_loc_data](size_t i, size_t j) { return output_state_loc_data[i] < output_state_loc_data[j]; }); 
    if (row_indices.size() < 1) { //Glynn anchor row, prevent streaming more than 256MB of data
        row_indices.push_back(mrows[0]);
        if (--output_state_loc[mrows[0]] == 1) {
          row_indices.push_back(mrows[0]);
          mrows.erase(mrows.begin());
        }
    }

    //construct multiplicity Gray code counters
    mulsum = 0;
    onerows = row_indices.size();
    
    
    // determine the number of smaller permanents
    totalPerms = 1;
    for (size_t i = 0; i < mrows.size(); i++) {
        row_indices.push_back(mrows[i]);
        curmp.push_back(output_state_loc[mrows[i]]);
        inp.push_back(output_state_loc[mrows[i]]);
        mulsum += output_state_loc[mrows[i]];
        totalPerms *= (output_state_loc[mrows[i]] + 1);
    }


    if (onerows < 1+dfe_basekernpow2) { //pass the calculation to CPU
        doCPU = true;
        return;
    }


}

/**
@brief ???????
*/
void
cGlynnPermanentCalculatorRepeatedMulti_DFE::prepareDataForRepeatedMulti_DFE()
{


    if (doCPU) {
        return;
    }

    

    // calulate the maximal sum of the columns to normalize the matrix
    matrix_base<Complex32> colSumMax( photons, 2);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
    //sum up vectors in first/upper-left and fourth/lower-right quadrants of the complex plane
    for (size_t i=0; i<row_indices.size(); i++) {
        for( size_t jdx=0; jdx<photons; jdx++) {
            for (int64_t idx = 0; idx < (i < onerows ? 1 : output_state_loc[row_indices[i]]); idx++) {
                size_t offset = row_indices[i]*matrix_init.stride + colIndices[jdx];
                int realPos = matrix_init[offset].real() > 0;
                int slopeUpLeft = realPos == (matrix_init[offset].imag() > 0);
                if (realPos) colSumMax[2*jdx+slopeUpLeft] += matrix_init[offset];
                else colSumMax[2*jdx+slopeUpLeft] -= matrix_init[offset];
            }
        }
    }

    //now try to add/subtract neighbor quadrant values to the prior sum vector to see if it increase the absolute value    
    for (size_t i=0; i<row_indices.size(); i++) {
        for( size_t jdx=0; jdx<photons; jdx++) {
            for (int64_t idx = 0; idx < (i < onerows ? 1 : output_state_loc[row_indices[i]]); idx++) {
                size_t offset = row_indices[i]*matrix_init.stride + colIndices[jdx];
                int realPos = matrix_init[offset].real() > 0;
                int slopeUpLeft = realPos == (matrix_init[offset].imag() > 0);
                Complex32 value1 = colSumMax[2*jdx+1-slopeUpLeft] + matrix_init[offset];
                Complex32 value2 = colSumMax[2*jdx+1-slopeUpLeft] - matrix_init[offset];
                colSumMax[2*jdx+1-slopeUpLeft] = std::norm(value1) > std::norm(value2) ? value1 : value2;
            }
        }
    } 

    // calculate the renormalization coefficients
    matrix_base<long double> renormalize_data(1, photons);
    for (size_t jdx=0; jdx<photons; jdx++ ) {
        renormalize_data[jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
        //printf("%d %.21Lf\n", jdx, renormalize_data[jdx]);
    }

    // renormalize the input matrix and convert to fixed point maximizing precision via long doubles
    // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
    const size_t rows = row_indices.size();
    //const size_t numinits = 4;
    const size_t max_fpga_cols = max_dim / numinits;
    const size_t actualinits = (photons + max_fpga_cols-1) / max_fpga_cols;
    matrix_base<ComplexFix16> mtxprefix[numinits];
    const long double fixpow = 1ULL << 62;
    //tbb::parallel_for( (size_t)0, actualinits, (size_t)1, [&](size_t kdx){
    for (size_t kdx = 0; kdx < numinits; kdx++) {
        mtxprefix[kdx] = matrix_base<ComplexFix16>(rows, max_fpga_cols);
        memset(mtxprefix[kdx].get_data(), 0.0, mtxprefix[kdx].size()*sizeof(ComplexFix16));
    }

    for (size_t kdx = 0; kdx < actualinits; kdx++) {
      size_t basecol = max_fpga_cols * kdx;
      size_t lastcol = photons<=basecol ? 0 : std::min(max_fpga_cols, photons-basecol);
      for (size_t idx=0; idx < rows; idx++) {
        size_t offset_small = idx*mtxprefix[kdx].stride;

        for (size_t jdx = 0; jdx < lastcol; jdx++) {
          size_t offset = row_indices[idx]*matrix_init.stride + colIndices[basecol+jdx];
          mtxprefix[kdx][offset_small+jdx].real = llrintl((long double)matrix_init[offset].real() * fixpow / renormalize_data[basecol+jdx]);
          mtxprefix[kdx][offset_small+jdx].imag = llrintl((long double)matrix_init[offset].imag() * fixpow / renormalize_data[basecol+jdx]);
          if (idx >= onerows) { //start with all positive Gray codes, so sum everything onto the adjust row
              for (int64_t j = 0; j < output_state_loc[row_indices[idx]]; j++) {
                  mtxprefix[kdx][jdx].real += mtxprefix[kdx][offset_small+jdx].real;
                  mtxprefix[kdx][jdx].imag += mtxprefix[kdx][offset_small+jdx].imag;
              }
          }
        }

        memset(&mtxprefix[kdx][offset_small+lastcol], 0.0, sizeof(ComplexFix16)*(max_fpga_cols-lastcol));
      }
      for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxprefix[kdx][jdx].real = fixpow;
    }
/*
    std::cout << "export mtxprefix" << std::endl;
    for (size_t kdx = 0; kdx < numinits; kdx++) {
        matrix_base<ComplexFix16>& mtxprefix_tmp = mtxprefix[kdx];
        long double factor = (long double)(1ULL<<62);
        long double tmp2 = 0.0L;
        ComplexFix16* data = mtxprefix_tmp.get_data();
        for(int i = 0; i < mtxprefix_tmp.size(); i++) {  
            ComplexFix16 tmp = data[i];
            tmp2 = tmp2 + (long double)tmp.real/factor + (long double)tmp.imag/factor;   
        }
        std::cout << tmp2 << std::endl;  
    }
*/      

  
    uint64_t gcodeidx = 0, cur_multiplicity = 1, skipidx = (1ULL << curmp.size())-1; //gcodeidx is direction bit vector, skipidx set to not skip all indexes - technically "not skip index"
    size_t bytesPerMatrix = onerows*max_fpga_cols*sizeof(uint64_t)*2;
    size_t maxmatrices = (1ULL << 28) / bytesPerMatrix;

    //matrix_base<ComplexFix16> mtxfix[numinits] = {};
    mtxfix = new matrix_base<ComplexFix16>[numinits];
    for (size_t i = 0; i < numinits; i++) {
        mtxfix[i] = matrix_base<ComplexFix16>(onerows * totalPerms, max_fpga_cols);
        memset(mtxfix[i].get_data(), 0.0, mtxfix[i].size()*sizeof(ComplexFix16));
    };

    //assert(mtxfix[i].stride == mtxfix[i].cols);
    renormalize_data_all = matrix_base<long double>(totalPerms, photons);
    for (size_t i = 0; i < totalPerms; i++) {
        memcpy(renormalize_data_all.get_data()+photons*i, renormalize_data.get_data(), photons * sizeof(long double));
    }

    while (true) {

        //size_t offset_small = mplicity.size()*onerows*max_fpga_cols;
        size_t offset_small = mplicity.size()*onerows*max_fpga_cols;
        for (size_t i = 0; i < actualinits; i++) {
            memcpy(mtxfix[i].get_data()+offset_small, mtxprefix[i].get_data(), sizeof(ComplexFix16) * onerows * max_fpga_cols);
        }
        mplicity.push_back(cur_multiplicity);   


        

        //if (skipidx == 0) { 
        if (skipidx == 0) { //all multiplicities completed   
  
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
 


    if (doCPU) { //compute with other method
      GlynnPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_init, input_state, output_state);
      unlock_lib();
      return perm;
    }



    //assert(mtxfix[i].stride == mtxfix[i].cols);


    if (totalPerms == 1) {
        matrix&& modifiedInterferometerMatrix = adaptInterferometerGlynnMultiplied(matrix_init, &input_state, &output_state );
        GlynnPermanentCalculator_DFE(modifiedInterferometerMatrix, perm, useDual, useFloat);
        unlock_lib();
        return perm;
    }

    lock_lib();
    int useFloat = 0; 

    //const size_t numinits = 4;
    //matrix_base<ComplexFix16> mtxfix[numinits] = {};

    size_t permBase = 0;
    std::vector<Complex16> perms;
    perms.resize(totalPerms);



    ComplexFix16** mtx_fix_data = new ComplexFix16*[numinits];
    for (size_t i = 0; i < numinits; i++) {
        mtx_fix_data[i] = mtxfix[i].get_data();
    }

    if (useFloat) {
        calcPermanentGlynnDFEF( (const ComplexFix16**)mtx_fix_data, renormalize_data_all.get_data(), onerows, photons, totalPerms, perms.data()+permBase);
    }
    else {
        calcPermanentGlynnDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data_all.get_data(), onerows, photons, totalPerms, perms.data()+permBase);
    }

    delete[](mtx_fix_data);

                                         


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

    return perm;


}



/**
@brief ???????
*/
void 
cGlynnPermanentCalculatorRepeatedMulti_DFE::reset() {

    
    doCPU = false;

    if (mtxfix) {
        for (int idx=0; idx<numinits; idx++ ) {
            mtxfix[idx] = matrix_base<ComplexFix16>(0,0);
        }
       
        delete[] mtxfix;
        mtxfix = NULL;
    }



    renormalize_data_all = matrix_base<long double>(0,0);
    onerows = 0;
    photons = 0;
    mplicity.clear();
////
    mrows.clear();
    row_indices.clear();
    colIndices.clear();
    curmp.clear();
    inp.clear();
/////
    mulsum = 0;



}


/**
@brief ???????
*/
void
GlynnPermanentCalculatorRepeatedMulti_DFE(matrix& matrix_init, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual)
{

    lock_lib();
    int useFloat = 0;
    if (!useFloat) init_dfe_lib(DFE_MAIN, useDual);
    else if (useFloat) init_dfe_lib(DFE_FLOAT, useDual);  


    cGlynnPermanentCalculatorRepeatedMulti_DFE DFEcalculator(matrix_init, input_state, output_state, useDual );
    DFEcalculator.determineMultiplicitiesForRepeatedMulti_DFE();
    DFEcalculator.prepareDataForRepeatedMulti_DFE();
    perm = DFEcalculator.calculate();

    
    unlock_lib();
    return;


}


/**
@brief ???????
*/
matrix transpose_reorder_rows(matrix& matrix_mtx, std::vector<uint8_t> & rowchange_indices, int transpose)
{
    matrix matrix_rows(rowchange_indices.size(), transpose ? matrix_mtx.rows : matrix_mtx.cols);
    if (transpose) {
        for (size_t i = 0; i < matrix_mtx.rows; i++) {
            size_t offset = i*matrix_mtx.stride;
            for (size_t j = 0; j < rowchange_indices.size(); j++) {
                matrix_rows[j*matrix_rows.stride+i] = matrix_mtx[offset+rowchange_indices[j]];
            }
        }
    } else {
        for (size_t i = 0; i < rowchange_indices.size(); i++) {
            size_t offset = rowchange_indices[i]*matrix_mtx.stride;
            size_t newoffset = i*matrix_rows.stride;
            for (size_t j = 0; j < matrix_mtx.cols; j++) {
                matrix_rows[newoffset+j] = matrix_mtx[offset+j];
            }
        }
    }
    return matrix_rows;
}


void location_to_counter(std::vector<uint64_t>& count, std::vector<uint64_t>& inp, uint64_t loc)
{
    for (size_t i = 0; i < inp.size(); i++) {
        count.push_back(loc % inp[i]);
        loc /= inp[i]; 
    }
}
void counter_to_gcode(std::vector<uint64_t>& gcode, std::vector<uint64_t>& counterChain, std::vector<uint64_t>& inp)
{
    gcode = counterChain;
    int parity = 0;
    for (size_t j = inp.size()-1; j != ~0ULL; j--) {
        if (parity) gcode[j] += inp[j];
        if (((counterChain[j] & 1) != 0) && (((inp[j] & 1) != 0) || (counterChain[j] < inp[j])))
            parity = !parity;
    }
}
uint64_t divide_gray_code(std::vector<uint64_t>& inp, std::vector<uint64_t>& mplicity, std::vector<uint8_t>& initDirections, uint8_t loopLength)
{
    uint64_t total = 1;
    for (size_t i = 0; i < inp.size(); i++) { total *= inp[i]; }
    uint64_t segment = total / loopLength, rem = total % loopLength;
    uint64_t cursum = 0;
    initDirections.resize(loopLength * inp.size()); //for initDirections - * mulsum
    for (size_t i = 0; i < loopLength; i++) {
        std::vector<uint64_t> loc, gcode;
        location_to_counter(loc, inp, cursum);
        counter_to_gcode(gcode, loc, inp);
        uint64_t bincoeff = 1;
        //uint64_t k_base = 0;
        for (size_t j = 0; j < gcode.size(); j++) {
            bool curdir =  gcode[j] < inp[j];
            uint64_t curval = curdir ? inp[j]-1-gcode[j] : gcode[j]-inp[j];
            bincoeff *= binomialCoeff(inp[j], curval);
            initDirections[j*loopLength+i] = loc[j];
            /*int64_t curmp = (curval << 1) - inp[j];
            uint64_t k = 0;
            for (k = 0; k < inp[j]; k++) { //expand Gray code into a bit vector, staggered by loopLength
                initDirections[(k_base+k)*loopLength+i] = k < curval ? 1 : 0;
            }
            k_base += inp[j]; */
            //initDirections XORed together gives the starting parity, computed on DFE
        }
        mplicity.push_back(bincoeff);
        cursum += segment + ((i < rem) ? 1 : 0);
    }
    return total;
}

matrix input_to_bincoeff_indices(matrix& matrix_mtx, PicState_int64& input_state, int useDual, std::vector<uint8_t> & rowchange_indices, std::vector<uint64_t> & mplicity, std::vector<uint8_t>& initDirections, uint8_t & onerows, uint64_t & changecount, uint8_t & mulsum, int transpose, int loopLength)
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
  while (row_indices.size() > 1+dfe_basekernpow2) { //for binomial coefficients to work we must fix onerows to one tick on the kernel per the Gray code fixed rows
      mrows.push_back(row_indices.back());
      row_indices.erase(row_indices.end()-1);
  } 
  onerows = row_indices.size(), mulsum = 0, changecount = 0;
  std::vector<uint64_t> curmp, inp;
  for (size_t i = 0; i < mrows.size(); i++) {
    row_indices.push_back(mrows[i]);
    //curmp.push_back(0); //curmp.push_back(input_state[mrows[i]]);
    inp.push_back(input_state[mrows[i]]+1);
    mulsum += input_state[mrows[i]];
  }
  matrix matrix_rows = transpose_reorder_rows(matrix_mtx, row_indices, transpose);
  for (size_t i = 0; i < row_indices.size(); i++) {
      rowchange_indices.push_back(i < onerows ? 1 : input_state[row_indices[i]]); 
      //for (size_t j = i < onerows ? 1 : input_state[row_indices[i]]; j != 0; j--) {
      //  rowchange_indices.push_back(i);
      //}
  }
  if (mrows.size() == 0) { mplicity.push_back(1); return matrix_rows; }
  changecount = divide_gray_code(inp, mplicity, initDirections, loopLength) - 1;
  return matrix_rows;
  /*std::vector<uint8_t> k; k.resize(inp.size());
  uint64_t cur_multiplicity = 1;
  while (true) {
      mplicity.push_back(cur_multiplicity);
      size_t j = 0;
      for (size_t i = 0; i < curmp.size(); i++) {
          if (curmp[i] == inp[i]-1) { curmp[i] = 0; j++; }
          else { curmp[i]++; break; }
      }
      if (j == inp.size()) {
          return matrix_rows;
      }
      bool curdir =  k[j] < inp[j];
      cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[j]-1, curdir ? inp[j]-1-k[j] : k[j]-inp[j]);
      //rowchange_indices.push_back((onerows+j) | (curdir ? 0x80 : 0)); //high bit indicates subtraction
      changecount++;
      for (size_t i = 0; i <= j; i++)
          k[i] = (k[i] != (inp[i] << 1)-1) ? k[i] + 1 : 0;
  }*/
  /*
  uint64_t gcodeidx = 0, cur_multiplicity = 1, skipidx = (1ULL << curmp.size())-1;
  while (true) {
    mplicity.push_back(cur_multiplicity);
    if (skipidx == 0) {
        return matrix_rows;
    }
    size_t i = __builtin_ctzll(skipidx);
    bool curdir = (gcodeidx & (1ULL << i)) == 0;
    cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[i], (curmp[i] + inp[i]) / 2);
    curmp[i] = curdir ? curmp[i] - 2 : curmp[i] + 2;
    rowchange_indices.push_back((onerows+i) | (curdir ? 0x80 : 0)); //high bit indicates subtraction
    changecount++;
    if ((!curdir && curmp[i] == inp[i]) || (curdir && curmp[i] == -inp[i])) skipidx ^= ((1ULL << (i+1)) - 1);
    else skipidx ^= ((1ULL << i) - 1);
    gcodeidx ^= (1ULL << i) - 1;
  }*/
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
    uint64_t t1 = 1, t2 = 1;   
    for (size_t i = 0; i < input_state.size(); i++) {
        photons += input_state[i];
        t1 *= (input_state[i]+1); t2 *= (output_state[i]+1);
    }
    if (!calcPermanentGlynnRepDFE || photons < 1+dfe_basekernpow2) { //compute with other method
      GlynnPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_init, input_state, output_state);
      unlock_lib();
      return;
    }
    int transpose = t1 < t2; //transpose if needed to reduce complexity on rows direction
    std::vector<uint8_t> rowchange_indices;
    std::vector<uint64_t> mplicity;
    std::vector<uint8_t> initDirections;
    uint8_t onerows, mulsum; uint64_t changecount;
    PicState_int64 adj_input_state = transpose ? input_state.copy() : output_state.copy();
    int loopLength = 20;
    matrix matrix_mtx = input_to_bincoeff_indices(matrix_init, adj_input_state, useDual, rowchange_indices, mplicity, initDirections, onerows, changecount, mulsum, transpose, loopLength); 
    
    // calulate the maximal sum of the columns to normalize the matrix
    matrix_base<Complex32> colSumMax( matrix_mtx.cols, 2);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
    std::vector<std::vector<uint8_t>> sortedSlopes(matrix_mtx.cols);
    for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
        sortedSlopes[jdx] = rowchange_indices;         
        sort(sortedSlopes[jdx].begin(), sortedSlopes[jdx].end(), [&matrix_mtx, jdx](uint8_t a, uint8_t b){
            return matrix_mtx[a*matrix_mtx.stride+jdx].real()*matrix_mtx[b*matrix_mtx.stride+jdx].imag() <
                    matrix_mtx[b*matrix_mtx.stride+jdx].real()*matrix_mtx[a*matrix_mtx.stride+jdx].imag();
        }); //real1/imag1 < real2/imag2 -> real1*imag2<real2*imag1, also std::arg but more expensive
    }
    //sum up vectors in first/upper-left and fourth/lower-right quadrants
    for (size_t i=0; i<photons; i++) {
        for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
            size_t offset = rowchange_indices[i]*matrix_mtx.stride + jdx;
            int realPos = matrix_mtx[offset].real() > 0;
            int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
            if (realPos) colSumMax[2*jdx+slopeUpLeft] += matrix_mtx[offset];
            else colSumMax[2*jdx+slopeUpLeft] -= matrix_mtx[offset];
        }
    }
    //now try to add/subtract neighbor quadrant values to the prior sum vector to see if it increase the absolute value 
    for (size_t i=0; i<photons; i++) {
        for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
            size_t offset = rowchange_indices[i]*matrix_mtx.stride + jdx;
            int realPos = matrix_mtx[offset].real() > 0;
            int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
            Complex32 value1 = colSumMax[2*jdx+1-slopeUpLeft] + matrix_mtx[offset];
            Complex32 value2 = colSumMax[2*jdx+1-slopeUpLeft] - matrix_mtx[offset];
            colSumMax[2*jdx+1-slopeUpLeft] = std::norm(value1) > std::norm(value2) ? value1 : value2;
        } 
    }       

    // calculate the renormalization coefficients
    matrix_base<long double> renormalize_data(1, matrix_mtx.cols);
    for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
        //printf("%d %.21Lf %f\n", jdx, renormalize_data[jdx]);
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
      for (size_t j = transpose ? output_state[i] : input_state[i]; j != 0; j--) {
        colIndices.push_back(i);
      }
    }
    for (size_t i = (colIndices.size() % 16 == 0) ? 0 : (16 - colIndices.size() % 16); i != 0; i--) colIndices.push_back(0); //round up to nearest 16 bytes to allow streaming
    for (size_t i = (rowchange_indices.size() % 16 == 0) ? 0 : (16 - rowchange_indices.size() % 16); i != 0; i--) rowchange_indices.push_back(0); //round up to nearest 16 bytes to allow streaming
    for (size_t i = (renormalize_data.size() % 2 == 0) ? 0 : 1; i != 0; i--) mplicity.push_back(0); //round up to nearest 16 bytes to allow streaming
    for (size_t i = (initDirections.size() % 16 == 0) ? 0 : (16 - initDirections.size() % 16); i != 0; i--) initDirections.push_back(0); //round up to nearest 16 bytes to allow streaming
    calcPermanentGlynnRepDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, colIndices.data(),
      rowchange_indices.data(), initDirections.data(), photons, onerows, mplicity.data(), changecount, mulsum, &perm);

    unlock_lib();
    return;
}

void
GlynnPermanentCalculatorRepeatedBatch_DFE(matrix& matrix_init, std::vector<PicState_int64>& input_states,
    std::vector<std::vector<PicState_int64>>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual)
{
    for (size_t i = 0; i < input_states.size(); i++) {
        perm[i].resize(output_states[i].size());
        for (size_t j = 0; j < output_states[i].size(); j++) {
            GlynnPermanentCalculatorRepeated_DFE(matrix_init, input_states[i], output_states[i][j], perm[i][j], useDual); 
        }
    }
}


}
