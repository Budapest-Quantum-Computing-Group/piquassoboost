#include "GlynnPermanentCalculatorRepeatedDFE.h"
#include "GlynnPermanentCalculatorRepeated.h"
#include "BBFGPermanentCalculatorRepeated.h"
#include "BatchedPermanentCalculator.h"
#include <fstream>
#include <iostream>

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif
#include <vector>
#include "common_functionalities.h"

typedef void(*CALCPERMGLYNNREPDFE)(const pic::ComplexFix16**, const long double*, const uint64_t, const uint64_t, const unsigned char*,
  const uint8_t*, const uint8_t*, const uint8_t, const uint8_t, const uint64_t*, const uint64_t, const uint8_t, const int, const uint64_t, pic::Complex16*);
typedef int(*INITPERMGLYNNREPDFE)(int, size_t*, size_t*);
typedef void(*FREEPERMGLYNNREPDFE)(void);

CALCPERMGLYNNREPDFE calcPermanentGlynnRepDFE = NULL;
INITPERMGLYNNREPDFE initializeRep_DFE = NULL;
FREEPERMGLYNNREPDFE releiveRep_DFE = NULL;

typedef void(*CALCPERMGLYNNDFE)(const pic::ComplexFix16**, const long double*, const uint64_t, const uint64_t, const uint64_t, pic::Complex16*);
extern "C" CALCPERMGLYNNDFE calcPermanentGlynnDFE;
size_t dfe_loop_length = 0;

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
inline long long doubleToLLRaw(double d)
{
    double* pd = &d;
    long long* pll = (long long*)pd;
    return *pll;
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


    //mtxfix = NULL;
    mtxfix_batched = NULL;
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
    //mtxfix = NULL;//new matrix_base<ComplexFix16>[numinits];
    mtxfix_batched = new matrix_base<ComplexFix16>[numinits];
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

    if (mtxfix_batched) {
        for (int idx=0; idx<numinits; idx++ ) {
            mtxfix_batched[idx] = matrix_base<ComplexFix16>(0,0);
        }
  
        delete[] mtxfix_batched;
        mtxfix_batched = NULL;  
    }

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


    input_states.resize( colIndices.size() );
    colIndices_batched.resize( colIndices.size() );

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
int 
cGlynnPermanentCalculatorRepeatedMulti_DFE::determineBatchIterations() {


    batch_iterations.clear();
    const int64_t memory_limit_max = 0.6e11; //60 GB
    int64_t memory_limit = 0.05e11; //5 GB
    const size_t max_fpga_cols = max_dim / numinits;
    
    int64_t total_memory_needed = numinits*onerows * totalPerms * colIndices.size()*max_fpga_cols*sizeof(ComplexFix16);
    int64_t batches_num = ceil( (double)total_memory_needed/memory_limit );

    while ( batches_num > colIndices.size() && memory_limit <  memory_limit_max ) {
        memory_limit += 0.05e11;
        batches_num = ceil( (double)total_memory_needed/memory_limit );
    }
  
    if ( batches_num > colIndices.size() ) {
        std::cout << "cGlynnPermanentCalculatorRepeatedMulti_DFE::determineIterations(): Not enough memory for further processing" << std::endl;
        return 1;
    }
    

    
    batch_iterations = PicVector<size_t>(batches_num, 0 );    
    size_t batch_num_loc = 0;

    while ( batch_num_loc < colIndices.size() ) {

        for (int64_t idx=0; idx<batches_num; idx++ ) {
        
            batch_iterations[idx]++;
            batch_num_loc++;
//std::cout << batch_num_loc << std::endl;
            if ( batch_num_loc == colIndices.size() ) {
                break;
            }

        }

    }

/*
    for (int64_t idx=0; idx<batches_num; idx++ ) {
        
        if (batches_num>1) {
            std::cout << "batch size: " << batch_iterations[idx] << std::endl;        
        }
    
    }
*/

    
    
    return 0;


}


/**
@brief ????????????
*/
int 
cGlynnPermanentCalculatorRepeatedMulti_DFE::reserveSpace( size_t jdx )  {

    const size_t max_fpga_cols = max_dim / numinits;
    
    size_t rows = onerows * totalPerms * batch_iterations[jdx];
/*    
    // test the size of the memory to be allocated
    if ( numinits*rows*max_fpga_cols*sizeof(ComplexFix16) > 4e9 ) {
        std::cout << double(numinits*rows*max_fpga_cols*sizeof(ComplexFix16))/1e9 << " GB needs to be allocated" << std::endl;
    }
*/    
    for (size_t i = 0; i < numinits; i++) {
        try {
            mtxfix_batched[i] = matrix_base<ComplexFix16>(0,0);
            mtxfix_batched[i] = matrix_base<ComplexFix16>(rows, max_fpga_cols);
        }
        catch (std::bad_alloc & ba)
        {
            std::cout << "failed to allocate  mtxfix_batched!" << std::endl;
            return 1;
        }
        memset(mtxfix_batched[i].get_data(), 0.0, mtxfix_batched[i].size()*sizeof(ComplexFix16));
    };    


    try {
        renormalize_data_batched = matrix_base<long double>(0, 0);
        renormalize_data_batched = matrix_base<long double>(totalPerms*batch_iterations[jdx], photons);
    }
    catch (std::bad_alloc & ba)
    {
        std::cout << "failed to allocate renormalize_data_batched!" << std::endl;
        return 2;
    }
    
    

    
    return 0;
}




/**
@brief ???????
*/
void 
cGlynnPermanentCalculatorRepeatedMulti_DFE::addInputState(size_t idx, PicState_int64& input_state_in ) {


    input_states[idx] = input_state_in;

}


/**
@brief ???????
*/
void
cGlynnPermanentCalculatorRepeatedMulti_DFE::initiateBatch(size_t idx) {

    // add a photon to the current output state
    PicState_int64&& input_state_loc = input_state.copy();
    input_state_loc[colIndices[idx]]--;
    input_state_loc.number_of_photons--;

    input_states[idx] = input_state_loc;

    colIndices_batched[idx] = colIndices;///////////////
    colIndices_batched[idx].erase( colIndices_batched[idx].begin() + idx ) ;/////////////

}


/**
@brief ???????
*/
void 
cGlynnPermanentCalculatorRepeatedMulti_DFE::determineMultiplicities() {


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
cGlynnPermanentCalculatorRepeatedMulti_DFE::determineNormalization()
{

    size_t photons_init = photons+1;

    // calulate the maximal sum of the columns to normalize the matrix
    matrix_base<Complex32> colSumMax( photons_init, 2);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
    //sum up vectors in first/upper-left and fourth/lower-right quadrants of the complex plane
    for (size_t i=0; i<row_indices.size(); i++) {

        for( size_t jdx=0; jdx<photons_init; jdx++) {

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

        for( size_t jdx=0; jdx<photons_init; jdx++) {

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
    renormalize_data = matrix_base<long double>(1, photons_init);
    for (size_t jdx=0; jdx<photons_init; jdx++ ) {
        renormalize_data[jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
        //printf("%d %.21Lf\n", jdx, renormalize_data[jdx]);
    }


}


/**
@brief ???????
*/
void
cGlynnPermanentCalculatorRepeatedMulti_DFE::prepareDataForRepeatedMulti_DFE(size_t insideBatch_idx, size_t batch_idx_offset)
{

    if (doCPU) {
        return;
    }

    size_t batch_idx = insideBatch_idx + batch_idx_offset;

    // construct array stroing the renormalization coefficients excluding the column determined by batch_idx
    matrix_base<long double> renormalize_data_loc(1, photons);
    if ( batch_idx>0) {
        memcpy( renormalize_data_loc.get_data(), renormalize_data.get_data(), batch_idx*sizeof(long double) );
    }
    if ( batch_idx<photons ) {
        memcpy( renormalize_data_loc.get_data()+batch_idx, renormalize_data.get_data()+batch_idx+1, (photons-batch_idx)*sizeof(long double) );
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
        memset(mtxprefix[kdx].get_data(), 0, mtxprefix[kdx].size()*sizeof(ComplexFix16));
    }

    for (size_t kdx = 0; kdx < actualinits; kdx++) {
      size_t basecol = max_fpga_cols * kdx;
      size_t lastcol = photons<=basecol ? 0 : std::min(max_fpga_cols, photons-basecol);
      for (size_t idx=0; idx < rows; idx++) {
        size_t offset_small = idx*mtxprefix[kdx].stride;

        for (size_t jdx = 0; jdx < lastcol; jdx++) {
          size_t offset = row_indices[idx]*matrix_init.stride + colIndices_batched[batch_idx][basecol+jdx];
          mtxprefix[kdx][offset_small+jdx].real = llrintl((long double)matrix_init[offset].real() * fixpow / renormalize_data_loc[basecol+jdx]);
          mtxprefix[kdx][offset_small+jdx].imag = llrintl((long double)matrix_init[offset].imag() * fixpow / renormalize_data_loc[basecol+jdx]);
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
    matrix_base<ComplexFix16> mtxfix[numinits];
    for (size_t i = 0; i < numinits; i++) {
/*
        mtxfix[i] = matrix_base<ComplexFix16>(onerows * totalPerms, max_fpga_cols);
        memset(mtxfix[i].get_data(), 0.0, mtxfix[i].size()*sizeof(ComplexFix16));
*/
        mtxfix[i] = matrix_base<ComplexFix16>(mtxfix_batched[i].get_data() + insideBatch_idx*onerows*totalPerms*max_fpga_cols, onerows * totalPerms, max_fpga_cols, mtxfix_batched[i].stride);
    };

    //assert(mtxfix[i].stride == mtxfix[i].cols);
    matrix_base<long double> renormalize_data_all(renormalize_data_batched.get_data()+insideBatch_idx*totalPerms*photons, totalPerms, photons, renormalize_data_batched.stride);
    for (size_t i = 0; i < totalPerms; i++) {
        memcpy(renormalize_data_all.get_data()+photons*i, renormalize_data_loc.get_data(), photons * sizeof(long double));
    }

    std::vector<uint64_t> curmp_loc = curmp;
    size_t perm_idx = 0;

    while (true) {

        size_t offset_small = perm_idx*onerows*max_fpga_cols;
        for (size_t i = 0; i < actualinits; i++) {
            memcpy(mtxfix[i].get_data()+offset_small, mtxprefix[i].get_data(), sizeof(ComplexFix16) * onerows * max_fpga_cols);
        }

        if ( insideBatch_idx==0 ) {
            mplicity.push_back(cur_multiplicity);   
        }


        

        //if (skipidx == 0) { 
        if (skipidx == 0) { //all multiplicities completed   
  
            return;

        }
        
      
        size_t i = __builtin_ctzll(skipidx); //count of trailing zeros to compute next change index
        bool curdir = (gcodeidx & (1ULL << i)) == 0;

        if ( insideBatch_idx==0 ) {
            cur_multiplicity = binomial_gcode(cur_multiplicity, curdir, inp[i], (curmp_loc[i] + inp[i]) / 2);
        }

        curmp_loc[i] = curdir ? curmp_loc[i] - 2 : curmp_loc[i] + 2;
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

        if ((!curdir && curmp_loc[i] == inp[i]) || (curdir && curmp_loc[i] == -inp[i])) skipidx ^= ((1ULL << (i+1)) - 1); //set all skipping before and including current index
        else skipidx ^= ((1ULL << i) - 1); //flip all skipping which come before current index
        gcodeidx ^= (1ULL << i) - 1; //flip all directions which come before current index

        perm_idx++;
    }



}




/**
@brief ???????
*/
matrix
cGlynnPermanentCalculatorRepeatedMulti_DFE::calculate(size_t batch_idx, size_t batch_index_offset)
{

    matrix perms(1, batch_iterations[batch_idx]);

    if (doCPU) { //compute with other method
        GlynnPermanentCalculatorRepeatedLongDouble gpc;
        for ( size_t idx=0; idx<batch_iterations[batch_idx]; idx++) {
            perms[idx] = gpc.calculate(matrix_init, input_states[idx+batch_index_offset], output_state);
        }
        unlock_lib();
        return perms;
    }



    //assert(mtxfix[i].stride == mtxfix[i].cols);

/*
    if (totalPerms == 1) {
        matrix&& modifiedInterferometerMatrix = adaptInterferometerGlynnMultiplied(matrix_init, &input_state, &output_state );
        GlynnPermanentCalculator_DFE(modifiedInterferometerMatrix, perm, useDual, useFloat);
        unlock_lib();
        return perm;
    }
*/
    lock_lib();
    int useFloat = 0; 

    //const size_t numinits = 4;
    //matrix_base<ComplexFix16> mtxfix[numinits] = {};

    const size_t max_fpga_cols = max_dim / numinits;


    matrix perms_DFE(1, totalPerms*batch_iterations[batch_idx]);


    ComplexFix16** mtx_fix_data = new ComplexFix16*[numinits];
    for (size_t i = 0; i < numinits; i++) {
        mtx_fix_data[i] = mtxfix_batched[i].get_data();
    }


    if (useFloat) {
        calcPermanentGlynnDFEF( (const ComplexFix16**)mtx_fix_data, renormalize_data_batched.get_data(), onerows, photons, totalPerms*batch_iterations[batch_idx], perms_DFE.get_data());
    }
    else {
        calcPermanentGlynnDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data_batched.get_data(), onerows, photons, totalPerms*batch_iterations[batch_idx], perms_DFE.get_data());
    }

    delete[](mtx_fix_data);

                          
    for (size_t idx=0; idx<batch_iterations[batch_idx]; idx++) {
        Complex32 res_tmp(0.0,0.0);

        // select submatrix of perms_DFE to sum up to a single permanent
        matrix perm_batch(perms_DFE.get_data()+idx*totalPerms, 1, totalPerms);

        for (size_t i = 0; i < perm_batch.size(); i++) {
            if (i & 1) {
                res_tmp -= ToComplex32(perm_batch[i]) * (long double)mplicity[i]; 
            }
            else {
                res_tmp += ToComplex32(perm_batch[i]) * (long double)mplicity[i];
            }
        }
        Complex16 perm = ToComplex16(res_tmp / (long double)(1ULL << mulsum)); //2**mulsum is the effective number of permanents or sum of all multiplicities
        perms[idx] = perm;

    }

    unlock_lib();

    return perms;


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
    size_t photons = 0;
    uint64_t t1 = 1, t2 = 1;
    for (size_t i = 0; i < input_state.size(); i++) {
        photons += input_state[i];
        t1 *= (input_state[i]+1);
        if (i < output_state.size()) t2 *= (output_state[i]+1);
    }
    if (!calcPermanentGlynnDFE ||
        photons < 1+dfe_basekernpow2) { //compute with other method
      BBFGPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_init, input_state, output_state, true);
      unlock_lib();
      return;
    }
    int transpose = t1 < t2; //transpose if needed to reduce complexity on rows direction
    const size_t max_dim = dfe_mtx_size;
    //convert multiplicities of rows and columns to indices
    std::vector<unsigned char> colIndices; colIndices.reserve(max_dim);
    for (size_t i = 0; i < (transpose ? output_state.size() : input_state.size()); i++) {
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
    //while (row_indices.size() < 1+dfe_basekernpow2) { //Glynn anchor row, plus 2/3 anchor rows needed for binary Gray code in kernel
    while (row_indices.size() < 1) { //Glynn anchor row, prevent streaming more than 256MB of data
        row_indices.push_back(mrows[0]);
        if (--adj_input_state[mrows[0]] == 1) {
          row_indices.push_back(mrows[0]);
          mrows.erase(mrows.begin());
        }
    }
    //construct multiplicity Gray code counters
    uint8_t mulsum = 0, onerows = row_indices.size();
    std::vector<uint64_t> curmp, inp;
    uint64_t totalPerms = 1;
    for (size_t i = 0; i < mrows.size(); i++) {
        //for (size_t j = 0; j < adj_input_state[mrows[i]]; j++)
        row_indices.push_back(mrows[i]);
        curmp.push_back(adj_input_state[mrows[i]]);
        inp.push_back(adj_input_state[mrows[i]]);
        mulsum += adj_input_state[mrows[i]];
        totalPerms *= (adj_input_state[mrows[i]] + 1);
    }
    if (onerows < 1+dfe_basekernpow2) { //compute with other method
      BBFGPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_init, input_state, output_state, true);
      unlock_lib();
      return;
    }
    if (mrows.size() == 0) {
        matrix matrix_rows = transpose_reorder_rows_cols(matrix_init, row_indices, colIndices, transpose);
        GlynnPermanentCalculator_DFE(matrix_rows, perm, useDual, useFloat);
        unlock_lib();
        return;
    }

    // calulate the maximal sum of the columns to normalize the matrix
    matrix_base<Complex32> colSumMax( photons, 2);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
    //sum up vectors in first/upper-left and fourth/lower-right quadrants
    for (size_t i=0; i<row_indices.size(); i++) {
        //size_t offset = (transpose ? colIndices[i] : row_indices[i]) * matrix_init.stride;
        for( size_t jdx=0; jdx<photons; jdx++) {
            for (int64_t idx = 0; idx < (i < onerows ? 1 : adj_input_state[row_indices[i]]); idx++) {
                size_t offset = transpose ? colIndices[jdx]*matrix_init.stride+row_indices[i] : row_indices[i]*matrix_init.stride+colIndices[jdx];
                int realPos = matrix_init[offset].real() > 0;
                int slopeUpLeft = realPos == (matrix_init[offset].imag() > 0);
                if (realPos) colSumMax[2*jdx+slopeUpLeft] += matrix_init[offset];
                else colSumMax[2*jdx+slopeUpLeft] -= matrix_init[offset];
            }
        }
    }
    //now try to add/subtract neighbor quadrant values to the prior sum vector to see if it increase the absolute value    
    for (size_t i=0; i<row_indices.size(); i++) {
        //size_t offset = (transpose ? colIndices[i] : row_indices[i]) * matrix_init.stride;
        for( size_t jdx=0; jdx<photons; jdx++) {
            for (int64_t idx = 0; idx < (i < onerows ? 1 : adj_input_state[row_indices[i]]); idx++) {
                size_t offset = transpose ? colIndices[jdx]*matrix_init.stride+row_indices[i] : row_indices[i]*matrix_init.stride+colIndices[jdx];
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
          mtxprefix[i][offset_small+jdx].real = llrintl((long double)matrix_init[offset].real() * fixpow / renormalize_data[basecol+jdx]);
          mtxprefix[i][offset_small+jdx].imag = llrintl((long double)matrix_init[offset].imag() * fixpow / renormalize_data[basecol+jdx]);
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
    
    size_t bytesPerMatrix = onerows*max_fpga_cols*sizeof(uint64_t)*2;
    size_t maxmatrices = (1ULL << 28) / bytesPerMatrix;
    matrix_base<ComplexFix16> mtxfix[numinits] = {};
    for (size_t i = 0; i < actualinits; i++)
        mtxfix[i] = matrix_base<ComplexFix16>(onerows * maxmatrices, max_fpga_cols);
  
    Complex32 res(0.0, 0.0);
    uint64_t gcodeidx = 0, cur_multiplicity = 1, skipidx = (1ULL << curmp.size())-1; //gcodeidx is direction bit vector, skipidx set to not skip all indexes - technically "not skip index"
    std::vector<uint64_t> mplicity; //binomial coefficient multipliers
    size_t permBase = 0;
    ComplexFix16* mtx_fix_data[numinits];
    matrix_base<long double> renormalize_data_all(maxmatrices, photons);            
    for (size_t i = 0; i < maxmatrices; i++) memcpy(renormalize_data_all.get_data()+photons*i, renormalize_data.get_data(), photons * sizeof(long double));
    for (size_t j = 0; j < numinits; j++) mtx_fix_data[j] = mtxfix[j].get_data();
    std::vector<Complex16> perms;
    perms.resize(totalPerms);
    while (true) {
        size_t offset_small = (mplicity.size()-permBase)*onerows*max_fpga_cols;
        for (size_t i = 0; i < actualinits; i++) {
            memcpy(mtxfix[i].get_data()+offset_small, mtxprefix[i].get_data(), sizeof(ComplexFix16) * onerows * max_fpga_cols);
        }
        mplicity.push_back(cur_multiplicity);
        if (skipidx == 0 || (mplicity.size() - permBase) == maxmatrices) { //all multiplicities completed
            //GlynnPermanentCalculatorBatch_DFE(matrices, perms, useDual, false);
            //note: stride must equal number of columns, or this will not work as the C call expects contiguous data
            size_t numPerms = std::min(maxmatrices, totalPerms - permBase);
            //assert(mtxfix[i].stride == mtxfix[i].cols);
            calcPermanentGlynnDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data_all.get_data(), onerows, photons, numPerms, perms.data()+permBase);
            if (skipidx == 0) {
                for (size_t i = 0; i < perms.size(); i++) {
                    if (i & 1) res -= ToComplex32(perms[i]) * (long double)mplicity[i]; 
                    else res += ToComplex32(perms[i]) * (long double)mplicity[i];
                }
                perm = ToComplex16(res / (long double)(1ULL << mulsum)); //2**mulsum is the effective number of permanents or sum of all multiplicities
                unlock_lib();
                return;
            }
            permBase += numPerms;
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

void
GlynnPermanentCalculatorRepeatedMultiInputBatch_DFE(matrix& matrix_init, std::vector<std::vector<PicState_int64>>& input_states,
    std::vector<PicState_int64>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual)
{
    for (size_t i = 0; i < output_states.size(); i++) {
        perm[i].resize(input_states[i].size());
        for (size_t j = 0; j < input_states[i].size(); j++) {
            GlynnPermanentCalculatorRepeatedMulti_DFE(matrix_init, input_states[i][j], output_states[i], perm[i][j], useDual); 
        }
    }
}

void
GlynnPermanentCalculatorRepeatedMultiOutputBatch_DFE(matrix& matrix_init, std::vector<PicState_int64>& input_states,
    std::vector<std::vector<PicState_int64>>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual)
{
    for (size_t i = 0; i < input_states.size(); i++) {
        perm[i].resize(output_states[i].size());
        for (size_t j = 0; j < output_states[i].size(); j++) {
            GlynnPermanentCalculatorRepeatedMulti_DFE(matrix_init, input_states[i], output_states[i][j], perm[i][j], useDual); 
        }
    }
}


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
void counter_to_gcode(std::vector<uint64_t>& gcode, const std::vector<uint64_t>& counterChain, const std::vector<uint64_t>& inp)
{
    gcode.resize(inp.size());
    int parity = 0;
    for (size_t j = inp.size()-1; j != ~0ULL; j--) {
        gcode[j] = counterChain[j] + (parity ? inp[j] : 0);
        parity = gcode[j] & 1;
        //parity ^= (parity ? inp[j] - 1 - counterChain[j] : counterChain[j]) & 1; //ultimately mathematically equivalent...  
    }
}
uint64_t divide_gray_code(std::vector<uint64_t>& inp, std::vector<uint64_t>& mplicity, std::vector<uint8_t>& initDirections, int& initParities, uint8_t loopLength)
{
    uint64_t total = 1;
    for (size_t i = 0; i < inp.size(); i++) { total *= inp[i]; }
    uint64_t segment = total / loopLength, rem = total % loopLength;
    uint64_t cursum = 0;
    if (total < loopLength && inp[inp.size()-1]==2) initDirections.resize(loopLength * (inp.size()-1) + total); 
    else initDirections.resize(loopLength * inp.size()); //for initDirections - * mulsum
    initParities = 0;    
    for (size_t i = 0; i < loopLength; i++) {
        if ((cursum & 1) != 0) initParities |= 1 << i; 
        std::vector<uint64_t> loc, gcode;
        location_to_counter(loc, inp, cursum);
        counter_to_gcode(gcode, loc, inp);
        uint64_t bincoeff = 1;
        //uint64_t k_base = 0;
        for (size_t j = 0; j < gcode.size(); j++) {
            bool curdir =  gcode[j] < inp[j];
            uint64_t curval = curdir ? inp[j]-1-gcode[j] : gcode[j]-inp[j];
            bincoeff *= binomialCoeffInt64(inp[j]-1, curval);
            //add the initial parity as the high bit of the byte - because counter_to_gcode computes it backwards, it would create unnecessary logic in the kernel when we have 2 free bits wasted regardless for the 6-bit counters
            if (i < total || j != gcode.size()-1 || inp[j] != 2) 
                initDirections[j*loopLength+i] = loc[j] | (curdir ? 0x80 : 0);
            /*int64_t curmp = (curval << 1) - inp[j];
            uint64_t k = 0;
            for (k = 0; k < inp[j]; k++) { //expand Gray code into a bit vector, staggered by loopLength
                initDirections[(k_base+k)*loopLength+i] = k >= curval ? 1 : 0;
            }
            k_base += inp[j]; */
            //initDirections XORed together gives the starting parity, computed on DFE
        }
        if (i < total) mplicity.push_back(bincoeff);
        cursum += segment + ((i < rem) ? 1 : 0);
    }
    return total;
}

matrix input_to_bincoeff_indices(matrix& matrix_mtx, PicState_int64& input_state, int useDual, std::vector<uint8_t> & rowchange_indices, std::vector<uint64_t> & mplicity, std::vector<uint8_t>& initDirections, uint8_t & onerows, uint64_t & changecount, uint8_t & mulsum, int& initParities, int transpose, int loopLength)
{
  std::vector<uint8_t> mrows;
  std::vector<uint8_t> row_indices;
  for (size_t i = 0; i < input_state.size(); i++) {
    //for binomial coefficients to work we must fix onerows to one tick on the kernel per the Gray code fixed rows, so exactly onerows == 1+dfe_basekernpow2
    if (input_state[i] == 1 && row_indices.size() < 1+dfe_basekernpow2) row_indices.push_back(i);
    else if (input_state[i] != 0) mrows.push_back(i);
  }
  sort(mrows.begin(), mrows.end(), [&input_state](size_t i, size_t j) { return input_state[i] < input_state[j]; }); 
  while (row_indices.size() < 1+dfe_basekernpow2) { //Glynn anchor row, plus 2/3 anchor rows needed for binary Gray code in kernel
    row_indices.push_back(mrows[0]);
    if (--input_state[mrows[0]] == 1 && row_indices.size() < 1+dfe_basekernpow2) {
      row_indices.push_back(mrows[0]);
      mrows.erase(mrows.begin());
    }
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
  if (mrows.size() == 0) { initParities = 0; mplicity.push_back(1); return matrix_rows; }
  changecount = divide_gray_code(inp, mplicity, initDirections, initParities, loopLength) - 1;
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

bool colMux = false;

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculatorRepeated_DFE(matrix& matrix_init, PicState_int64& input_state,
    PicState_int64& output_state, Complex16& perm, int useDual, int useFloat)
{
    lock_lib();
    if (!useFloat) init_dfe_lib(DFE_REP, useDual);
    else init_dfe_lib(DFE_REP_FLOAT, useDual);    
    size_t photons = 0;
    uint64_t t1 = 1, t2 = 1;   
    for (size_t i = 0; i < input_state.size(); i++) {
        photons += input_state[i];
        t1 *= (input_state[i]+1);
        if (i < output_state.size()) t2 *= (output_state[i]+1);
    }
    if (!calcPermanentGlynnRepDFE || photons < 1+dfe_basekernpow2) { //compute with other method
      BBFGPermanentCalculatorRepeated gpc;
      perm = gpc.calculate(matrix_init, input_state, output_state, true);
      unlock_lib();
      return;
    }
    int transpose = t1 < t2; //transpose if needed to reduce complexity on rows direction
    std::vector<uint8_t> rowchange_indices;
    std::vector<uint64_t> mplicity;
    std::vector<uint8_t> initDirections;
    uint8_t onerows, mulsum; uint64_t changecount;
    PicState_int64 adj_input_state = transpose ? input_state.copy() : output_state.copy();
    int initParities;
    matrix matrix_mtx = input_to_bincoeff_indices(matrix_init, adj_input_state, useDual, rowchange_indices, mplicity, initDirections, onerows, changecount, mulsum, initParities, transpose, dfe_loop_length); 
    
    matrix_base<long double> renormalize_data(1, matrix_mtx.cols);
    if (!useFloat) {
        // calulate the maximal sum of the columns to normalize the matrix
        matrix_base<Complex32> colSumMax( matrix_mtx.cols, 2);
        memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
        //sum up vectors in first/upper-left and fourth/lower-right quadrants
        for (size_t idx = 0; idx < matrix_mtx.rows; idx++) {
            for (size_t i=0; i<rowchange_indices[idx]; i++) {
                for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
                    size_t offset = idx*matrix_mtx.stride + jdx;
                    int realPos = matrix_mtx[offset].real() > 0;
                    int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
                    if (realPos) colSumMax[2*jdx+slopeUpLeft] += matrix_mtx[offset];
                    else colSumMax[2*jdx+slopeUpLeft] -= matrix_mtx[offset];
                }
            }
        }
        //now try to add/subtract neighbor quadrant values to the prior sum vector to see if it increase the absolute value 
        for (size_t idx = 0; idx < matrix_mtx.rows; idx++) {
            for (size_t i=0; i<rowchange_indices[idx]; i++) {
                for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
                    size_t offset = idx*matrix_mtx.stride + jdx;
                    int realPos = matrix_mtx[offset].real() > 0;
                    int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
                    Complex32 value1 = colSumMax[2*jdx+1-slopeUpLeft] + matrix_mtx[offset];
                    Complex32 value2 = colSumMax[2*jdx+1-slopeUpLeft] - matrix_mtx[offset];
                    colSumMax[2*jdx+1-slopeUpLeft] = std::norm(value1) > std::norm(value2) ? value1 : value2;
                }
            } 
        }       
    
        // calculate the renormalization coefficients
        for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
            renormalize_data[jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
            //printf("%d %.21Lf %f\n", jdx, renormalize_data[jdx]);
        }
    }

    std::vector<unsigned char> colIndices; colIndices.reserve(photons);
    for (size_t i = 0; i < (transpose ? output_state.size() : input_state.size()); i++) {
      for (size_t j = transpose ? output_state[i] : input_state[i]; j != 0; j--) {
        colIndices.push_back(i);
      }
    }

    // renormalize the input matrix and convert to fixed point maximizing precision via long doubles
    // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
    const size_t max_dim = dfe_mtx_size;
    const size_t rows = matrix_mtx.rows;
    const size_t numinits = 4;
    const size_t max_fpga_cols = max_dim / numinits;
    const size_t cols = colMux ? matrix_mtx.cols : photons;
    const size_t actualinits = (cols + max_fpga_cols-1) / max_fpga_cols;
    matrix_base<ComplexFix16> mtxfix[numinits] = {};
    const long double fixpow = 1ULL << 62;
    const double fOne = doubleToLLRaw(1.0);
    int adjLoopLength = changecount+1 < (unsigned)dfe_loop_length && rowchange_indices[rows - 1] == 1 ? changecount+1 : dfe_loop_length;
    for (size_t i = 0; i < actualinits; i++) {
      mtxfix[i] = matrix_base<ComplexFix16>((rows-1)*dfe_loop_length+adjLoopLength, max_fpga_cols+1); //one extra for row multiplicities, initial directions and Gray codes, binomial coefficients
      size_t basecol = max_fpga_cols * i;
      size_t lastcol = cols<=basecol ? 0 : std::min(max_fpga_cols, cols-basecol);
      for (size_t idx=0; idx < rows; idx++) {
        size_t offset = idx * matrix_mtx.stride;
        size_t offset_small = idx*dfe_loop_length*mtxfix[i].stride;
        for (size_t jdx = 0; jdx < lastcol; jdx++) {
          mtxfix[i][offset_small+jdx].real = useFloat ? doubleToLLRaw(matrix_mtx[offset+(colMux ? basecol+jdx : colIndices[basecol+jdx])].real()) : llrintl((long double)matrix_mtx[offset+(colMux ? basecol+jdx : colIndices[basecol+jdx])].real() * fixpow / renormalize_data[colMux ? basecol+jdx : colIndices[basecol+jdx]]);
          mtxfix[i][offset_small+jdx].imag = useFloat ? doubleToLLRaw(matrix_mtx[offset+(colMux ? basecol+jdx : colIndices[basecol+jdx])].imag()) : llrintl((long double)matrix_mtx[offset+(colMux ? basecol+jdx : colIndices[basecol+jdx])].imag() * fixpow / renormalize_data[colMux ? basecol+jdx : colIndices[basecol+jdx]]);
          //printf("%d %d %d %llX %llX\n", i, idx, jdx, mtxfix[i][offset_small+jdx].real, mtxfix[i][offset_small+jdx].imag); 
        }
        memset(&mtxfix[i][offset_small+lastcol], 0, sizeof(ComplexFix16)*(max_fpga_cols+1-lastcol));
        mtxfix[i][offset_small+max_fpga_cols].real |= rowchange_indices[idx]; //maximum of 40*6=240 bits
        memset(&mtxfix[i][offset_small+mtxfix[i].stride], 0, ((idx == rows-1 ? adjLoopLength : dfe_loop_length)-1)*sizeof(ComplexFix16)*(max_fpga_cols+1));
      }
      for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxfix[i][jdx].real = useFloat ? fOne : fixpow; 
      for (size_t idx = 0; idx < initDirections.size(); idx++) {
          mtxfix[i][(onerows*dfe_loop_length+idx)*mtxfix[i].stride+max_fpga_cols].real |= ((initDirections[idx] & 0x3f) << 6) | ((initDirections[idx] & 0x80) << (6+6-7)); //maximum of 37*20*7=5180 bits
      } 
      for (size_t idx = 0; idx < mplicity.size(); idx++) {
          mtxfix[i][((rows-1)*dfe_loop_length+idx)*mtxfix[i].stride+max_fpga_cols].real |= mplicity[idx] << (6+6+1); //maximum of 38*20=760 bits
      }
    }

    //note: stride must equal number of columns, or this will not work as the C call expects contiguous data
    ComplexFix16* mtx_fix_data[numinits];
    //assert(mtxfix[i].stride == mtxfix[i].cols);
    for (size_t i = 0; i < numinits; i++) mtx_fix_data[i] = mtxfix[i].get_data();
    
    if (colMux) for (size_t i = (colIndices.size() % 16 == 0) ? 0 : (16 - colIndices.size() % 16); i != 0; i--) colIndices.push_back(0); //round up to nearest 16 bytes to allow streaming
    calcPermanentGlynnRepDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), rows, cols, colIndices.data(),
      rowchange_indices.data(), initDirections.data(), photons, onerows, mplicity.data(), changecount, mulsum, initParities, 1, &perm);

    unlock_lib();
    return;
}

void
GlynnPermanentCalculatorRepeatedInputBatch_DFE(matrix& matrix_init, std::vector<std::vector<PicState_int64>>& input_states,
    std::vector<PicState_int64>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual, int useFloat)
{
    if (output_states.size() == 0) return;
    lock_lib();
    if (!useFloat) init_dfe_lib(DFE_REP, useDual);
    else init_dfe_lib(DFE_REP_FLOAT, useDual);    
    size_t photons = 0;
    for (size_t i = 0; i < output_states[0].size(); i++) {
        photons += output_states[0][i];
    }
    if (!calcPermanentGlynnRepDFE || photons < 1+dfe_basekernpow2) { //compute with other method
        BBFGPermanentCalculatorRepeated gpc;
        for (size_t i = 0; i < output_states.size(); i++) {
            perm[i].resize(input_states[i].size());
            for (size_t j = 0; j < input_states[i].size(); j++) {
                //GlynnPermanentCalculatorRepeated_DFE(matrix_init, input_states[i][j], output_states[i], perm[i][j], useDual); 
                perm[i][j] = gpc.calculate(matrix_init, input_states[i][j], output_states[i], true);
            }
        }
        unlock_lib();
        return;
    }
    for (size_t outp = 0; outp < output_states.size(); outp++) {
        std::vector<uint8_t> rowchange_indices;
        std::vector<uint64_t> mplicity;
        std::vector<uint8_t> initDirections;
        uint8_t onerows, mulsum; uint64_t changecount;
        PicState_int64 adj_input_state = output_states[outp].copy();
        int initParities;
        matrix matrix_mtx = input_to_bincoeff_indices(matrix_init, adj_input_state, useDual, rowchange_indices, mplicity, initDirections, onerows, changecount, mulsum, initParities, false, dfe_loop_length); 
        
        matrix_base<long double> renormalize_data(1, matrix_mtx.cols);
        if (!useFloat) {
            // calulate the maximal sum of the columns to normalize the matrix
            matrix_base<Complex32> colSumMax( matrix_mtx.cols, 2);
            memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
            //sum up vectors in first/upper-left and fourth/lower-right quadrants
            for (size_t idx = 0; idx < matrix_mtx.rows; idx++) {
                for (size_t i=0; i<rowchange_indices[idx]; i++) {
                    for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
                        size_t offset = idx*matrix_mtx.stride + jdx;
                        int realPos = matrix_mtx[offset].real() > 0;
                        int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
                        if (realPos) colSumMax[2*jdx+slopeUpLeft] += matrix_mtx[offset];
                        else colSumMax[2*jdx+slopeUpLeft] -= matrix_mtx[offset];
                    }
                }
            }
            //now try to add/subtract neighbor quadrant values to the prior sum vector to see if it increase the absolute value 
            for (size_t idx = 0; idx < matrix_mtx.rows; idx++) {
                for (size_t i=0; i<rowchange_indices[idx]; i++) {
                    for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
                        size_t offset = idx*matrix_mtx.stride + jdx;
                        int realPos = matrix_mtx[offset].real() > 0;
                        int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
                        Complex32 value1 = colSumMax[2*jdx+1-slopeUpLeft] + matrix_mtx[offset];
                        Complex32 value2 = colSumMax[2*jdx+1-slopeUpLeft] - matrix_mtx[offset];
                        colSumMax[2*jdx+1-slopeUpLeft] = std::norm(value1) > std::norm(value2) ? value1 : value2;
                    }
                } 
            }       
        
            // calculate the renormalization coefficients
            for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
                renormalize_data[jdx] = std::abs(std::norm(colSumMax[2*jdx]) > std::norm(colSumMax[2*jdx+1]) ? colSumMax[2*jdx] : colSumMax[2*jdx+1]);
                //printf("%d %.21Lf %f\n", jdx, renormalize_data[jdx]);
            }
        }
    
        // renormalize the input matrix and convert to fixed point maximizing precision via long doubles
        // SLR and DFE input matrix with 1.0 filling on top row, 0 elsewhere 
        const size_t max_dim = dfe_mtx_size;
        const size_t rows = matrix_mtx.rows;
        const size_t numinits = 4;
        const size_t max_fpga_cols = max_dim / numinits;
        size_t actualinits = (matrix_mtx.cols + max_fpga_cols-1) / max_fpga_cols;
        matrix_base<ComplexFix16> mtxfix[colMux ? numinits : actualinits];
        const long double fixpow = 1ULL << 62;
        const double fOne = doubleToLLRaw(1.0);
        int adjLoopLength = changecount+1 < (unsigned)dfe_loop_length && rowchange_indices[rows - 1] == 1 ? changecount+1 : dfe_loop_length;
        for (size_t i = 0; i < actualinits; i++) {
          mtxfix[i] = matrix_base<ComplexFix16>((rows-1)*dfe_loop_length+adjLoopLength, max_fpga_cols+1); //one extra for row multiplicities, initial directions and Gray codes, binomial coefficients
          size_t basecol = max_fpga_cols * i;
          size_t lastcol = matrix_mtx.cols<=basecol ? 0 : std::min(max_fpga_cols, matrix_mtx.cols-basecol);
          for (size_t idx=0; idx < rows; idx++) {
            size_t offset = idx * matrix_mtx.stride;
            size_t offset_small = idx*dfe_loop_length*mtxfix[i].stride;
            for (size_t jdx = 0; jdx < lastcol; jdx++) {
              mtxfix[i][offset_small+jdx].real = useFloat ? doubleToLLRaw(matrix_mtx[offset+basecol+jdx].real()) : llrintl((long double)matrix_mtx[offset+basecol+jdx].real() * fixpow / renormalize_data[basecol+jdx]);
              mtxfix[i][offset_small+jdx].imag = useFloat ? doubleToLLRaw(matrix_mtx[offset+basecol+jdx].imag()) : llrintl((long double)matrix_mtx[offset+basecol+jdx].imag() * fixpow / renormalize_data[basecol+jdx]);
              //printf("%d %d %d %llX %llX\n", i, idx, jdx, mtxfix[i][offset_small+jdx].real, mtxfix[i][offset_small+jdx].imag); 
            }
            memset(&mtxfix[i][offset_small+lastcol], 0, sizeof(ComplexFix16)*(max_fpga_cols+1-lastcol));
            mtxfix[i][offset_small+max_fpga_cols].real |= rowchange_indices[idx]; //maximum of 40*6=240 bits
            memset(&mtxfix[i][offset_small+mtxfix[i].stride], 0, ((idx == rows-1 ? adjLoopLength : dfe_loop_length)-1)*sizeof(ComplexFix16)*(max_fpga_cols+1));
          }
          for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxfix[i][jdx].real = useFloat ? fOne : fixpow; 
          for (size_t idx = 0; idx < initDirections.size(); idx++) {
              mtxfix[i][(onerows*dfe_loop_length+idx)*mtxfix[i].stride+max_fpga_cols].real |= ((initDirections[idx] & 0x3f) << 6) | ((initDirections[idx] & 0x80) << (6+6-7)); //maximum of 37*20*7=5180 bits
          } 
          for (size_t idx = 0; idx < mplicity.size(); idx++) {
              mtxfix[i][((rows-1)*dfe_loop_length+idx)*mtxfix[i].stride+max_fpga_cols].real |= mplicity[idx] << (6+6+1); //maximum of 38*20=760 bits
          }  
        }        
        
        std::vector<unsigned char> colIndices; colIndices.reserve(photons * input_states[outp].size());
        for (size_t inp = 0; inp < input_states[outp].size(); inp++) {
          for (size_t i = 0; i < input_states[outp][inp].size(); i++) {
              for (size_t j = input_states[outp][inp][i]; j != 0; j--) {
                colIndices.push_back(i);
              }
          }
        }
        
        //rowchange_indices.resize(rows * input_states[outp].size());
        //mplicity.resize(adjLoopLength * input_states[outp].size());
        //int numInitDir = initDirections.size();
        //initDirections.resize(dfe_loop_length * (rows - onerows) * input_states[outp].size());
        size_t mtxsize = ((rows-1)*dfe_loop_length+adjLoopLength) * (max_fpga_cols + 1);
        matrix_base<ComplexFix16> mtxmuxed[numinits] = {};
        if (!colMux) {
            actualinits = (photons + max_fpga_cols-1) / max_fpga_cols;
            for (size_t i = 0; i < actualinits; i++) {
                mtxmuxed[i] = matrix_base<ComplexFix16>(((rows-1)*dfe_loop_length+adjLoopLength) * input_states[outp].size(), max_fpga_cols + 1);
            }
        }
        for (size_t i = 0; i < input_states[outp].size(); i++) { //could improve by copying 1, then 2, then 4, then 8, etc...copy doubling strategy
            /*if (i != 0) { 
                std::copy_n(rowchange_indices.begin(), rows, rowchange_indices.begin() + rows * i);
                std::copy_n(mplicity.begin(), adjLoopLength, mplicity.begin() + adjLoopLength * i);
                std::copy_n(initDirections.begin(), numInitDir, initDirections.begin() + numInitDir * i);
            }*/
            for (size_t j = 0; j < actualinits; j++) {
                if (colMux) {
                    if (i != 0) memcpy(&mtxfix[j][i*mtxsize], &mtxfix[j][0], sizeof(ComplexFix16)*mtxsize);
                } else { //mux the columns on CPU
                    size_t basecol = max_fpga_cols * j;
                    size_t lastcol = photons<=basecol ? 0 : std::min(max_fpga_cols, photons-basecol);
                    for (size_t idx=0; idx < rows; idx++) {
                        size_t offset = i * mtxsize + idx * dfe_loop_length * mtxmuxed[j].stride;
                        for (size_t jdx = 0; jdx < lastcol; jdx++) {
                            size_t idxmtxfix = colIndices[photons*i+basecol+jdx] / max_fpga_cols;
                            mtxmuxed[j][offset+jdx] = mtxfix[idxmtxfix][idx*dfe_loop_length*mtxfix[idxmtxfix].stride+colIndices[photons*i+basecol+jdx] % max_fpga_cols];
                        }
                        memset(&mtxmuxed[j][offset+lastcol], 0, sizeof(ComplexFix16)*(max_fpga_cols+1-lastcol));
                        mtxmuxed[j][offset+max_fpga_cols].real = mtxfix[0][idx*dfe_loop_length*mtxfix[0].stride+max_fpga_cols].real; //maximum of 40*6=240 bits
                        memcpy(&mtxmuxed[j][offset+mtxmuxed[j].stride], &mtxfix[0][(idx*dfe_loop_length+1)*mtxfix[0].stride], ((idx == rows-1 ? adjLoopLength : dfe_loop_length)-1)*sizeof(ComplexFix16)*(max_fpga_cols+1));
                    }
                    for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxmuxed[j][i*mtxsize+jdx].real = useFloat ? fOne : fixpow;
                }
            }
        }
        
        
        if (colMux) for (size_t i = (colIndices.size() % 16 == 0) ? 0 : (16 - colIndices.size() % 16); i != 0; i--) colIndices.push_back(0); //round up to nearest 16 bytes to allow streaming
        //for (size_t i = (rowchange_indices.size() % 16 == 0) ? 0 : (16 - rowchange_indices.size() % 16); i != 0; i--) rowchange_indices.push_back(0); //round up to nearest 16 bytes to allow streaming
        //for (size_t i = (mplicity.size() % 2 == 0) ? 0 : 1; i != 0; i--) mplicity.push_back(0); //round up to nearest 16 bytes to allow streaming
        //for (size_t i = (initDirections.size() % 16 == 0) ? 0 : (16 - initDirections.size() % 16); i != 0; i--) initDirections.push_back(0); //round up to nearest 16 bytes to allow streaming

        //note: stride must equal number of columns, or this will not work as the C call expects contiguous data
        ComplexFix16* mtx_fix_data[numinits];
        //assert(mtxfix[i].stride == mtxfix[i].cols);
        for (size_t i = 0; i < numinits; i++) mtx_fix_data[i] = colMux ? mtxfix[i].get_data() : mtxmuxed[i].get_data();
    
        calcPermanentGlynnRepDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), rows, colMux ? matrix_mtx.cols : photons, colIndices.data(),
          rowchange_indices.data(), initDirections.data(), photons, onerows, mplicity.data(), changecount, mulsum, initParities, input_states[outp].size(), perm[outp].data());
    }
    unlock_lib();
}


void
GlynnPermanentCalculatorRepeatedOutputBatch_DFE(matrix& matrix_init, std::vector<PicState_int64>& input_states,
    std::vector<std::vector<PicState_int64>>& output_states, std::vector<std::vector<Complex16>>& perm, int useDual, int useFloat)
{
    if (input_states.size() == 0) return;
    lock_lib();
    if (!useFloat) init_dfe_lib(DFE_REP, useDual);
    else init_dfe_lib(DFE_REP_FLOAT, useDual);    
    size_t photons = 0;
    for (size_t i = 0; i < input_states[0].size(); i++) {
        photons += input_states[0][i];
    }
    if (1 || !calcPermanentGlynnRepDFE || photons < 1+dfe_basekernpow2) { //compute with other method
        //BBFGPermanentCalculatorRepeated gpc;
        for (size_t i = 0; i < input_states.size(); i++) {
            perm[i].resize(output_states[i].size());
            for (size_t j = 0; j < output_states[i].size(); j++) {
                GlynnPermanentCalculatorRepeated_DFE(matrix_init, input_states[i], output_states[i][j], perm[i][j], useDual, useFloat); 
                //perm[i][j] = gpc.calculate(matrix_init, input_states[i], output_states[i][j], true, false);
            }
        }
        unlock_lib();
        return;
    }
    for (size_t inp = 0; inp < input_states.size(); inp++) {
    /*
        std::vector<uint8_t> allOneRows; allOneRows.reserve(output_states[inp].size());
        std::vector<uint8_t> allMulSums; allMulSums.reserve(output_states[inp].size());
        std::vector<uint64_t> allChangeCounts; allChangeCounts.reserve(output_states[inp].size());
        std::vector<int> allInitParities; allInitParities.reserve(output_states[inp].size());
        std::vector<uint8_t> allRowchange_indices;
        std::vector<uint64_t> allMplicity;
        std::vector<uint8_t> allInitDirections;
        for size_t outp = 0; outp < output_states[inp].size(); outp++) {
            std::vector<uint8_t> rowchange_indices;
            std::vector<uint64_t> mplicity;
            std::vector<uint8_t> initDirections;
            uint8_t onerows, mulsum; uint64_t changecount;
            PicState_int64 adj_input_state = output_states[inp][outp].copy();
            int initParities;
            matrix matrix_mtx = input_to_bincoeff_indices(matrix_init, adj_input_state, useDual, rowchange_indices, mplicity, initDirections, onerows, changecount, mulsum, initParities, transpose, dfe_loop_length);
    
            // calulate the maximal sum of the columns to normalize the matrix
            matrix_base<Complex32> colSumMax( matrix_mtx.cols, 2);
            memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex32) );
            //sum up vectors in first/upper-left and fourth/lower-right quadrants
            for (size_t idx = 0; idx < matrix_mtx.rows; idx++) {
                for (size_t i=0; i<rowchange_indices[idx]; i++) {
                    for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
                        size_t offset = idx*matrix_mtx.stride + jdx;
                        int realPos = matrix_mtx[offset].real() > 0;
                        int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
                        if (realPos) colSumMax[2*jdx+slopeUpLeft] += matrix_mtx[offset];
                        else colSumMax[2*jdx+slopeUpLeft] -= matrix_mtx[offset];
                    }
                }
            }
            //now try to add/subtract neighbor quadrant values to the prior sum vector to see if it increase the absolute value 
            for (size_t idx = 0; idx < matrix_mtx.rows; idx++) {
                for (size_t i=0; i<rowchange_indices[idx]; i++) {
                    for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
                        size_t offset = idx*matrix_mtx.stride + jdx;
                        int realPos = matrix_mtx[offset].real() > 0;
                        int slopeUpLeft = realPos == (matrix_mtx[offset].imag() > 0);
                        Complex32 value1 = colSumMax[2*jdx+1-slopeUpLeft] + matrix_mtx[offset];
                        Complex32 value2 = colSumMax[2*jdx+1-slopeUpLeft] - matrix_mtx[offset];
                        colSumMax[2*jdx+1-slopeUpLeft] = std::norm(value1) > std::norm(value2) ? value1 : value2;
                    }
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
                  mtxfix[i][offset_small+jdx].real = llrintl((long double)matrix_mtx[offset+jdx].real() * fixpow / renormalize_data[basecol+jdx]);
                  mtxfix[i][offset_small+jdx].imag = llrintl((long double)matrix_mtx[offset+jdx].imag() * fixpow / renormalize_data[basecol+jdx]);
                  //printf("%d %d %d %llX %llX\n", i, idx, jdx, mtxfix[i][offset_small+jdx].real, mtxfix[i][offset_small+jdx].imag); 
                }
                memset(&mtxfix[i][offset_small+lastcol], 0, sizeof(ComplexFix16)*(max_fpga_cols-lastcol));
              }
              for (size_t jdx = lastcol; jdx < max_fpga_cols; jdx++) mtxfix[i][jdx].real = fixpow; 
            }        
            
        }        
        std::vector<unsigned char> colIndices; colIndices.reserve(photons * output_states[i].size());
        for (size_t i = 0; i < input_states[inp].size(); i++) {
          for (size_t j = input_state[inp][i]; j != 0; j--) {
            colIndices.push_back(i);
          }
        }

        for (size_t i = (colIndices.size() % 16 == 0) ? 0 : (16 - colIndices.size() % 16); i != 0; i--) colIndices.push_back(0); //round up to nearest 16 bytes to allow streaming
        for (size_t i = (rowchange_indices.size() % 16 == 0) ? 0 : (16 - rowchange_indices.size() % 16); i != 0; i--) rowchange_indices.push_back(0); //round up to nearest 16 bytes to allow streaming
        for (size_t i = (mplicity.size() % 2 == 0) ? 0 : 1; i != 0; i--) mplicity.push_back(0); //round up to nearest 16 bytes to allow streaming
        for (size_t i = (initDirections.size() % 16 == 0) ? 0 : (16 - initDirections.size() % 16); i != 0; i--) initDirections.push_back(0); //round up to nearest 16 bytes to allow streaming
    
        calcPermanentGlynnRepDFE( (const ComplexFix16**)mtx_fix_data, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, colIndices.data(),
          rowchange_indices.data(), initDirections.data(), photons, onerows, mplicity.data(), changecount, mulsum, initParities, output_states[inp].size(), perm[inp].data());
        */
    }

    unlock_lib();
}


}
