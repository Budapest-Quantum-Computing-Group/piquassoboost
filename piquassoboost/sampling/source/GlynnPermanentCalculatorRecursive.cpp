#include <iostream>
#include "GlynnPermanentCalculatorRecursive.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>

namespace pic {




GlynnPermanentCalculatorRecursive::GlynnPermanentCalculatorRecursive() {}


Complex16 GlynnPermanentCalculatorRecursive::calculate(
    matrix &mtx,
    PicState_int64& input_state,
    PicState_int64& output_state
) {

    // column multiplicities are determined by the input state
    array_int col_multiplicities(input_state.size());
    for (unsigned int i = 0; i < input_state.size(); i++){
        col_multiplicities[i] = input_state[i];
    }
    // row multiplicities are determined by the output state
    array_int row_multiplicities(output_state.size());
    for (unsigned int i = 0; i < output_state.size(); i++){
        row_multiplicities[i] = output_state[i];
    }

    GlynnPermanentCalculatorRecursiveTask calculator(mtx, row_multiplicities, col_multiplicities);
    return calculator.calculate( );
    
}



GlynnPermanentCalculatorRecursiveTask::GlynnPermanentCalculatorRecursiveTask(
    matrix &mtx,
    array_int& row_multiplicities,
    array_int& col_multiplicities
)
    : mtx(mtx)
    , row_multiplicities(row_multiplicities)
    , col_multiplicities(col_multiplicities)
{
    Complex16* mtx_data = mtx.get_data();
    
    // calculate and store 2*mtx being used later in the recursive calls
    mtx2 = matrix32( mtx.rows, mtx.cols);
    Complex32* mtx2_data = mtx2.get_data();

    tbb::parallel_for( tbb::blocked_range<size_t>(0, mtx.rows), [&](tbb::blocked_range<size_t> r) {
        for (size_t row_idx=r.begin(); row_idx<r.end(); ++row_idx){

            size_t row_offset   = row_idx*mtx.stride;
            size_t row_offset_2 = row_idx*mtx2.stride;
            for (size_t col_idx=0; col_idx<mtx.cols; ++col_idx) {
                mtx2_data[row_offset_2+col_idx] = 2*mtx_data[ row_offset + col_idx ];
            }

        }
    });

    // minimal index set to higher than maximal in case of empty row_multiplicities
    minimalIndex = row_multiplicities.size();

    // deltaLimits stores the index range of the deltas:
    // first not zero has to be one smaller than the multiplicity
    // others have to be the multiplicity
    deltaLimits = array_int(row_multiplicities.size());
    for (unsigned int i = 0; i < row_multiplicities.size(); i++){
        if (row_multiplicities[i]>0){
            if (minimalIndex > i){
                deltaLimits[i] = row_multiplicities[i]-1;
                minimalIndex = i;
            }else{
                deltaLimits[i] = row_multiplicities[i];
            }
        }else{
            deltaLimits[i] = 0;
        }
    }
}


Complex16
GlynnPermanentCalculatorRecursiveTask::calculate() {

    // if all the elements in row multiplicities are zero, returning default value
    if (minimalIndex == row_multiplicities.size()){
        return Complex16(1.0, 0.0);
    }

    Complex16* mtx_data = mtx.get_data();

    // calculate the initial sum of the columns
    matrix32 colSum( mtx.cols, 1);
    Complex32* colSum_data = colSum.get_data();
    memset( colSum_data, 0.0, colSum.size()*sizeof(Complex32));

    tbb::parallel_for( tbb::blocked_range<size_t>(0, mtx.cols), [&](tbb::blocked_range<size_t> r) {
        for (size_t col_idx=r.begin(); col_idx<r.end(); ++col_idx){

            size_t row_offset = 0;
            for (size_t row_idx=0; row_idx<mtx.rows; ++row_idx) {
                colSum_data[col_idx] += row_multiplicities[row_idx] * mtx_data[ row_offset + col_idx ];
                row_offset += mtx.stride;
            }

        }
    });

    // thread local storage for partial permanent
    priv_addend = tbb::combinable<ComplexM<long double>> {[](){return ComplexM<long double>();}};


    // start the iterations over vectors of deltas
    // values :
    //   colSum              : vector of sums of the columns
    //   sign                : multiplication of all deltas: 1 by default (all deltas are +1)
    //   index_min           : first index greater than 0
    //   currentMultiplicity : multiplicity of the current delta vector
    IterateOverDeltas( colSum, 1, minimalIndex, 1 );


    // sum up partial permanents
    Complex32 permanent( 0.0, 0.0 );

    priv_addend.combine_each([&](ComplexM<long double> &a) {
        permanent = permanent + a.get();
    });

    size_t sumMultiplicities = 0;
    for (size_t idx = 0; idx < row_multiplicities.size(); idx++){
        sumMultiplicities += row_multiplicities[idx];
    }

    permanent = permanent / (long double)power_of_2( (unsigned long long) (sumMultiplicities-1) );

    return Complex16(permanent.real(), permanent.imag());
}


void 
GlynnPermanentCalculatorRecursiveTask::IterateOverDeltas(
    matrix32& colSum,
    int sign,
    int index_min,
    int currentMultiplicity
) {
    Complex32* colSum_data = colSum.get_data();

    // Calculate the partial permanent
    Complex32 colSumProd(1.0,0.0);
    for (unsigned int idx=0; idx<colSum.rows; idx++) {
        for (int jdx = 0; jdx < col_multiplicities[idx]; jdx++){
            colSumProd = colSumProd * colSum_data[idx];
        }
    }

    // add partial permanent to the local value
    // multiplicity is given by the binomial formula of multiplicity over sign
    ComplexM<long double> &permanent_priv = priv_addend.local();
    permanent_priv += currentMultiplicity * sign * colSumProd;

    tbb::parallel_for( tbb::blocked_range<int>(index_min, mtx.rows), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx){
            int localSign = sign;
            // create an altered vector from the current delta
            matrix32 colSum_new = colSum.copy();

            Complex32* mtx2_data = mtx2.get_data();
            Complex32* colSum_new_data = colSum_new.get_data();

            size_t row_offset = idx*mtx2.stride;

            // deltaLimits is the same as row_multiplicity except the first non-zero element
            for (int indexOfMultiplicity = 1; indexOfMultiplicity <= deltaLimits[idx]; indexOfMultiplicity++){
                for (unsigned int jdx=0; jdx<mtx2.cols; jdx++) {
                    colSum_new_data[jdx] = colSum_new_data[jdx] - mtx2_data[row_offset+jdx];
                }

                localSign *= -1;
                IterateOverDeltas(
                    colSum_new, 
                    localSign,
                    idx+1,
                    currentMultiplicity*binomialCoeff(deltaLimits[idx], indexOfMultiplicity)
                );
            }
        }
    });

    return;
}









} // PIC
