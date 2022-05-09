#include <iostream>
#include "GlynnPermanentCalculatorRepeated.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>

namespace pic {



/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
template <typename matrix_type, typename precision_type>
GlynnPermanentCalculatorRepeated<matrix_type, precision_type>::GlynnPermanentCalculatorRepeated() {}



/**
@brief Call to calculate the permanent via Glynn formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion)
@param mtx The effective scattering matrix of a boson sampling instance
@param input_state The input state
@param output_state The output state
@return Returns with the calculated permanent
*/
template <typename matrix_type, typename precision_type>
Complex16 GlynnPermanentCalculatorRepeated<matrix_type, precision_type>::calculate(
    matrix &mtx,
    PicState_int64& input_state,
    PicState_int64& output_state
) {

    int sum_input_states = sum(input_state);
    int sum_output_states = sum(output_state);
    if ( sum_input_states != sum_output_states) {
        std::string error("GlynnPermanentCalculatorRepeated::calculate:  Number of input and output states should be equal");
        throw error;
    }

    if (mtx.rows == 0 || sum_input_states == 0 || sum_output_states == 0)
        return Complex16(1.0, 0.0);

    // column multiplicities are determined by the input state
    PicState_int col_multiplicities =
        convert_PicState_int64_to_PicState_int(input_state);
    PicState_int row_multiplicities =
    // row multiplicities are determined by the output state
        convert_PicState_int64_to_PicState_int(output_state);

    GlynnPermanentCalculatorRepeatedTask<matrix_type, precision_type> calculator(mtx, row_multiplicities, col_multiplicities);
    return calculator.calculate( );
}


/**
@brief Default constructor of the class.
@param mtx Unitary describing a quantum circuit
@param row_multiplicities vector describing the row multiplicity
@param col_multiplicities vector describing the column multiplicity
@return Returns with the instance of the class.
*/
template <typename matrix_type, typename precision_type>
GlynnPermanentCalculatorRepeatedTask<matrix_type, precision_type>::GlynnPermanentCalculatorRepeatedTask(
    matrix &mtx,
    PicState_int& row_multiplicities,
    PicState_int& col_multiplicities
)
    : mtx(mtx)
    , row_multiplicities(row_multiplicities)
    , col_multiplicities(col_multiplicities)
{
    Complex16* mtx_data = mtx.get_data();
    
    // calculate and store 2*mtx being used later in the recursive calls
    mtx2 = matrix_type( mtx.rows, mtx.cols);
    Complex_base<precision_type>* mtx2_data = mtx2.get_data();

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
    deltaLimits = PicState_int(row_multiplicities.size());
    for (size_t i = 0; i < row_multiplicities.size(); i++){
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

/**
@brief Call to calculate the permanent via Glynn formula. scales with n*2^n
@return Returns with the calculated permanent
*/
template <typename matrix_type, typename precision_type>
Complex16
GlynnPermanentCalculatorRepeatedTask<matrix_type, precision_type>::calculate() {

    // if all the elements in row multiplicities are zero, returning default value
    if (minimalIndex == row_multiplicities.size()){
        return Complex16(1.0, 0.0);
    }

    Complex16* mtx_data = mtx.get_data();

    // calculate the initial sum of the columns
    matrix_type colSum( mtx.cols, 1);
    Complex_base<precision_type>* colSum_data = colSum.get_data();
    memset( colSum_data, 0.0, colSum.size()*sizeof(Complex_base<precision_type>));

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
    priv_addend = tbb::combinable<ComplexM<precision_type>> {[](){return ComplexM<precision_type>();}};


    // start the iterations over vectors of deltas
    // values :
    //   colSum              : vector of sums of the columns
    //   sign                : multiplication of all deltas: 1 by default (all deltas are +1)
    //   index_min           : first index greater than 0
    //   currentMultiplicity : multiplicity of the current delta vector
    IterateOverDeltas( colSum, 1, minimalIndex, 1 );


    // sum up partial permanents
    Complex_base<precision_type> permanent( 0.0, 0.0 );

    priv_addend.combine_each([&](ComplexM<precision_type> &a) {
        permanent = permanent + a.get();
    });

    size_t sumMultiplicities = 0;
    for (size_t idx = 0; idx < row_multiplicities.size(); idx++){
        sumMultiplicities += row_multiplicities[idx];
    }

    permanent = permanent / (precision_type)power_of_2( (unsigned long long) (sumMultiplicities-1) );

    return Complex16(permanent.real(), permanent.imag());
}

/**
@brief Method to span parallel tasks via iterative function calls.
(new task is spanned by altering one element in the vector of deltas)
@param colSum The sum of \f$ \delta_j a_{ij} \f$ in Eq. (S2) of arXiv:1606.05836
@param sign The current product \f$ \prod\delta_i $\f
@param index_min \f$ \delta_j a_{ij} $\f with \f$ 0<i<index_min $\f are kept constant, while the signs of \f$ \delta_i \f$  with \f$ i>=idx_min $\f are changed.
@param currentMultiplicity multiplicity of the current delta vector
*/
template <typename matrix_type, typename precision_type>
void 
GlynnPermanentCalculatorRepeatedTask<matrix_type, precision_type>::IterateOverDeltas(
    matrix_type& colSum,
    int sign,
    size_t index_min,
    int currentMultiplicity
) {
    Complex_base<precision_type>* colSum_data = colSum.get_data();

    // Calculate the partial permanent
    Complex_base<precision_type> colSumProd(1.0,0.0);
    for (unsigned int idx=0; idx<colSum.rows; idx++) {
        for (int jdx = 0; jdx < col_multiplicities[idx]; jdx++){
            colSumProd = colSumProd * colSum_data[idx];
        }
    }

    // add partial permanent to the local value
    // multiplicity is given by the binomial formula of multiplicity over sign
    ComplexM<precision_type> &permanent_priv = priv_addend.local();
    permanent_priv += currentMultiplicity * sign * colSumProd;

    tbb::parallel_for( tbb::blocked_range<int>(index_min, mtx.rows), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx){
            int localSign = sign;
            // create an altered vector from the current delta
            matrix_type colSum_new = colSum.copy();

            Complex_base<precision_type>* mtx2_data = mtx2.get_data();
            Complex_base<precision_type>* colSum_new_data = colSum_new.get_data();

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
