#include <iostream>
#include "GlynnPermanentCalculator.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>


namespace pic {




/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculator::GlynnPermanentCalculator() {}


/**
@brief Wrapper method to calculate the permanent via Glynn formula. scales with n*2^n
@param mtx Unitary describing a quantum circuit
@return Returns with the calculated permanent
*/
Complex16
GlynnPermanentCalculator::calculate(matrix &mtx) {
    if (mtx.rows == 0)
        return Complex16(1.0, 0.0);

    GlynnPermanentCalculatorTask calculator;
    return calculator.calculate( mtx );
}



/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculatorTask::GlynnPermanentCalculatorTask() {}

/**
@brief Call to calculate the permanent via Glynn formula. scales with n*2^n
@param mtx Unitary describing a quantum circuit
@return Returns with the calculated permanent
*/
Complex16
GlynnPermanentCalculatorTask::calculate(matrix &mtx) {

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


    // calulate the initial sum of the columns
    matrix32 colSum( mtx.cols, 1);
    Complex32* colSum_data = colSum.get_data();
    memset( colSum_data, 0.0, colSum.size()*sizeof(Complex32));



    tbb::parallel_for( tbb::blocked_range<size_t>(0, mtx.cols), [&](tbb::blocked_range<size_t> r) {
        for (size_t col_idx=r.begin(); col_idx<r.end(); ++col_idx){

            size_t row_offset = 0;
            for (size_t row_idx=0; row_idx<mtx.rows; ++row_idx) {
                colSum_data[col_idx] += mtx_data[ row_offset + col_idx ];
                row_offset += mtx.stride;
            }

        }
    });

    // thread local storage for partial permanent
    priv_addend = tbb::combinable<ComplexM<long double>> {[](){return ComplexM<long double>();}};


    // start the iterations over vectors of deltas
    IterateOverDeltas( colSum, 1, 1 );


    // sum up partial permanents
    Complex32 permanent( 0.0, 0.0 );

    priv_addend.combine_each([&](ComplexM<long double> &a) {
        permanent = permanent + a.get();
    });

    permanent = permanent / (long double)power_of_2( (unsigned long long) (mtx.rows-1) );

  


    return Complex16(permanent.real(), permanent.imag());


}



/**
@brief Method to span parallel tasks via iterative function calls. (new task is spanned by altering one element in the vector of deltas)
@param colSum The sum of \f$ \delta_j a_{ij} \f$ in Eq. (S2) of arXiv:1606.05836
@param index_min \f$ \delta_j a_{ij} $\f with \f$ 0<i<index_min $\f are kept constant, while the signs of \f$ \delta_i \f$  with \f$ i>=idx_min $\f are changed.
@param sign The current product \f$ \prod\delta_i $\f
*/
void 
GlynnPermanentCalculatorTask::IterateOverDeltas( matrix32& colSum, int sign, int index_min ) {

    Complex32* colSum_data = colSum.get_data();

    // Calculate the partial permanent
    Complex32 colSumProd(1.0,0.0);
    for (int idx=0; idx<colSum.rows; idx++) {
        colSumProd = colSumProd * colSum_data[idx];
    }

    // add partial permanent to the local value
    ComplexM<long double> &permanent_priv = priv_addend.local();
    permanent_priv += sign*colSumProd;


    tbb::parallel_for( tbb::blocked_range<int>(index_min,mtx2.rows), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx){

            // create an altered vector from the current delta
            matrix32 colSum_new = colSum.copy();

            Complex32* mtx2_data = mtx2.get_data();
            Complex32* colSum_new_data = colSum_new.get_data();

            size_t row_offset = idx*mtx2.stride;

            for (int jdx=0; jdx<mtx2.cols; jdx++) {
                colSum_new_data[jdx] = colSum_new_data[jdx] - mtx2_data[row_offset+jdx];
            }

            // spawn new iteration            
            IterateOverDeltas( colSum_new, -sign, idx+1 );
        }
    });


/*
    for (int idx=index_min; idx<colSum.size(); ++idx){

        // create an altered vector from the current delta
        matrix32 colSum_new = colSum.copy();

        Complex32* mtx2_data = mtx2.get_data();
        Complex32* colSum_new_data = colSum_new.get_data();

        size_t row_offset = idx*mtx2.stride;

        for (int jdx=0; jdx<mtx2.cols; jdx++) {
            colSum_new_data[jdx] = colSum_new_data[jdx] - mtx2_data[row_offset+jdx];
        }

        // spawn new iteration            
        IterateOverDeltas( colSum_new, -sign, idx+1 );
    }
  */


    return;
}









} // PIC
