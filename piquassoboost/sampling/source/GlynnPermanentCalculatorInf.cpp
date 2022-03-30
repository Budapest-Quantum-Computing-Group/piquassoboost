#include <iostream>
#include "GlynnPermanentCalculatorInf.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>

namespace pic {




/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculatorInf::GlynnPermanentCalculatorInf() {}


/**
@brief Wrapper method to calculate the permanent via Glynn formula. scales with n*2^n
@param mtx Unitary describing a quantum circuit
@return Returns with the calculated permanent
*/
Complex16
GlynnPermanentCalculatorInf::calculate(matrix &mtx) {
    if (mtx.rows == 0)
        return Complex16(1.0, 0.0);

    GlynnPermanentCalculatorInfTask calculator;
    return calculator.calculate( mtx );
}

#define REALPART(c) reinterpret_cast<FloatInf(&)[2]>(c)[0]
#define IMAGPART(c) reinterpret_cast<FloatInf(&)[2]>(c)[1]

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculatorInfTask::GlynnPermanentCalculatorInfTask() {}

/**
@brief Call to calculate the permanent via Glynn formula. scales with n*2^n
@param mtx Unitary describing a quantum circuit
@return Returns with the calculated permanent
*/
Complex16
GlynnPermanentCalculatorInfTask::calculate(matrix &mtx) {

    Complex16* mtx_data = mtx.get_data();
    
    // calculate and store 2*mtx being used later in the recursive calls
    mtx2 = matrix( mtx.rows, mtx.cols);
    Complex16* mtx2_data = mtx2.get_data();

    tbb::parallel_for( tbb::blocked_range<size_t>(0, mtx.rows), [&](tbb::blocked_range<size_t> r) {
        for (size_t row_idx=r.begin(); row_idx<r.end(); ++row_idx){

            size_t row_offset   = row_idx*mtx.stride;
            size_t row_offset_2 = row_idx*mtx2.stride;
            for (size_t col_idx=0; col_idx<mtx.rows; ++col_idx) {
                mtx2_data[row_offset_2+col_idx] = 2*mtx_data[ row_offset + col_idx ];
            }

        }
    });   


    // calulate the initial sum of the columns
    ComplexInf* colSum_data = new ComplexInf[mtx.rows];

    tbb::parallel_for( tbb::blocked_range<size_t>(0, mtx.rows), [&](tbb::blocked_range<size_t> r) {
        for (size_t col_idx=r.begin(); col_idx<r.end(); ++col_idx){
            size_t row_offset = 0;
            for (size_t row_idx=0; row_idx<mtx.rows; ++row_idx) {
                REALPART(colSum_data[col_idx]) += mtx_data[ row_offset + col_idx ].real();
                IMAGPART(colSum_data[col_idx]) += mtx_data[ row_offset + col_idx ].imag();
                row_offset += mtx.stride;
            }

        }
    });

    // thread local storage for partial permanent
    priv_addend = tbb::combinable<ComplexInf> {[](){ return ComplexInf(); }};


    // start the iterations over vectors of deltas
    IterateOverDeltas( colSum_data, 1, 1 );

    // sum up partial permanents
    ComplexInf permanent;

    priv_addend.combine_each([&](ComplexInf &a) {
        REALPART(permanent) += REALPART(a);
        IMAGPART(permanent) += IMAGPART(a);
    });

    permanent /= (long double)power_of_2( (unsigned long long) (mtx.rows-1) );

    delete [] colSum_data;


    return Complex16(REALPART(permanent), IMAGPART(permanent));


}



/**
@brief Method to span parallel tasks via iterative function calls. (new task is spanned by altering one element in the vector of deltas)
@param colSum The sum of \f$ \delta_j a_{ij} \f$ in Eq. (S2) of arXiv:1606.05836
@param index_min \f$ \delta_j a_{ij} $\f with \f$ 0<i<index_min $\f are kept constant, while the signs of \f$ \delta_i \f$  with \f$ i>=idx_min $\f are changed.
@param sign The current product \f$ \prod\delta_i $\f
*/
void 
GlynnPermanentCalculatorInfTask::IterateOverDeltas( pic::ComplexInf* colSum_data, int sign, int index_min ) {

    //pic::ComplexInf* colSum_data = colSum.get_data();

    // Calculate the partial permanent
    pic::ComplexInf colSumProd(colSum_data[0]);
    for (int idx=1; idx<mtx2.cols; idx++) { //(a+bi)(c+di)=ac-bd+(bc+ad)i or (ac - bd)+((a+b)*(c+d)-ac-bd)i
        //colSumProd *= colSum_data[idx];
        /*FloatInf acbd(REALPART(colSumProd) * REALPART(colSum_data[idx]));
        acbd -= IMAGPART(colSumProd) * IMAGPART(colSum_data[idx]);
        FloatInf bcad(IMAGPART(colSumProd) * REALPART(colSum_data[idx]));
        bcad += REALPART(colSumProd) * IMAGPART(colSum_data[idx]);
        REALPART(colSumProd) = acbd;
        IMAGPART(colSumProd) = bcad;*/
        FloatInf ac(std::move(REALPART(colSumProd) * REALPART(colSum_data[idx])));
        FloatInf bd(std::move(IMAGPART(colSumProd) * IMAGPART(colSum_data[idx])));
        FloatInf p(std::move(REALPART(colSumProd) + IMAGPART(colSumProd)));
        p *= REALPART(colSum_data[idx]) + IMAGPART(colSum_data[idx]); p -= ac + bd;
        ac -= bd;
        REALPART(colSumProd) = ac;
        IMAGPART(colSumProd) = p;
    }

    // add partial permanent to the local value
    pic::ComplexInf &permanent_priv = priv_addend.local();
    if (sign > 0) {
        REALPART(permanent_priv) += REALPART(colSumProd);
        IMAGPART(permanent_priv) += IMAGPART(colSumProd);
    } else {
        REALPART(permanent_priv) -= REALPART(colSumProd);
        IMAGPART(permanent_priv) -= IMAGPART(colSumProd);
    }
    


    tbb::parallel_for( tbb::blocked_range<int>(index_min,mtx2.rows), [&](tbb::blocked_range<int> r) {
        for (size_t idx=r.begin(); idx<r.end(); ++idx){

            // create an altered vector from the current delta
            pic::ComplexInf* colSum_new_data = new pic::ComplexInf[mtx2.cols];

            pic::Complex16* mtx2_data = mtx2.get_data();
            
            size_t row_offset = idx*mtx2.stride;

            for (size_t jdx=0; jdx<mtx2.cols; jdx++) {
                REALPART(colSum_new_data[jdx]) = REALPART(colSum_data[jdx]);
                REALPART(colSum_new_data[jdx]) -= mtx2_data[row_offset+jdx].real(); 
                IMAGPART(colSum_new_data[jdx]) = IMAGPART(colSum_data[jdx]);
                IMAGPART(colSum_new_data[jdx]) -= mtx2_data[row_offset+jdx].imag();
            }

            // spawn new iteration            
            IterateOverDeltas( colSum_new_data, -sign, idx+1 );
            delete [] colSum_new_data;
        }
    });


/*
    for (int idx=index_min; idx<colSum.size(); ++idx){
        // create an altered vector from the current delta
        matrix_base<pic::ComplexInf> colSum_new = colSum.copy();
        pic::ComplexInf* mtx2_data = mtx2.get_data();
        pic::ComplexInf* colSum_new_data = colSum_new.get_data();
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
