/**
 * Copyright 2021 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LONG_DOUBLE_CUTOFF
#define LONG_DOUBLE_CUTOFF 50
#endif // LONG_DOUBLE_CUTOFF

#include <iostream>
#include "BBFGPermanentCalculatorRepeated.h"
#include "BBFGPermanentCalculatorRepeated.hpp"

#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>
#include <assert.h>





static tbb::spin_mutex my_mutex;
/*
double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {

Complex16 product_reduction( const matrix& mtx ) {

    Complex16 ret = mtx[0];
    for (size_t idx=1; idx<mtx.size(); idx++) {
        ret *= mtx[idx];
    }
    
    return ret;

}



Complex32 product_reduction( const matrix32& mtx ) {

    Complex32 ret = mtx[0];
    for (size_t idx=1; idx<mtx.size(); idx++) {
        ret *= mtx[idx];
    }
    
    return ret;


}


/**
@brief Constructor of the class.

@param mtx_in ?????????????,,
@return Returns with the instance of the class.
*/
BBFGPermanentCalculatorRepeated::BBFGPermanentCalculatorRepeated(  ) {


    //mtx = mtx_in;

}

/**
@brief Destructor of the class.
*/
BBFGPermanentCalculatorRepeated::~BBFGPermanentCalculatorRepeated() {


}


/**
@brief Call to calculate the hafnian of a complex matrix
@param use_extended Logical variable to indicate whether use extended precision for cholesky decomposition (default), or not.
@return Returns with the calculated hafnian
*/
Complex16
BBFGPermanentCalculatorRepeated::calculate(matrix& mtx, PicState_int64& col_mult64, PicState_int64& row_mult64, bool use_extended) {


    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return 1.0;
    }
    
    PicState_int row_mult( row_mult64.size() );
    for (size_t idx=0; idx<row_mult.size(); idx++) {
        row_mult[idx] = (int)row_mult64[idx];
    }


    PicState_int col_mult( col_mult64.size() );
    for (size_t idx=0; idx<col_mult.size(); idx++) {
        col_mult[idx] = (int)col_mult64[idx];
    }


    Complex16&& ret = calculate( mtx, col_mult, row_mult, use_extended );
    return ret;

}

/**
@brief Call to calculate the hafnian of a complex matrix
@param use_extended Logical variable to indicate whether use extended precision for cholesky decomposition (default), or not.
@return Returns with the calculated hafnian
*/
Complex16
BBFGPermanentCalculatorRepeated::calculate(matrix& mtx, PicState_int& col_mult, PicState_int& row_mult, bool use_extended) {


    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return 1.0;
    }
    
     

    if (use_extended) {
        matrix32 mtx32(mtx.rows, mtx.cols);
        for( size_t idx=0; idx<mtx.size(); idx++ ) {
            mtx32[idx].real( mtx[idx].real() );
            mtx32[idx].imag( mtx[idx].imag() );
        }
        BBFGPermanentCalculatorRepeated_Tasks<matrix32, Complex32, long double> permanent_calculator(mtx32, col_mult, row_mult);
        return permanent_calculator.calculate();
    }
    else {
        BBFGPermanentCalculatorRepeated_Tasks<matrix, Complex16, double> permanent_calculator(mtx, col_mult, row_mult);
        return permanent_calculator.calculate();
    }




}










} // PIC
