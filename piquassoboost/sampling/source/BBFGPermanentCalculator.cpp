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
#include "BBFGPermanentCalculator.h"
#include "BBFGPermanentCalculator.hpp"

#ifdef __MPFR__
#include "InfinitePrecisionComplex.h"
#endif

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


template<typename T>
T product_reduction( const matrix_base<T>& mtx ) {

    T ret = mtx[0];
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
BBFGPermanentCalculator::BBFGPermanentCalculator(  ) {


    //mtx = mtx_in;

}

/**
@brief Destructor of the class.
*/
BBFGPermanentCalculator::~BBFGPermanentCalculator() {


}


/**
@brief Call to calculate the hafnian of a complex matrix
@param use_extended Logical variable to indicate whether use extended precision for cholesky decomposition (default), or not.
@return Returns with the calculated hafnian
*/
Complex16
BBFGPermanentCalculator::calculate(matrix& mtx_in, bool use_extended, bool use_inf) {


    if (mtx_in.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return 1.0;
    }
    
    
    Update_mtx(mtx_in);    
  
    if (use_inf) {
#ifdef __MPFR__
        BBFGPermanentCalculator_Tasks<matrix, ComplexInf, FloatInf> permanent_calculator(mtx);
        return permanent_calculator.calculate();
#else
    std::string error("BBFGPermanentCalculator::calculate:  MPFR Infinite Precision not included");
        throw error;
#endif
    } else
    if (use_extended) {
        matrix32 mtx32(mtx.rows, mtx.cols);
        for( size_t idx=0; idx<mtx.size(); idx++ ) {
            mtx32[idx].real( mtx[idx].real() );
            mtx32[idx].imag( mtx[idx].imag() );
        }
        BBFGPermanentCalculator_Tasks<matrix32, Complex32, long double> permanent_calculator(mtx32);
        return permanent_calculator.calculate();
    }
    else {
        BBFGPermanentCalculator_Tasks<matrix, Complex16, double> permanent_calculator(mtx);
        return permanent_calculator.calculate();
    }




}


/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void
BBFGPermanentCalculator::Update_mtx( matrix &mtx_in ){

    mtx = mtx_in;
}








} // PIC
