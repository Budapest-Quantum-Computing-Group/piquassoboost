/**
 * Copyright 2022 Budapest Quantum Computing Group
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

#ifndef GLYNN_PERMANENT_CALCULATOR_SIMPLE_HPP
#define GLYNN_PERMANENT_CALCULATOR_SIMPLE_HPP

#include <bitset>
#include <iostream>
#include "GlynnPermanentCalculatorSimple.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>


namespace pic {




/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
template<typename matrix_type, typename precision_type>
GlynnPermanentCalculatorSimple<matrix_type, precision_type>::GlynnPermanentCalculatorSimple() {}


/**
@brief Wrapper method to calculate the permanent via Glynn formula. scales with n*2^n
@param mtx Unitary describing a quantum circuit
@return Returns with the calculated permanent
*/
template<typename matrix_type, typename precision_type>
inline Complex16
GlynnPermanentCalculatorSimple<matrix_type, precision_type>::calculate(matrix &mtx) {
    GlynnPermanentCalculatorSimpleTask<matrix_type, precision_type> calculator;
    
    return calculator.calculate( mtx );
}


/**
@brief Wrapper method to calculate the permanent via Glynn formula. scales with n^2 * 2^n
@param mtx Unitary describing a quantum circuit
@return Returns with the calculated permanent
*/
template<typename matrix_type, typename precision_type>
Complex16
GlynnPermanentCalculatorSimpleTask<matrix_type, precision_type>::calculate(matrix &mtx) {
    // edge cases
    if (mtx.rows == 0 || mtx.cols == 0)
        return Complex16(1.0, 0.0);
    if (mtx.rows != mtx.cols){
        std::cerr << "Matrix is not rectangular! Returning 0." << std::endl;
        return Complex16(0.0, 0.0);
    }

    const size_t powerOf2 = power_of_2(mtx.rows - 1);

    // thread local storage for partial sums
    partialSums = tbb::combinable<ComplexM<precision_type> > {[](){return ComplexM<precision_type>();}};

    tbb::parallel_for( tbb::blocked_range<size_t>(0, powerOf2), [&](tbb::blocked_range<size_t> range) {

        // container for delta values
        std::vector<int> deltas;
        deltas.reserve(mtx.rows);

        // container for the product of deltas (first the number of -1's)
        int sign = 0;

        // local variable for filling the container deltas
        size_t firstDelta = range.begin();

        for (int i = 1; i < mtx.rows; ++i){
            if (0 == firstDelta % 2){
                deltas.insert(deltas.begin(), +1);
            }else{
                deltas.insert(deltas.begin(), -1);
                sign++;
            }
            // bitwise shift in firstDelta
            firstDelta >>= 1;
        }

        // first delta always 1
        deltas.insert(deltas.begin(), +1);


        // calculate the value of sign
        sign = sign % 2 ? -1 : 1;


        //for (size_t d = 0; d < powerOf2; ++d){
        for (size_t d = range.begin(); d < range.end(); ++d){

            // calculate the product of the row sums with delta coefficient
            // \prod_{i=1}^n \sum_{j=1}^n delta_j * a_{ij}
            Complex_base<precision_type> product(1.0, 0.0);
            for (size_t rowIndex = 0; rowIndex < mtx.rows; ++rowIndex){
                auto firstInRowElement = mtx[rowIndex * mtx.stride];
                // initial value of the first element in the row
                Complex_base<precision_type> rowSum(firstInRowElement.real(), firstInRowElement.imag());

                for (size_t colIndex = 1; colIndex < mtx.cols; ++colIndex){
                    rowSum += deltas[colIndex] * mtx[rowIndex * mtx.stride + colIndex];    
                }

                product *= rowSum;
            }
            

            // add partial permanent to the local value
            ComplexM<precision_type> &partialSumLocal = this->partialSums.local();
            partialSumLocal += sign * product;

            
            // change of deltas and sign
            size_t nextD = d+1;
            int indexChanged = 1;
            while(nextD > 0){
                if (nextD % 2 == 1){
                    nextD = 0;
                }else{
                    nextD >>= 1;
                    ++indexChanged;
                }
            }

            // change sign if needed
            if (indexChanged % 2 != 0){
                sign *= -1;
            }

            // change container deltas
            for (size_t deltaIndex = 0; deltaIndex < indexChanged; ++deltaIndex){
                deltas[mtx.rows - deltaIndex - 1] *= -1;
            }
        }
    });


    // sum up partial permanent values
    Complex_base<precision_type> permanent( 0.0, 0.0 );

    partialSums.combine_each([&](ComplexM<precision_type> &a) {
        permanent = permanent + a.get();
    });

    // normalize permanent value
    permanent = permanent / (precision_type)powerOf2;
    
    return Complex16(permanent.real(), permanent.imag());
}







} // PIC

#endif // GLYNN_PERMANENT_CALCULATOR_SIMPLE_HPP
