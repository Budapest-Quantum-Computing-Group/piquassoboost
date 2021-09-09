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

#ifndef PicRealM_H
#define PicRealM_H

#include <assert.h>
#include <cstddef>
#include <complex>
#include <cmath>
#include <vector>
#include "PicComplex.hpp"

#include <iostream>

#ifndef ORDER_CUTOFF
#define ORDER_CUTOFF 1e-5
#endif


/// The namespace of the Picasso project
namespace pic {


/**
@brief Multi-precision representation of real numbers used in the PiquassoBoost package.
This class can be used to sum up many real numbers of different orders. (Used in torontonian calculations)
*/
template<class scalar>
class RealM {


private:

    /// addends storing the components of the number of different orders. The stored number is the sum of the addends
    std::vector<scalar> addends;

public:


/**
@brief Nullary Contructor of the class
@return Returns with the created class instance
*/
RealM<scalar> () {

};



/**
@brief Contructor of the class
@param a The real number
@return Returns with the created class instance
*/
RealM<scalar>( scalar a) {

    // reserve space for 3 orders
    addends.reserve(3);

    addends.push_back(a);

};

/**
@brief Define operator += for the multiprecision real number representation
@param value A scalar valued input.
*/
void operator +=( const scalar &value) {

    add( value, 0);

    return;
};



private:

/**
@brief Add a real number to the stored multiprecision number
@param value A scalar valued input.
@param index
*/
void add( const scalar &value, size_t index) {

    if ( addends.size() <= index) {
        addends.push_back(value);
        return;
    }

    scalar tmp = addends[index];

    if ( std::abs(tmp)*ORDER_CUTOFF > std::abs(value) ) {
        addends[index] = value;
        add( tmp, index+1);
    }
    else if ( std::abs(tmp) < std::abs(value)*ORDER_CUTOFF ) {
        add( value, index+1);
    }
    else {
        addends[index] += value;
    }


    return;
};


public:

/**
@brief Get the sum of the addends stored in the number representation
@return The stored number in scalar format.
*/
scalar get() {

    if ( addends.size() == 0) {
        return 0.0;
    }

    scalar tmp = 0.0;

    for (size_t idx=0; idx<addends.size(); idx++) {
        tmp += addends[idx];
    }


    return tmp;
};





};



}  //PIC

#endif
