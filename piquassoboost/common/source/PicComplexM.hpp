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

#ifndef PicComplexM_H
#define PicComplexM_H

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
@brief Multi-precision representation of complex numbers used in the PiquassoBoost package.
This class can be used to sum up many complex numbers of different orders. (Used in hafnian calculations)
*/
template<class scalar>
class ComplexM {


private:

    /// addends storing the components of the number of different orders. The stored number is the sum of the addends
    std::vector<Complex_base<scalar>> addends;

public:


/**
@brief Nullary Contructor of the class
@return Returns with the created class instance
*/
ComplexM<scalar> () {

};

/**
@brief Contructor of the class
@param a The real part of the complex number
@param b The imaginary part of the complex number
@return Returns with the created class instance
*/
ComplexM<scalar> ( scalar a, scalar b) {

    // reserve space for 3 orders
    addends.reserve(3);

    addends.push_back(Complex_base<scalar>(a,b));

};


/**
@brief Contructor of the class
@param a The real part of the complex number (The imaginary part is set to zero)
@return Returns with the created class instance
*/
ComplexM<scalar>( scalar a) {

    // reserve space for 3 orders
    addends.reserve(3);

    addends.push_back(Complex_base<scalar>(a, 0.0));

};

/**
@brief Define operator += for the multiprecision complex number representation
@param value A Complex_base<scalar> valued input.
*/
void operator +=( const Complex_base<scalar> &value) {

    add( value, 0);

    return;
};



/**
@brief Define operator += for the multiprecision complex number representation
@param value A Complex_base<scalar> valued input.
*/
void operator -=( const Complex_base<scalar> &value) {

    subtract( value, 0);

    return;
};



private:

/**
@brief Add a complex number to the stored multiprecision number
@param value A Complex_base<scalar> valued input.
@param index
*/
void add( const Complex_base<scalar> &value, size_t index) {

    if ( addends.size() <= index) {
        addends.push_back(value);
        return;
    }

    Complex_base<scalar> tmp = addends[index];

    if ( tmp*ORDER_CUTOFF > value ) {
        addends[index] = value;
        add( tmp, index+1);
    }
    else if ( tmp < value*ORDER_CUTOFF ) {
        add( value, index+1);
    }
    else {
        addends[index] += value;
    }


    return;
};


/**
@brief Add a complex number to the stored multiprecision number
@param value A Complex_base<scalar> valued input.
@param index
*/
void subtract( const Complex_base<scalar> &value, size_t index) {

    if ( addends.size() <= index) {
        addends.push_back(value);
        return;
    }

    Complex_base<scalar> tmp = addends[index];

    if ( tmp*ORDER_CUTOFF > value ) {
        addends[index] = value;
        add( tmp, index+1);
    }
    else if ( tmp < value*ORDER_CUTOFF ) {
        add( value, index+1);
    }
    else {
        addends[index] -= value;
    }


    return;
};


public:

/**
@brief Get the sum of the addends stored in the number representation
@return The stored number in  Complex_base<scalar> format.
*/
Complex_base<scalar> get() {

    if ( addends.size() == 0) {
        return Complex_base<scalar>(0.0,0.0);
    }

    Complex_base<scalar> tmp(0.0,0.0);

    for (size_t idx=0; idx<addends.size(); idx++) {
//std::cout <<  "pocket " << idx << ": " << addends[idx] << std::endl;
        tmp += addends[idx];
    }


    return tmp;
};



/**
@brief Override operator + of the STL complex class.
@param value An std::complex valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
/*
Complex_base<scalar> operator+( const std::complex<scalar> &value) {

    return Complex_base<scalar>(this->real() + value.real(), this->imag() + value.imag());

};
*/

/**
@brief Override operator + of the STL complex class.
@param value A Complex_base<scalar> valued input.
@return Returns with the calculated value represented by an instance of the Complex_base<scalar> class.
*/
/*
Complex_base<scalar> operator+( const Complex_base<scalar> &value) {

    return Complex_base<scalar>(this->real() + value.real(), this->imag() + value.imag());

};
*/

/**
@brief Override operator -* of the STL complex class.
@param value A std::complex valued input.
@return Returns with the calculated value represented by an instance of the Complex_base<scalar> class.
*/
/*
Complex_base<scalar> operator-( const std::complex<scalar> &value) {

    return Complex_base<scalar>(this->real() - value.real(), this->imag() - value.imag());

};
*/

/**
@brief Override operator = of the STL complex class.
@param value A std::complex valued input.
*/
/*
void operator=( const std::complex<scalar> &value) {
    this->real(value.real());
    this->imag(value.imag());
    return;

};
*/

/**
@brief Override operator + of the STL complex class.
@param value A Complex_base<scalar> valued input.
*/
/*
void operator=( const Complex_base<scalar> &value) {
    this->real(value.real());
    this->imag(value.imag());
    return;

};
*/

/**
@brief Override operator + of the STL complex class.
@param value A double valued input (would be set as the real part of the complex number).
*/
/*
void operator=( const double &value) {
    this->real(value);
    this->imag(0);
    return;

};
*/



};



}  //PIC

#endif
