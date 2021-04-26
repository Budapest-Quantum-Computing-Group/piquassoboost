#ifndef PicComplex_H
#define PicComplex_H

#include <assert.h>
#include <cstddef>
#include <complex>
#include <cmath>



/// The namespace of the Picasso project
namespace pic {

/**
@brief Double precision complex number representation used in the CPiquasso package.
This enables better performance than the STL version of the complex number.
*/
template<class scalar>
class Complex_base : public std::complex<scalar> {



public:


/**
@brief Nullary Contructor of the class
@return Returns with the created class instance
*/
Complex_base<scalar> () {

};

/**
@brief Contructor of the class
@param a The real part of the complex number
@param b The imaginary part of the complex number
@return Returns with the created class instance
*/
Complex_base<scalar> ( scalar a, scalar b) {

    this->real(a);
    this->imag(b);

};


/**
@brief Contructor of the class
@param a The real part of the complex number (The imaginary part is set to zero)
@return Returns with the created class instance
*/
Complex_base<scalar> ( scalar a) {

    this->real(a);
    this->imag(0.0);

};


/**
@brief Override operator / of the STL complex class.
@param value A double valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
Complex_base<scalar> operator/( const double &value) {

    return Complex_base<scalar>(this->real()/value, this->imag()/value);

};


/**
@brief Override operator / of the STL complex class.
@param value A double valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
Complex_base<scalar> operator/( const long double &value) {

    return Complex_base<scalar>(this->real()/value, this->imag()/value);

};


/**
@brief Override operator / of the STL complex class.
@param value A Complex16 valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
Complex_base<scalar> operator/( const Complex_base<scalar> &value) {

    double norm_input = value.real()*value.real() + value.imag()*value.imag();

    return Complex_base<scalar>((this->real()*value.real() + this->imag()*value.imag())/norm_input, (this->imag()*value.real() - this->real()*value.imag())/norm_input);

};


/**
@brief Override operator + of the STL complex class.
@param value An std::complex valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
Complex_base<scalar> operator+( const std::complex<scalar> &value) {

    return Complex_base<scalar>(this->real() + value.real(), this->imag() + value.imag());

};

/**
@brief Override operator + of the STL complex class.
@param value A Complex_base<scalar> valued input.
@return Returns with the calculated value represented by an instance of the Complex_base<scalar> class.
*/
Complex_base<scalar> operator+( const Complex_base<scalar> &value) {

    return Complex_base<scalar>(this->real() + value.real(), this->imag() + value.imag());

};


/**
@brief Override operator * of the STL complex class.
@param value A std::complex valued input.
@return Returns with the calculated value represented by an instance of the Complex_base<scalar> class.
*/
Complex_base<scalar> operator*( const std::complex<scalar> &value) {

    return Complex_base<scalar>(this->real()*value.real() - this->imag()*value.imag(), this->real()*value.imag() + this->imag()*value.real());

};

/**
@brief Override operator * of the STL complex class.
@param value A Complex_base<scalar> valued input.
@return Returns with the calculated value represented by an instance of the Complex_base<scalar> class.
*/
Complex_base<scalar> operator*( const Complex_base<scalar> &value) {

    return Complex_base<scalar>(this->real()*value.real() - this->imag()*value.imag(), this->real()*value.imag() + this->imag()*value.real());

};


/**
@brief Override operator * of the STL complex class.
@param value A double valued input.
@return Returns with the calculated value represented by an instance of the Complex_base<scalar> class.
*/
Complex_base<scalar> operator*( const double &value) {

    return Complex_base<scalar>(this->real()*value, this->imag()*value);

};


/**
@brief Override operator * of the STL complex class.
@param value A double valued input.
@return Returns with the calculated value represented by an instance of the Complex_base<scalar> class.
*/
Complex_base<scalar> operator*( const long double &value) {

    return Complex_base<scalar>(this->real()*value, this->imag()*value);

};


/**
@brief Override operator -* of the STL complex class.
@param value A std::complex valued input.
@return Returns with the calculated value represented by an instance of the Complex_base<scalar> class.
*/
Complex_base<scalar> operator-( const std::complex<scalar> &value) {

    return Complex_base<scalar>(this->real() - value.real(), this->imag() - value.imag());

};


/**
@brief Override operator = of the STL complex class.
@param value A std::complex valued input.
*/
void operator=( const std::complex<scalar> &value) {
    this->real(value.real());
    this->imag(value.imag());
    return;

};


/**
@brief Override operator + of the STL complex class.
@param value A Complex_base<scalar> valued input.
*/
void operator=( const Complex_base<scalar> &value) {
    this->real(value.real());
    this->imag(value.imag());
    return;

};

/**
@brief Override operator + of the STL complex class.
@param value A double valued input (would be set as the real part of the complex number).
*/
void operator=( const double &value) {
    this->real(value);
    this->imag(0);
    return;

};



/**
@brief Override operator > of the STL complex class.
@param value A Complex_base<scalar> valued input.
*/
bool operator>( const Complex_base<scalar> &value) {
    if ( (this->real()*this->real() + this->imag()*this->imag()) > (value.real()*value.real() + value.imag()*value.imag()) ) {
        return true;
    }
    else {
        return false;
    }

};



/**
@brief Override operator< of the STL complex class.
@param value A Complex_base<scalar> valued input.
*/
bool operator<( const Complex_base<scalar> &value) {
    if ( (this->real()*this->real() + this->imag()*this->imag()) < (value.real()*value.real() + value.imag()*value.imag()) ) {
        return true;
    }
    else {
        return false;
    }

};

};



template< class scalar >
Complex_base<scalar> operator* (const Complex_base<scalar>& a, const double &b ) {

    return Complex_base<scalar>( a.real()*b, a.imag()*b);

}


template< class scalar >
Complex_base<scalar> operator* (const double &b, const Complex_base<scalar>& a ) {

    return Complex_base<scalar>( a.real()*b, a.imag()*b);

}


/**
@brief Calculates the product of two complex numbers, where one of them is conjugated.
Thus an extra operation of conjugation can be saved in the calculations.
@param a A complex number
@param b A complex number to be conjugated when calculating the product
@return Returns with the calculated product
*/
template<class complex>
inline complex mult_a_bconj( complex &a, complex &b) {
    return complex(a.real()*b.real() + a.imag()*b.imag(), a.imag()*b.real() - a.real()*b.imag() );
}


/// aliasing the representation 16 byte Complex numbers
using Complex16 = Complex_base<double>;

/// aliasing the representation 32 byte Complex numbers
using Complex32 = Complex_base<long double>;

}  //PIC

#endif
