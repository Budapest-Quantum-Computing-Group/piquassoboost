#ifndef PicTypes_H
#define PicTypes_H

#include <assert.h>
#include <cstddef>
#include <complex>
#include <cmath>
#include <immintrin.h>

// platform independent types
#include <stdint.h>

#ifndef CACHELINE
#define CACHELINE 64
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#if BLAS==0 // undefined blas
    /// Set the number of threads on runtime in MKL
    void omp_set_num_threads(int num_threads);
    /// get the number of threads in MKL
    int omp_get_max_threads();
#elif BLAS==1 // MKL
    /// Set the number of threads on runtime in MKL
    void MKL_Set_Num_Threads(int num_threads);
    /// get the number of threads in MKL
    int mkl_get_max_threads();
#elif BLAS==2 // OpenBLAS
    /// Set the number of threads on runtime in OpenBlas
    void openblas_set_num_threads(int num_threads);
    /// get the number of threads in OpenBlas
    int openblas_get_num_threads();
#endif

#ifdef __cplusplus
}
#endif


/// The namespace of the Picasso project
namespace pic {


/// @brief Alias for the standard STL complex type
//using std::complex<scalar> = std::complex<scalar>;


/**
@brief Double precision complex number representation used in the CPiquasso package.
This enables better performance than the STL version of the complex number.
Since std::complex<double> is a template class, the implementation of the Complex16 class must be in the header file
*/
template<class scalar>
class Complex_base : public std::complex<scalar> {



public:

// reusing the constructors of the parent class
using std::complex<scalar>::complex;


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

};




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
