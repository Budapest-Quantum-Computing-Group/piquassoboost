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
using Complex_Base = std::complex<double>;


/**
@brief Double precision complex number representation used in the CPiquasso package.
This enables better performance than the STL version of the complex number.
Since std::complex<double> is a template class, the implementation of the Complex16 class must be in the header file
*/
class Complex16 : public Complex_Base {



public:

// reusing the constructors of the parent class
using Complex_Base::Complex_Base;


/**
@brief Override operator / of the STL complex class.
@param value A double valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
Complex16 operator/( const double &value) {

    return Complex16(this->real()/value, this->imag()/value);

};


/**
@brief Override operator / of the STL complex class.
@param value A Complex16 valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
Complex16 operator/( const Complex16 &value) {

    double norm_input = value.real()*value.real() + value.imag()*value.imag();

    return Complex16((this->real()*value.real() + this->imag()*value.imag())/norm_input, (this->imag()*value.real() - this->real()*value.imag())/norm_input);

};


/**
@brief Override operator + of the STL complex class.
@param value A Complex_Base valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
Complex16 operator+( const Complex_Base &value) {

    return Complex16(this->real() + value.real(), this->imag() + value.imag());

};

/**
@brief Override operator + of the STL complex class.
@param value A Complex16 valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
Complex16 operator+( const Complex16 &value) {

    return Complex16(this->real() + value.real(), this->imag() + value.imag());

};


/**
@brief Override operator * of the STL complex class.
@param value A Complex_Base valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
Complex16 operator*( const Complex_Base &value) {

    return Complex16(this->real()*value.real() - this->imag()*value.imag(), this->real()*value.imag() + this->imag()*value.real());

};

/**
@brief Override operator * of the STL complex class.
@param value A Complex16 valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
Complex16 operator*( const Complex16 &value) {

    return Complex16(this->real()*value.real() - this->imag()*value.imag(), this->real()*value.imag() + this->imag()*value.real());

};

/**
@brief Override operator * of the STL complex class.
@param value A double valued input.
@return Returns with the calculated value represented by an instance of the Complex16 class.
*/
Complex16 operator*( const double &value) {

    return Complex16(this->real()*value, this->imag()*value);

};

/**
@brief Override operator = of the STL complex class.
@param value A Complex_Base valued input.
*/
void operator=( const Complex_Base &value) {
    this->real(value.real());
    this->imag(value.imag());
    return;

};


/**
@brief Override operator + of the STL complex class.
@param value A Complex16 valued input.
*/
void operator=( const Complex16 &value) {
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


inline Complex16 mult_a_bconj( Complex16 &a, Complex16 &b) {
    return Complex16(a.real()*b.real() + a.imag()*b.imag(), a.imag()*b.real() - a.real()*b.imag() );
}


}  //PIC

#endif
