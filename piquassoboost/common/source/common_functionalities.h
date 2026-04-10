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

#ifndef COMMON_FUNCTIONALITIES_H
#define COMMON_FUNCTIONALITIES_H


#include "PicVector.hpp"
#include "PicState.h"
#include "matrix.h"

namespace pic {


/**
@brief Function to calculate factorial of a number.
@param n The input number
@return Returns with the factorial of the number
*/
double factorial(int64_t n);


/**
@brief Calculates the n-th power of 2.
@param n An natural number
@return Returns with the n-th power of 2.
*/
unsigned long long power_of_2(unsigned long long n);


/**
@brief Call to calculate sum of integers stored in a container
@param vec a container of integers
@return Returns with the sum of the elements of the container
*/
int sum( PicVector<int> vec);


/**
@brief Call to calculate sum of integers stored in a container
@param vec a PicState_int64 instance
@return Returns with the sum of the elements of the container
*/
int64_t sum( const PicState_int64& vec);

/**
@brief Call to calculate sum of integers stored in a container
@param vec a PicState_int64 instance
@return Returns with the sum of the elements of the container
*/
int sum( const PicState_int& vec);


// export the template functions with the template parameters
// Do we need this??
//template
//int sum<int>( PicVector<int> vec );


/**
@brief Call to calculate the Binomial Coefficient C(n, k)
@param n The integer n
@param k The integer k
@return Returns with the Binomial Coefficient C(n, k).
*/
int binomialCoeff(int n, int k);


/**
@brief Call to calculate the Binomial Coefficient C(n, k) in int64_t
@param n The integer n
@param k The integer k
@return Returns with the Binomial Coefficient C(n, k).
*/
int64_t binomialCoeffInt64(int n, int k);


/**
@brief Free externally handed-off aligned matrix/state data in the native library.
@param ptr The data pointer to release.
*/
void free_external_data(void* ptr);
  

// GCC/Clang provide __int128 natively; MSVC does not.
// On MSVC we emulate a 128-bit unsigned integer with two uint64_t fields
// (lo = bits 0-63, hi = bits 64-127) and provide all operators required by
// binomialCoeffTemplated and BBFGPermanentCalculatorRepeated:
//   construction from int, operator+/+=, operator*=, and explicit casts to
//   double/long double.  Multiplication uses _umul128 for carry-correct
//   64×64→128-bit partial products.
#ifdef _MSC_VER
#include <intrin.h>       // _umul128
#include <stdint.h>
#include <type_traits>    // std::is_signed
#ifndef PIQ_UINT128_DEFINED
#define PIQ_UINT128_DEFINED
struct piq_uint128 {
    uint64_t lo;  ///< bits  0–63
    uint64_t hi;  ///< bits 64–127

    // --- construction -------------------------------------------------------
    piq_uint128() : lo(0), hi(0) {}

    // Construct from any integral type.  For signed types a negative value is
    // sign-extended into hi; for unsigned types hi is always zero.
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    piq_uint128(T v) : lo(static_cast<uint64_t>(v)),
                       hi(std::is_signed<T>::value && static_cast<int64_t>(v) < 0
                          ? ~uint64_t(0) : 0) {}

    // --- assignment from integral type --------------------------------------
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    piq_uint128& operator=(T v) {
        lo = static_cast<uint64_t>(v);
        hi = (std::is_signed<T>::value && static_cast<int64_t>(v) < 0) ? ~uint64_t(0) : 0;
        return *this;
    }

    // --- addition -----------------------------------------------------------
    piq_uint128& operator+=(const piq_uint128& rhs) {
        const uint64_t old_lo = lo;
        lo  += rhs.lo;
        hi  += rhs.hi + (lo < old_lo ? uint64_t(1) : uint64_t(0));  // carry
        return *this;
    }

    piq_uint128 operator+(const piq_uint128& rhs) const {
        piq_uint128 r(*this);
        r += rhs;
        return r;
    }

    // --- multiplication (truncated to 128 bits) -----------------------------
    // (hi_a·2^64 + lo_a)·(hi_b·2^64 + lo_b)  mod 2^128
    //   = lo_a·lo_b  +  (lo_a·hi_b + hi_a·lo_b)·2^64   (mod 2^128)
    piq_uint128& operator*=(const piq_uint128& rhs) {
        const uint64_t a_lo = lo, a_hi = hi;
        const uint64_t b_lo = rhs.lo, b_hi = rhs.hi;
        uint64_t prod_hi;
        lo  = _umul128(a_lo, b_lo, &prod_hi);   // full 128-bit product of low halves
        hi  = prod_hi;
        hi += a_lo * b_hi;   // cross term: only contributes to hi word
        hi += a_hi * b_lo;   // cross term: only contributes to hi word
        // a_hi·b_hi contributes at 2^128 and above — discard (mod 2^128)
        return *this;
    }

    piq_uint128 operator*(const piq_uint128& rhs) const {
        piq_uint128 r(*this);
        r *= rhs;
        return r;
    }

    // --- division by a small positive 64-bit integer -------------------------
    // Used in the running binomial-coefficient update in BBFGPermanentCalculator
    // Repeated: result * factor / divisor, where divisor is always an exact
    // divisor of the numerator (no rounding).  Uses two-step long division in
    // base 2^64:  q_hi = hi/d, rem = hi%d,  q_lo = udiv128(rem, lo, d).
    piq_uint128& operator/=(uint64_t d) {
        uint64_t rem;
        uint64_t q_hi = hi / d;
        uint64_t r_hi = hi % d;
        lo = _udiv128(r_hi, lo, d, &rem);
        hi = q_hi;
        return *this;
    }
    piq_uint128  operator/ (uint64_t d) const { piq_uint128 r(*this); r /= d; return r; }
    // Allow signed int divisors (always positive in practice)
    piq_uint128& operator/=(int d) { return operator/=(static_cast<uint64_t>(d)); }
    piq_uint128  operator/ (int d) const { piq_uint128 r(*this); r /= d; return r; }

    // --- conversion to floating-point (for (precision_type) cast) -----------
    // 2^64 as a compile-time constant avoids runtime integer→float conversion.
    static constexpr double     k2_64d  = 18446744073709551616.0;
    static constexpr long double k2_64ld = 18446744073709551616.0L;

    explicit operator double()      const { return hi * k2_64d  + static_cast<double>(lo);      }
    explicit operator long double() const { return hi * k2_64ld + static_cast<long double>(lo); }
    explicit operator float()       const { return static_cast<float>(static_cast<double>(*this)); }
};
#endif // PIQ_UINT128_DEFINED
typedef piq_uint128 piq_int128_t;
#else
typedef __int128 piq_int128_t;
#endif

/**
@brief Call to calculate the Binomial Coefficient C(n, k) in __int128
@param n The integer n
@param k The integer k
@return Returns with the Binomial Coefficient C(n, k).
*/
piq_int128_t binomialCoeffInt128(int n, int k);


/**
@brief Function which checks whether the given matrix is symmetric or not.
@param mtx_in The given matrix.
@param tolerance The tolerance value for being 2 different values equal.
@return True if the @p mtx_in is symmetric and false otherwise.
*/
template<typename scalar>
bool isSymmetric( matrix_base<scalar> mtx_in);
template<typename scalar>
bool isSymmetric( matrix_base<scalar> mtx_in, double tolerance );


/**
@brief Function which checks whether the given matrix is hermitian or not.
@param mtx_in The given matrix.
@param tolerance The tolerance value for being 2 different values equal.
@return True if the @p mtx_in is hermitian and false otherwise.
*/
bool isHermitian( matrix mtx_in, double tolerance );


} // PIC

#endif // COMMON_FUNCTIONALITIES_H
