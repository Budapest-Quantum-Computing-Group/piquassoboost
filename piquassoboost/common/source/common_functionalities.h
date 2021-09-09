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
int sum( PicState_int64 vec);


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
@brief Function which checks whether the given matrix is symmetric or not.
@param mtx_in The given matrix.
@param tolerance The tolerance value for being 2 different values equal.
@return True if the @p mtx_in is symmetric and false otherwise.
*/
bool isSymmetric( matrix mtx_in, double tolerance );


/**
@brief Function which checks whether the given matrix is hermitian or not.
@param mtx_in The given matrix.
@param tolerance The tolerance value for being 2 different values equal.
@return True if the @p mtx_in is hermitian and false otherwise.
*/
bool isHermitian( matrix mtx_in, double tolerance );




/**
@brief Function which performs the operation \f$ ret = sum_i a_i*b_i \f$ of two complex vectors with AVX2 instruction set if available.
The number of elements in the vectors should be even, and the vectors should be aligned at 32bit boundaries.
@param a Pointer to the first vector
@param b Pointer to the second vector
@param element_num the number of elements;
@return The calculated sum
*/
static inline void vector_dot_vector( Complex16* a, Complex16* b, size_t element_num, Complex16& ret )

{


    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

    for(size_t kdx = 0; kdx<element_num; kdx=kdx+2) {
        __m256d a_vec = _mm256_load_pd((double*)(a + kdx));
        __m256d b_vec = _mm256_load_pd((double*)(b + kdx));

        // Multiply elements of a_vec and b_vec
        __m256d vec3 = _mm256_mul_pd(a_vec, b_vec);

        // Switch the real and imaginary elements of b_vec
        b_vec = _mm256_permute_pd(b_vec, 0x5);

        // Negate the imaginary elements of b_vec
        b_vec = _mm256_mul_pd(b_vec, neg);

        // Multiply elements of a_vec and the modified b_vec
        __m256d vec4 = _mm256_mul_pd(a_vec, b_vec);

        // Horizontally subtract the elements in vec3 and vec4
        b_vec = _mm256_hsub_pd(vec3, vec4);

        ret = ret + *((Complex16*)&b_vec[0]) + *((Complex16*)&b_vec[2]);


    }


/*
    for (size_t kdx=0; kdx<element_num; kdx++) {
        tmp += a[kdx] * b[kdx];
    }
*/
    return;

};

} // PIC

#endif // COMMON_FUNCTIONALITIES_H
