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


} // PIC

#endif // COMMON_FUNCTIONALITIES_H
