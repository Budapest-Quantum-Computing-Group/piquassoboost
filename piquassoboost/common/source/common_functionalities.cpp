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

#include <vector>

#include "common_functionalities.h"




namespace pic {



/**
@brief Function to calculate factorial of a number.
@param n The input number
@return Returns with the factorial of the number
*/
double factorial(int64_t n) {



    if ( n == 0 ) return 1;
    if ( n == 1 ) return 1;

    int64_t ret=1;

    for (int64_t idx=2; idx<=n; idx++) {
        ret = ret*idx;
    }

    return (double) ret;


}


/**
@brief Calculates the n-th power of 2.
@param n An natural number
@return Returns with the n-th power of 2.
*/
unsigned long long power_of_2(unsigned long long n) {
  if (n == 0) return 1;
  if (n == 1) return 2;

  return 2 * power_of_2(n-1);
}



/**
@brief Call to calculate sum of integers stored in a container
@param vec a container of integers
@return Returns with the sum of the elements of the container
*/
int sum( PicVector<int> vec) {

    int ret = 0;
    for (auto it=vec.begin(); it!=vec.end(); it++) {
        if ( *it == 0) {
            continue;
        }
        ret = ret + *it;
    }
    return ret;
}



/**
@brief Call to calculate sum of integers stored in a container
@param vec a PicState_int64 instance
@return Returns with the sum of the elements of the container
*/
int
sum( const PicState_int64& vec) {

    int ret = 0;
    for (size_t idx=0; idx<vec.size(); idx++) {
        if ( vec[idx] == 0) {
            continue;
        }
        ret = ret + vec[idx];
    }
    return ret;
}



/**
@brief Call to calculate the Binomial Coefficient C(n, k) templated version
@param n The integer n
@param k The integer k
@return Returns with the Binomial Coefficient C(n, k).
*/
template <typename int_type>
int_type binomialCoeffTemplated(int n, int k) {
   int_type C[k+1];
   memset(C, 0, sizeof(C));
   C[0] = 1;
   for (int i = 1; i <= n; i++) {
      for (int j = std::min(i, k); j > 0; j--)
         C[j] = C[j] + C[j-1];
   }
   return C[k];
}


inline int binomialCoeff(int n, int k){
    return binomialCoeffTemplated<int>(n, k);
}


inline int64_t binomialCoeffInt64(int n, int k){
    return binomialCoeffTemplated<int64_t>(n, k);
}


/**
@brief Function which checks whether the given matrix is symmetric or not.
@param mtx_in The given matrix.
@param tolerance The tolerance value for being 2 different values equal.
@return True if the @p mtx_in is symmetric and false otherwise.
*/
bool isSymmetric( matrix mtx_in, double tolerance ){
    if (mtx_in.rows != mtx_in.cols){
        return false;
    }
    const size_t dim = mtx_in.rows;

    for (size_t row_idx = 0; row_idx < dim; row_idx++){
        for (size_t col_idx = row_idx + 1; col_idx < dim; col_idx++){
            Complex16 diff = mtx_in[row_idx * dim + col_idx] - mtx_in[col_idx * dim + row_idx];
            if (std::abs(diff) > tolerance){
                return false;
            }
        }
    }
    return true;
}


/**
@brief Function which checks whether the given matrix is hermitian or not.
@param mtx_in The given matrix.
@param tolerance The tolerance value for being 2 different values equal.
@return True if the @p mtx_in is hermitian and false otherwise.
*/
bool isHermitian( matrix mtx_in, double tolerance ){
    if (mtx_in.rows != mtx_in.cols){
        return false;
    }
    const size_t dim = mtx_in.rows;

    for (size_t row_idx = 0; row_idx < dim; row_idx++){
        for (size_t col_idx = row_idx; col_idx < dim; col_idx++){
            Complex16 diff = mtx_in[row_idx * dim + col_idx] - std::conj(mtx_in[col_idx * dim + row_idx]);
            if (std::abs(diff) > tolerance){
                return false;
            }
        }
    }
    return true;
}



} // PIC

