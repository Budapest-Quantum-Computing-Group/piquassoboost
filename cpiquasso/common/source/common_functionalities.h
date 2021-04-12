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



} // PIC

#endif // COMMON_FUNCTIONALITIES_H
