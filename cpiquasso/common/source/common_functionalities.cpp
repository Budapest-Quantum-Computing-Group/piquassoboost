
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
sum( PicState_int64 vec) {

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
@brief Call to calculate the Binomial Coefficient C(n, k)
@param n The integer n
@param k The integer k
@return Returns with the Binomial Coefficient C(n, k).
*/
int binomialCoeff(int n, int k) {
   int C[k+1];
   memset(C, 0, sizeof(C));
   C[0] = 1;
   for (int i = 1; i <= n; i++) {
      for (int j = std::min(i, k); j > 0; j--)
         C[j] = C[j] + C[j-1];
   }
   return C[k];

}



} // PIC

