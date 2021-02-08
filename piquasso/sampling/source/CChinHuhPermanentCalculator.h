#ifndef CChinHuhPermanentCalculator_H
#define CChinHuhPermanentCalculator_H

#include "matrix.h"
#include "PicState.h"
#include <vector>
#include "PicVector.hpp"


namespace pic {

/**
@brief Call to print the elements of a container
@param vec a container
@return Returns with the sum of the elements of the container
*/
template <typename Container>
void print_state( Container state );

/**
@brief Call to calculate sum of integers stored in a container
@param vec a container if integers
@return Returns with the sum of the elements of the container
*/
template <typename scalar>
int sum( PicVector<scalar> vec);

/**
@brief Call to calculate sum of integers stored in a container
@param vec a PicState_int64 instance
@return Returns with the sum of the elements of the container
*/
int sum( PicState_int64 vec);

/**
@brief Call to calculate the Binomial Coefficient C(n, k)
@param n The integer n
@param k The integer k
@return Returns with the Binomial Coefficient C(n, k).
*/
int binomialCoeff(int n, int k);

/**
@brief Class representing a matrix Chin-Huh permanent calculator
*/
class CChinHuhPermanentCalculator {

protected:
    /// The effective scattering matrix of a boson sampling instance
    matrix mtx;
    /// The input state
    PicState_int64 input_state;
    /// The output state
    PicState_int64 output_state;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
CChinHuhPermanentCalculator();

/**
@brief Constructor of the class.
@param mtx_in The effective scattering matrix of a boson sampling instance
@param input_state_in The input state
@param output_state_in The output state
@return Returns with the instance of the class.
*/
CChinHuhPermanentCalculator( matrix &mtx_in, PicState_int64 &input_state_in, PicState_int64 &output_state_in);

/**
@brief Call to update the memroy addresses of the stored matrices
@param mtx_in The effective scattering matrix of a boson sampling instance
@param input_state_in The input state
@param output_state_in The output state
*/
void Update( matrix &mtx_in, PicState_int64 & input_state_in, PicState_int64 &output_state_in);


/**
@brief Call to calculate the permanent of the effective scattering matrix. Assuming that input state, output state and the matrix are
defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
permanent of an effective scattering matrix related to probability of obtaining output state from given
input state.
@return Returns with the calculated permanent
*/
Complex16 calculate();


/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void Update_mtx( matrix &mtx_in);


/**
@brief Call to update the memory address of the input_state
@param input_state_in The input state
*/
void Update_input_state(PicState_int64 &input_state_in);


/**
@brief Call to update the memory address of the output_state
@param output_state_in The output state
*/
void Update_output_state(PicState_int64 &moutput_state_in);


}; //CChinHuhPermanentCalculator


} // PIC

#endif
