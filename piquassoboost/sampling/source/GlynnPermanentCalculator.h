#ifndef GlynnPermanentCalculator_H
#define GlynnPermanentCalculator_H

#include "matrix.h"
#include "PicState.h"
#include <vector>
#include "PicVector.hpp"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif


namespace pic {

/**
@brief Call to print the elements of a container
@param vec a container
@return Returns with the sum of the elements of the container
*/
template <typename Container>
void print_state( Container state );


/**
@brief Class representing a matrix Chin-Huh permanent calculator
*/
class GlynnPermanentCalculator {

protected:
    /// The effective scattering matrix of a boson sampling instance
    matrix mtx;
public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculator();



/**
@brief Call to calculate the permanent of the effective scattering matrix. Assuming that input state, output state and the matrix are
defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
permanent of an effective scattering matrix related to probability of obtaining output state from given
input state.
@param mtx_in The effective scattering matrix of a boson sampling instance
@param input_state_in The input state
@param output_state_in The output state
@return Returns with the calculated permanent
*/
Complex16 calculate(matrix &mtx);





}; //GlynnPermanentCalculator






} // PIC

#endif
