#include <iostream>
#include "GlynnPermanentCalculator.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>

namespace pic {




/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculator::GlynnPermanentCalculator() {}




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
Complex16
GlynnPermanentCalculator::calculate(matrix &mtx) {


    


    return Complex16(0.0,0.0);


}









} // PIC
