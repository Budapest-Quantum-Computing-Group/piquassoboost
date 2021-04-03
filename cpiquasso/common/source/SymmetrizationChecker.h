#ifndef SYMMETRIZATION_CHECKER_H
#define SYMMETRIZATION_CHECKER_H

#include "matrix.h"

#include <tbb/tbb.h>

namespace pic {

/**
@brief Class representing symmetrization check of a given input matrix.
*/
class SymmetrizationChecker {

protected:
    /// The matrix under investigation of symmetrization check
    matrix mtx;
    /// raw pointer to result of the check
    int* result;

public:

/**
@brief Constructor of the class. Sets the value of the result_out to 1. 
@param mtx_in The matrix under investigation of symmetrization check.
@param result_out The outcome of the investigation: 0 if not symmetric, 1 if symmetric.
@return Returns with the instance of the class.
*/
SymmetrizationChecker( matrix &mtx_in, int* result_out);

/**
@brief Operator to extract the rows from the matrix
@param msg A TBB message fireing the node
*/
const tbb::flow::continue_msg&
operator()(const tbb::flow::continue_msg &msg);


}; //SymmetrizationChecker


} // PIC

#endif // SYMMETRIZATION_CHECKER_H
