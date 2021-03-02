
#ifndef TRANSFORM_ROWS_H
#define TRANSFORM_ROWS_H

#include "matrix.h"
#include "dependency_graph.h"


namespace pic {

/**
@brief Class representing the task B in the apply_to_C_and_G method: transforms the extracted rows by transformation matrix T.
*/
class Transform_Rows {

protected:
    /// The matrix to be transformed
    matrix mtx;
    /// raw pointer to the stored data in matrix mtx
    Complex16* mtx_data;
    /// The matrix of the transformation
    matrix T;
    /// raw pointer to the stored data in matrix T
    Complex16* T_data;

public:

/**
@brief Constructor of the class.
@param mtx_in The matrix to be transformed.
@param T_in The matrix of the transformation
@return Returns with the instance of the class.
*/
Transform_Rows( matrix &T_in, matrix &mtx_in);

/**
@brief Operator to transform the rows of the input matrix
@param msg A TBB message firing the node
*/
const tbb::flow::continue_msg & operator()(const tbb::flow::continue_msg &msg);




}; //Transform_Rows




} // PIC

#endif
