#ifndef TRANSFORM_COLS_H
#define TRANSFORM_COLS_H

#include "matrix.h"
#include "dependency_graph.h"


namespace pic {


/**
@brief Class representing the task D in the apply_to_C_and_G method: transforms the extracted columns by transformation matrix T.
*/
class Transform_Cols {

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
Transform_Cols( matrix &mtx_in, matrix &T_in );

/**
@brief Operator to transform the columns of the input matrix
@param msg A TBB message firing the node
*/
const tbb::flow::continue_msg & operator()(const tbb::flow::continue_msg &msg);




}; //Transform_Cols


} // PIC

#endif
