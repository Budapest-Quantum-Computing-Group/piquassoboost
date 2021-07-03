#ifndef EXTRACT_ROWS_H
#define EXTRACT_ROWS_H

#include "matrix.h"
#include "dependency_graph.h"


namespace pic {

/**
@brief Class representing the task A in the apply_to_C_and_G method: extract the rows to be transformed from the given matrix.
*/
class Extract_Rows {

protected:
    /// The matrix from which the rows to be transformed should be extracted into continuous memory space.
    matrix mtx;
    /// raw pointer to the stored data in matrix mtx
    Complex16* mtx_data;
    /// The resulting matrix containing the rows.
    matrix rows;
    /// raw pointer to the stored data in matrix rows
    Complex16* rows_data;
    /// A vector of row indices to be extracted
    std::vector<size_t> modes;

public:

/**
@brief Nullary of the class.
@return Returns with the instance of the class.
*/
Extract_Rows();

/**
@brief Constructor of the class.
@param mtx_in The matrix from which the rows corresponding to modes should be extracted into a continuous memory space.
@param rows_out The resulting matrix.
@param modes_in A vector of row indices to be extracted
@return Returns with the instance of the class.
*/
Extract_Rows( matrix &mtx_in, matrix &rows_out, std::vector<size_t> &modes_in);

/**
@brief Operator to extract the rows from the matrix
@param msg A TBB message fireing the node
*/
const tbb::flow::continue_msg & operator()(const tbb::flow::continue_msg &msg);


}; //Extract_Rows


} // PIC

#endif
