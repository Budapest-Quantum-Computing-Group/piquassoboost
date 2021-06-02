
#ifndef EXTRACT_CORNER_H
#define EXTRACT_CORNER_H

#include "matrix.h"
#include "dependency_graph.h"


namespace pic {

/**
@brief Class representing the task C in the apply_to_C_and_G method: extract the columns of a given matrix corresponding to the given indices.
*/
class Extract_Corner {

protected:
    /// The matrix from which the rows to be transformed should be extracted into continuous memory space.
    matrix mtx;
    /// The resulting matrix containing the columns.
    matrix cols;
    /// A vector of col indices to be extracted
    std::vector<size_t> modes;
    /// logical indexes of the columns of the matrix mtx. True values stand for columns corresponding to modes, and false otherwise.
    bool* cols_logical;

public:

/**
@brief Constructor of the class.
@param mtx_in The matrix from which the columns corresponding to modes should be extracted into a continuous memory space.
@param cols_out The resulting matrix containing the selected columns
@param modes_in A vector of row indecis to be extracted
@param cols_logical_in Preallocated array for the logical indices.
@return Returns with the instance of the class.
*/
Extract_Corner( matrix &mtx_in, matrix &cols_out, std::vector<size_t> &modes, bool* cols_logical_in);

/**
@brief Operator to extract the columns from the matrix.
@param msg A TBB message firing the node
*/
const tbb::flow::continue_msg & operator()(const tbb::flow::continue_msg &msg);




}; //Extract_Corner




} // PIC

#endif
