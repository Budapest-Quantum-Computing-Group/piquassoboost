#ifndef INSERT_TRANSFORMED_COLS_H
#define INSERT_TRANSFORMED_COLS_H

#include "matrix.h"
#include "dependency_graph.h"


namespace pic {




/**
@brief Class representing the task E in the apply_to_C_and_G method: insert the transformed columns into a matrix as an adjungate of the supplied rows.
Olny the modes labeled by false in cols_logical are inderted. (these corrensponds to rows that were not transformed.)
*/
class Insert_Transformed_Cols {

protected:
    /// The matrix from which the rows to be transformed should be extracted into continuous memory space.
    matrix rows;
    /// The resulting matrix containing the columns.
    matrix mtx;
    /// A vector of row inices to be extracted
    std::vector<size_t> modes;
    /// logical indexes of the columns of the matrix mtx. True values stand for columns corresponding to modes, and false otherwise.
    bool* cols_logical;
    /// logical vale: set true if the elements inserted to the transformed columns should be conjugated, or false otherwise. (True for matrix C and false for matrix G)
    bool conjugate_elements;


public:

/**
@brief Constructor of the class.
@param rows_in The matrix containing the transformed rows
@param mtx_out The resulting matrix where the columns should be inserted
@param modes_in A vector of indices to be inserted.
@param cols_logical_in Logical values indicating whether the given row was tarnsformed (true) or not (false)
@return Returns with the instance of the class.
*/
Insert_Transformed_Cols( matrix &rows_in, matrix &mtx_out, std::vector<size_t> &modes_in, bool* cols_logical_in, bool conjugate_elements_in);

/**
@brief Operator to insert the columns into the matrix.
@param msg A TBB message firing the node
*/
const tbb::flow::continue_msg & operator()(const tbb::flow::continue_msg &msg);




}; //Insert_Transformed_Cols







} // PIC

#endif
