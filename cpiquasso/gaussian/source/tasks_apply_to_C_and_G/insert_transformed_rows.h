#ifndef INSERT_TRANSFORMED_ROWS_H
#define INSERT_TRANSFORMED_ROWS_H

#include "matrix.h"
#include "dependency_graph.h"


namespace pic {




/**
@brief Class representing the task F in the apply_to_C_and_G method: insert the transformed rows into the matrix.
*/
class Insert_Transformed_Rows {

protected:
    /// The matrix containing the transformed rows (T @ mtx(modes,:)  )
    matrix rows;
    /// The matrix containing the transformed elements (T @ mtx(modes,modes) @ T^T)
    matrix corner;
    /// The resulting transformed matrix containing all the transformed modes
    matrix mtx;
    /// A vector contaning the modes to be transformed
    std::vector<size_t> modes;
    /// logical indexes of the columns of the matrix mtx. True values stand for columns corresponding to modes, and false otherwise.
    bool* cols_logical;

public:

/**
@brief Constructor of the class.
@param rows_in The matrix containing the transformed rows (T @ mtx(modes,:)  )
@param corner_in The matrix containing the transformed elements (T @ mtx(modes,modes) @ T^T)
@param mtx_out The resulting transformed matrix containing all the transformed modes
@param modes_in A vector of row indices that were transformed.
@return Returns with the instance of the class.
*/
Insert_Transformed_Rows( matrix &rows_in,matrix &corner_in, matrix &mtx_out, std::vector<size_t> &modes_in, bool* cols_logical_in);

/**
@brief Operator to insert the transformed rows into the matrix
@param msg A TBB message firing the node
*/
const tbb::flow::continue_msg & operator()(const tbb::flow::continue_msg &msg);




}; //Insert_Transformed_Rows


} // PIC

#endif
