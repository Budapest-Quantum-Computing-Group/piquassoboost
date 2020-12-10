#ifndef TASK_APPLY_C_AND_G_H
#define TASK_APPLY_C_AND_G_H

#include "matrix.h"
#include "dependency_graph.h"
#include "tbb/tbb.h"


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
