#include <iostream>
#include <tbb/tbb.h>
#include "tasks_apply_to_C_and_G.h"
#include "matrix.h"
#include "dot.h"
#include <memory.h>

namespace pic {


/**
@brief Nullary of the class.
@return Returns with the instance of the class.
*/
Extract_Rows::Extract_Rows() {

    rows_data = NULL;
    mtx_data = NULL;

}

/**
@brief Constructor of the class.
@param mtx_in The matrix from which the rows corresponding to modes should be extracted into a continuous memory space.
@param rows_out The resulting matrix.
@param modes_in A vector of row indices to be extracted
@return Returns with the instance of the class.
*/
Extract_Rows::Extract_Rows( matrix &mtx_in, matrix &rows_out, std::vector<size_t> &modes_in) {

    mtx = mtx_in;
    rows = rows_out;
    modes = modes_in;

    rows_data = rows.get_data();
    mtx_data = mtx.get_data();
}

/**
@brief Operator to extract the rows from the matrix
@param msg A TBB message firing the node
*/
const tbb::flow::continue_msg &
Extract_Rows::operator()(const tbb::flow::continue_msg &msg) {

#ifdef DEBUG
    std::cout << "Tasks apply_to_C_and_G: extracting rows" << std::endl;
#endif

    for( size_t i=0; i<modes.size(); i++ ) {
        size_t rows_offset = i*(mtx.cols);
        size_t mtx_offset = (modes[i])*(mtx.cols);
        memcpy( rows_data+rows_offset, mtx_data+mtx_offset, (mtx.cols)*sizeof(Complex16));
    }


    return msg;

}








/**
@brief Constructor of the class.
@param mtx_in The matrix to be transformed.
@param T_in The matrix of the transformation
@return Returns with the instance of the class.
*/
Transform_Rows::Transform_Rows( matrix &T_in, matrix &mtx_in ) {

    mtx = mtx_in;
    T = T_in;

    T_data = T.get_data();
    mtx_data = mtx.get_data();


}

/**
@brief Operator to extract a row labeled by i-th element of modes.
@param msg A TBB message fireing the node
*/
const tbb::flow::continue_msg &
Transform_Rows::operator()(const tbb::flow::continue_msg &msg) {

#ifdef DEBUG
    std::cout << "Tasks apply_to_C_and_G: transforming rows" << std::endl;
#endif

    // calculating the product T*mtx
    matrix dot_res = dot( T, mtx );

    // copy the result into the input matrix
    memcpy(mtx_data, dot_res.get_data(), mtx.rows*mtx.cols*sizeof(Complex16));

    return msg;

}




/**
@brief Constructor of the class.
@param mtx_in The matrix from which the columns corresponding to modes should be extracted into a continuous memory space.
@param cols_out The resulting matrix containing the selected columns
@param modes_in A vector of row indecis to be extracted
@param cols_logical_in Preallocated array for the logical indices.
@return Returns with the instance of the class.
*/
Extract_Corner::Extract_Corner( matrix &mtx_in, matrix &cols_out, std::vector<size_t> &modes_in, bool* cols_logical_in) {

    mtx = mtx_in;
    cols = cols_out;
    modes = modes_in;

    cols_logical = cols_logical_in;

}

/**
@brief Operator to extract the columns from the matrix.
@param msg A TBB message firing the node
*/
const tbb::flow::continue_msg &
Extract_Corner::operator()(const tbb::flow::continue_msg &msg) {

#ifdef DEBUG
    std::cout << "Tasks apply_to_C_and_G: extracting columns" << std::endl;
#endif




    // raw pointer to the stored data in matrix mtx
    Complex16* mtx_data = mtx.get_data();
    // raw pointer to the stored data in matrix rows
    Complex16* cols_data = cols.get_data();

    // number of columns in the matrix from we want to extract columns
    size_t col_num = mtx.cols;

    if (cols_logical != NULL) {
        memset(cols_logical, 0, col_num*sizeof(bool));
    }

    size_t transform_col_idx = 0;
    size_t transform_col_num = modes.size();
    size_t col_range = 1;
    // loop over the col indices to be transformed (indices are stored in attribute modes)
    while (true) {

        // condition to exit the loop: if there are no further columns then we exit the loop
        if ( transform_col_idx >= transform_col_num) {
            break;
        }

        // determine contiguous memory slices (column indices) to be transformed in the rows
        while (true) {

            // condition to exit the loop: if the difference of successive indices is greater than 1, the end of the contiguous memory slice is determined
            if ( transform_col_idx+col_range >= transform_col_num || modes[transform_col_idx+col_range] - modes[transform_col_idx+col_range-1] != 1 ) {
                break;
            }
            else {
                if (transform_col_idx+col_range+1 >= transform_col_num) {
                    break;
                }
                col_range = col_range + 1;

            }

        }

        // the column index in the matrix from we are bout the extract columns to be transformed
        size_t col_idx = modes[transform_col_idx];

        // row-wise parallelized loop to extract the columns to be transformed
        size_t N = transform_col_num;
        tbb::parallel_for((size_t)0, N, (size_t)1, [mtx_data, cols_data, &col_idx, &transform_col_idx, &col_range, &col_num, &transform_col_num](size_t i) {
            size_t mtx_offset = i*col_num + col_idx;
            size_t cols_offset = i*transform_col_num + transform_col_idx;
            memcpy(cols_data+cols_offset, mtx_data+mtx_offset, col_range*sizeof(Complex16));
        }); // TBB


        // setting the logical column indices (the columns to be transformed are set to true)
        if (cols_logical != NULL) {
            memset(cols_logical+modes[transform_col_idx], 1, col_range*sizeof(bool));
        }

        transform_col_idx = transform_col_idx + col_range;
        col_range = 1;

    }


    return msg;
}





/**
@brief Constructor of the class.
@param mtx_in The matrix to be transformed.
@param T_in The matrix of the transformation
@return Returns with the instance of the class.
*/
Transform_Cols::Transform_Cols( matrix &mtx_in, matrix &T_in ) {

    mtx = mtx_in;
    T = T_in;

    T_data = T.get_data();
    mtx_data = mtx.get_data();


}

/**
@brief Operator to transform the columns of the matrix mtx
@param msg A TBB message firing the node
*/
const tbb::flow::continue_msg &
Transform_Cols::operator()(const tbb::flow::continue_msg &msg) {

#ifdef DEBUG
    std::cout << "Tasks apply_to_C_and_G: transforming columns" << std::endl;
#endif

    // calculating the product mtx*T
    matrix dot_res = dot( mtx, T );

    // copy the result into the input matrix
    memcpy(mtx_data, dot_res.get_data(), mtx.rows*mtx.cols*sizeof(Complex16));


    return msg;

}




/**
@brief Constructor of the class.
@param rows_in The matrix containing the transformed rows
@param mtx_out The resulting matrix where the columns should be inserted
@param modes_in A vector of indices to be inserted.
@param cols_logical_in Logical values indicating whether the given row was tarnsformed (true) or not (false)
@return Returns with the instance of the class.
*/
Insert_Transformed_Cols::Insert_Transformed_Cols( matrix &rows_in, matrix &mtx_out, std::vector<size_t> &modes_in, bool* cols_logical_in, bool conjugate_elements_in) {

    rows = rows_in;
    mtx = mtx_out;

    modes = modes_in;
    cols_logical = cols_logical_in;
    conjugate_elements = conjugate_elements_in;

}

/**
@brief Operator to insert the columns into the matrix.
@param msg A TBB message firing the node
*/
const tbb::flow::continue_msg &
Insert_Transformed_Cols::operator()(const tbb::flow::continue_msg &msg) {


#ifdef DEBUG
    std::cout << "Tasks apply_to_C_and_G: inserting transformed columns from previously transformed rows" << std::endl;
#endif

    // raw pointer to the stored data in matrix mtx
    Complex16* mtx_data = mtx.get_data();
    // raw pointer to the stored data in matrix rows
    Complex16* rows_data = rows.get_data();

    // A vector of row inices to be extracted
    std::vector<size_t> &modes_loc = modes;
    // logical indexes of the columns of the matrix mtx. True values stand for columns corresponding to modes, and false otherwise.
    bool* cols_logical_loc = cols_logical;
    bool conjugate_elements_loc = conjugate_elements;
    // number of columns
    size_t col_num = rows.cols;

    // parallel for to insert the transformed columns into matrix mtx
    int N = (int)(modes.size());
    tbb::parallel_for(0, N, 1, [rows_data, mtx_data, modes_loc, cols_logical_loc, col_num, conjugate_elements_loc](int i) {

        // offset for the i-th row in the matrix rows
        int row_offset = i*col_num;
        Complex16* row_data = rows_data + row_offset;

        // the column index in the matrix where the elements are copied
        size_t mtx_col_idx = modes_loc[i];

        // inserting the complex conjugate of the row elements into the columns
        for ( size_t col_idx=0; col_idx<col_num; col_idx++) {

            // only those elements are filled, that are not associated with transformed rows (these can be done by the insertion of the transformed rows)
            if (cols_logical_loc[col_idx]) {
                //std::cout<< "skipping row " << col_idx << std::endl;
                continue;
            }

            // the offset of the given row in matrix where the elements are copied
            size_t mtx_row_offset = col_idx*col_num;

            // insert the conjugate transposition of the transformed rows in place of the transformed columns
            Complex16 element = row_data[col_idx];
            if (conjugate_elements_loc) {
                element.imag(-element.imag());
            }
            mtx_data[mtx_row_offset+mtx_col_idx] = element;
            //std::cout<< element.real << " " << element.imag <<std::endl;

        }


    }); // TBB


    return msg;

}


/**
@brief Constructor of the class.
@param rows_in The matrix containing the transformed rows (T @ mtx(modes,:)  )
@param corner_in The matrix containing the transformed elements (T @ mtx(modes,modes) @ T^T)
@param mtx_out The resulting transformed matrix containing all the transformed modes
@param modes_in A vector of row indices that were transformed.
@return Returns with the instance of the class.
*/
Insert_Transformed_Rows::Insert_Transformed_Rows( matrix &rows_in,matrix &corner_in, matrix &mtx_out, std::vector<size_t> &modes_in, bool* cols_logical_in) {

    rows = rows_in;
    corner = corner_in;
    mtx = mtx_out;

    modes = modes_in;
    cols_logical = cols_logical_in;

}

/**
@brief Operator to insert the transformed rows into the matrix
@param msg A TBB message fireing the node
*/
const tbb::flow::continue_msg &
Insert_Transformed_Rows::operator()(const tbb::flow::continue_msg &msg) {

#ifdef DEBUG
    std::cout << "Tasks apply_to_C_and_G: inserting transformed rows and corner elements" << std::endl;
#endif

    // raw pointer to the stored data in matrix mtx
    Complex16* mtx_data = mtx.get_data();
    // raw pointer to the stored data in matrix rows
    Complex16* rows_data = rows.get_data();
    // raw pointer to the stored data in matrix corner
    Complex16* corner_data = corner.get_data();

    // A vector of row inices to be extracted
    std::vector<size_t> &modes_loc = modes;
    // logical indexes of the columns of the matrix mtx. True values stand for columns corresponding to modes, and false otherwise.
    bool* cols_logical_loc = cols_logical;
    // number of columns
    size_t col_num = rows.cols;

    // inserting the the transformed rows into the matrix

    // parallel for to insert the transformed rows
    int N = (int)(modes.size());
    tbb::parallel_for(0, N, 1, [rows_data, corner_data, mtx_data, modes_loc, cols_logical_loc, col_num](int i) {

        // offset for the i-th row in the matrix rows
        int row_offset = i*col_num;

        // offset for the modes[i]-th row in the matrix mtx
        int mtx_offset = modes_loc[i]*col_num;

        // offset for the i-th row in the matrix corner
        int corner_offset = i*modes_loc.size();

        //std::cout<< i  << " " << row_offset << " " << mtx_offset << " " << corner_offset << std::endl;

        // raw pointers to the individual rows of the matrices
        Complex16* row = rows_data + row_offset;
        Complex16* mtx_row = mtx_data + mtx_offset;
        Complex16* corner_row = corner_data + corner_offset;

        size_t col_idx = 0;
        size_t col_range = 1;
        size_t corner_row_offset = 0;

        // copy the elements from the transformed rows
        while (true) {

            // condition to exit the loop
            if ( col_idx >= col_num ) {
                break;
            }

            //std::cout << col_idx << std::endl;


            // find continuous memory slices that can be copied
            while (true) {

                // condition to exit the loop
                if ( cols_logical_loc[col_idx] != cols_logical_loc[col_idx+col_range] ) {
                    break;
                }
                else {
                    if ( col_idx+col_range+1 >= col_num ) {
                        break;
                    }
                    col_range = col_range + 1;

                }

            }

            // determine whether to copy from rows or from corner
            if (cols_logical_loc[col_idx] == true) {
                // copy from corner
                for (size_t idx = 0; idx<col_range; idx++) {
                    mtx_row[modes_loc[corner_row_offset+idx]] = corner_row[corner_row_offset+idx];
                }
                corner_row_offset = corner_row_offset + col_range;

                //memcpy(mtx_row+col_idx, corner_row, col_range*sizeof(Complex16));
                //corner_row = corner_row + col_range;
            }
            else {
                // copy from rows
                memcpy(mtx_row+col_idx, row+col_idx, col_range*sizeof(Complex16));
            }

            col_idx = col_idx + col_range;
            col_range = 1;


        }

    }); // TBB

    return msg;
}



} // PIC
