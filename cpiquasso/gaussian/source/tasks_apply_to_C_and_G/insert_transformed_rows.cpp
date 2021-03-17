#include <iostream>
#include <tbb/tbb.h>
#include "insert_transformed_rows.h"
#include "matrix.h"
#include <memory.h>

namespace pic {



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
