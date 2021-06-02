#include <iostream>
#include <tbb/tbb.h>
#include "insert_transformed_cols.h"
#include "matrix.h"
#include "dot.h"
#include <memory.h>

namespace pic {





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




} // PIC
