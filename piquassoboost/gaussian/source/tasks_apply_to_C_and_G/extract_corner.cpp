/**
 * Copyright 2021 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <tbb/tbb.h>
#include "extract_corner.h"
#include "matrix.h"
#include "dot.h"
#include <memory.h>

namespace pic {


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
                col_range = col_range + 1;
            }

        }

        // the column index in the matrix from we are bout the extract columns to be transformed
        size_t col_idx = modes[transform_col_idx];

        // row-wise parallelized loop to extract the columns to be transformed
        size_t N = (transform_col_num);
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




} // PIC
