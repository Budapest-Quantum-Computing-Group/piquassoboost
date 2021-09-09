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
#include "extract_rows.h"
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










} // PIC
