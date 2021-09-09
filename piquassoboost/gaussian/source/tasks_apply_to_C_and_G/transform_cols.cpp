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
#include "transform_cols.h"
#include "matrix.h"
#include "dot.h"
#include <memory.h>

namespace pic {



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



} // PIC
