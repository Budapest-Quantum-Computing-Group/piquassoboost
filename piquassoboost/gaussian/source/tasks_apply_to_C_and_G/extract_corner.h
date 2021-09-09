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
