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
#include "CalculationHelpers.h"
#include "dot.h"
#include <memory.h>
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "tasks_apply_to_C_and_G/extract_rows.h"
#include "tasks_apply_to_C_and_G/transform_rows.h"
#include "tasks_apply_to_C_and_G/extract_corner.h"
#include "tasks_apply_to_C_and_G/transform_cols.h"
#include "tasks_apply_to_C_and_G/insert_transformed_cols.h"
#include "tasks_apply_to_C_and_G/insert_transformed_rows.h"

namespace pic {


/**
@brief Applies the matrix T to the C and G.
@param T The matrix of the transformation.
@param modes The modes, on which the matrix should operate.
@return Returns with 0 in case of success.
*/
int
apply_to_C_and_G( matrix &C, matrix &G, matrix &T, std::vector<size_t> modes ) {

#if BLAS==1
    int NumThreads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS==2
    int NumThreads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

    // creating TBB dependency graph
    tbb::flow::graph g;

    // creating starting node
    Node start_node(g, StartingNode());

    // TASK A: extract the rows to be transformed from matrices C and G such as:
    // C_rows = C( modes, : )
    // G_rows = G( modes, : )
    matrix C_rows( modes.size(), C.cols);
    matrix G_rows( modes.size(), G.cols);


    // creating task nodes for task A
    Node extract_rows_from_C_node(g, Extract_Rows( C, C_rows, modes ));
    Node extract_rows_from_G_node(g, Extract_Rows( G, G_rows, modes ));

    // adding edges between nodes
    tbb::flow::make_edge(start_node, extract_rows_from_C_node);
    tbb::flow::make_edge(start_node, extract_rows_from_G_node);

    // TASK B: multiply the extracted rows by the transformation matrix:
    // C_rows = (T*) @ C_rows
    // G_rows = T @ G_rows

    // creating task nodes for task B
    T.conjugate();
    Node transform_rows_C_node(g, Transform_Rows( T, C_rows ));
    T.conjugate();
    Node transform_rows_G_node(g, Transform_Rows( T, G_rows ));

    // adding edges between nodes
    tbb::flow::make_edge(extract_rows_from_C_node, transform_rows_C_node);
    tbb::flow::make_edge(extract_rows_from_G_node, transform_rows_G_node);

    // TASK C: extrcat the columns to be transformed from the matrices C_rows, and G_rows
    // C_corner = C_rows(:,modes)
    // G_corner = G_rows(:,modes)
    matrix C_corner( C_rows.rows, modes.size());
    matrix G_corner( G_rows.rows, modes.size());

    bool* cols_logical_C = (bool*)scalable_aligned_malloc( C_rows.cols*sizeof(bool), CACHELINE);
    bool* cols_logical_G = (bool*)scalable_aligned_malloc( C_rows.cols*sizeof(bool), CACHELINE);
    assert(cols_logical_C);
    assert(cols_logical_G);

    // creating task nodes for task C
    Node extract_cols_C_node(g, Extract_Corner( C_rows, C_corner, modes, cols_logical_C ));
    Node extract_cols_G_node(g, Extract_Corner( G_rows, G_corner, modes, cols_logical_G ));

    // adding edges between nodes
    tbb::flow::make_edge(transform_rows_C_node, extract_cols_C_node);
    tbb::flow::make_edge(transform_rows_G_node, extract_cols_G_node);

    // TASK D: multiply the extracted corner matrix by the transformation matrix:
    // C_corner = C_corner @ T.transpose
    // G_corner = G_corner @ (T.transpose).conjugate

    // creating task nodes for task D
    T.transpose();
    Node transform_cols_C_node(g, Transform_Cols( C_corner, T ));
    Node transform_cols_G_node(g, Transform_Cols( G_corner, T ));
    T.transpose();

    // adding edges between nodes
    tbb::flow::make_edge(extract_cols_C_node, transform_cols_C_node);
    tbb::flow::make_edge(extract_cols_G_node, transform_cols_G_node);


    // TASK E: insert the transformed columns determined as the adjungate of the transformed rows
    // C[:,all_other_modes] =  (C[all_other_modes,:].transpose()).conjugate()
    // G[:,all_other_modes] =  (G[all_other_modes,:].transpose())

    // creating task nodes for task E
    Node insert_cols_C_node(g, Insert_Transformed_Cols( C_rows, C, modes, cols_logical_C, /* conjugate elements = */ true ));
    Node insert_cols_G_node(g, Insert_Transformed_Cols( G_rows, G, modes, cols_logical_G, /* conjugate elements = */ false ));

    // adding edges between nodes
    tbb::flow::make_edge(extract_cols_C_node, insert_cols_C_node);
    tbb::flow::make_edge(extract_cols_G_node, insert_cols_G_node);

    // TASK F: insert the transformed rows into the matrices
    // C[modes,all_other_modes] =  C_rows[:,all_other_modes], C[modes,modes] = C_corner
    // G[modes,all_other_modes] =  G_rows[:,all_other_modes], G[modes,modes] = G_corner

    // creating task nodes for task F
    Node insert_rows_C_node(g, Insert_Transformed_Rows( C_rows, C_corner, C, modes, cols_logical_C ));
    Node insert_rows_G_node(g, Insert_Transformed_Rows( G_rows, G_corner, G, modes, cols_logical_G ));

    // adding edges between nodes
    tbb::flow::make_edge(transform_cols_C_node, insert_rows_C_node);
    tbb::flow::make_edge(transform_cols_G_node, insert_rows_G_node);



    // start the calculations over the graph
    start_node.try_put(tbb::flow::continue_msg());

    // make the graph waiting to finish all the calculations
    g.wait_for_all();

    scalable_aligned_free(cols_logical_C);
    scalable_aligned_free(cols_logical_G);

#if BLAS==1 //MKL
    MKL_Set_Num_Threads(NumThreads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(NumThreads);
#endif

    return 0;
}


} // PIC
