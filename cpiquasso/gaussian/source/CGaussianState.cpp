#include <iostream>
#include "CGaussianState.h"
#include "tasks_apply_to_C_and_G.h"
#include "dot.h"
#include <memory.h>
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"

namespace pic {

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
CGaussianState::CGaussianState() {}

/**
@brief Constructor of the class.
@param C_in Input matrix defined by
@param G_in Input matrix defined by
@param m_in Input matrix defined by
@return Returns with the instance of the class.
*/
CGaussianState::CGaussianState( matrix &C_in, matrix &G_in, matrix &m_in) {

    Update( C_in, G_in, m_in);
}


/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix
@param m_in The displacement of the Gaussian state
@return Returns with the instance of the class.
*/
CGaussianState::CGaussianState( matrix &covariance_matrix_in, matrix &m_in) {

    Update_covariance_matrix( covariance_matrix_in );
    Update_m(m_in);

}



/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix (the displacements are set to zeros)
@param m_in The displacement of the Gaussian state
@return Returns with the instance of the class.
*/
CGaussianState::CGaussianState( matrix &covariance_matrix_in) {

    Update_covariance_matrix( covariance_matrix_in );

    // set the displacement to zero
    m = matrix(1,covariance_matrix_in.rows);
    memset(m.get_data(), 0, m.size()*sizeof(Complex16));

}


/**
@brief Applies the matrix T to the C and G.
@param T The matrix of the transformation.
@param modes The modes, on which the matrix should operate.
@return Returns with 0 in case of success.
*/
int
CGaussianState::apply_to_C_and_G( matrix &T, std::vector<size_t> modes ) {

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



/**
@brief Call to get a reduced Gaussian state (i.e. the gaussian state represented by a subset of modes of the original gaussian state)
@param modes An instance of PicState_int64 containing the modes to be extracted from the original gaussian state
@return Returns with the reduced Gaussian state
*/
CGaussianState
CGaussianState::getReducedGaussianState( PicState_int64 &modes ) {

    size_t total_number_of_modes = covariance_matrix.rows/2;

    if (total_number_of_modes == 0) {
        std::cout << "There is no covariance matrix to be reduced. Exiting" << std::endl;
        exit(-1);
    }


    // the number of modes to be extracted
    size_t number_of_modes = modes.size();
    if (number_of_modes == total_number_of_modes) {
        return CGaussianState(covariance_matrix, m);
    }
    else if ( number_of_modes >= total_number_of_modes) {
        std::cout << "The number of modes to be extracted is larger than the posibble number of modes. Exiting" << std::endl;
        exit(-1);
    }



    // allocate data for the reduced covariance matrix
    matrix covariance_matrix_reduced(number_of_modes*2, number_of_modes*2);  // the size of the covariance matrix must be the double of the number of modes
    Complex16* covariance_matrix_reduced_data = covariance_matrix_reduced.get_data();
    Complex16* covariance_matrix_data = covariance_matrix.get_data();

    // allocate data for the reduced displacement
    matrix m_reduced(1, number_of_modes*2);
    Complex16* m_reduced_data = m_reduced.get_data();
    Complex16* m_data = m.get_data();


    size_t mode_idx = 0;
    size_t col_range = 1;
    // loop over the col indices to be transformed (indices are stored in attribute modes)
    while (true) {

        // condition to exit the loop: if there are no further columns then we exit the loop
        if ( mode_idx >= number_of_modes) {
            break;
        }

        // determine contiguous memory slices (column indices) to be extracted
        while (true) {

            // condition to exit the loop: if the difference of successive indices is greater than 1, the end of the contiguous memory slice is determined
            if ( mode_idx+col_range >= number_of_modes || modes[mode_idx+col_range] - modes[mode_idx+col_range-1] != 1 ) {
                break;
            }
            else {
                if (mode_idx+col_range+1 >= number_of_modes) {
                    break;
                }
                col_range = col_range + 1;

            }

        }

        // the column index in the matrix from we are bout the extract columns
        size_t col_idx = modes[mode_idx];

        // row-wise loop to extract the q quadrature columns from the covariance matrix
        for(size_t mode_row_idx=0; mode_row_idx<number_of_modes; mode_row_idx++) {

            // col-wise extraction of the q quadratures from the covariance matrix
            size_t cov_reduced_offset = mode_idx*covariance_matrix_reduced.stride + mode_idx;
            size_t cov_offset = modes[mode_idx]*covariance_matrix.stride + col_idx;
            memcpy(covariance_matrix_reduced_data + cov_reduced_offset, covariance_matrix_data + cov_offset , col_range*sizeof(Complex16));

            // col-wise extraction of the p quadratures from the covariance matrix
            cov_reduced_offset = cov_reduced_offset + number_of_modes;
            cov_offset = cov_offset + total_number_of_modes;
            memcpy(covariance_matrix_reduced_data + cov_reduced_offset, covariance_matrix_data + cov_offset , col_range*sizeof(Complex16));

        }


        // row-wise loop to extract the p quadrature columns from the covariance matrix
        for(size_t mode_row_idx=0; mode_row_idx<number_of_modes; mode_row_idx++) {

            // col-wise extraction of the q quadratures from the covariance matrix
            size_t cov_reduced_offset = (mode_idx+number_of_modes)*covariance_matrix_reduced.stride + mode_idx;
            size_t cov_offset = (modes[mode_idx]+total_number_of_modes)*covariance_matrix.stride + col_idx;
            memcpy(covariance_matrix_reduced_data + cov_reduced_offset, covariance_matrix_data + cov_offset , col_range*sizeof(Complex16));

            // col-wise extraction of the p quadratures from the covariance matrix
            cov_reduced_offset = cov_reduced_offset + number_of_modes;
            cov_offset = cov_offset + total_number_of_modes;
            memcpy(covariance_matrix_reduced_data + cov_reduced_offset, covariance_matrix_data + cov_offset , col_range*sizeof(Complex16));
        }

        // extract modes from the displacement
        memcpy(m_reduced_data + mode_idx, m_data + col_idx, col_range*sizeof(Complex16)); // q quadratires
        memcpy(m_reduced_data + mode_idx + number_of_modes, m_data + col_idx + total_number_of_modes, col_range*sizeof(Complex16)); // p quadratures

        mode_idx = mode_idx + col_range;
        col_range = 1;

    }


    // creating the reduced Gaussian state
    CGaussianState ret(covariance_matrix_reduced, m_reduced);

    //m_reduced.print_matrix();
    //covariance_matrix_reduced.print_matrix();

    return ret;


}



/**
@brief Call to update the memory addresses of the stored matrices
@param C_in Input matrix defined by
@param G_in Input matrix defined by
@param m_in Input matrix defined by
*/
void
CGaussianState::Update( matrix &C_in, matrix &G_in, matrix &m_in) {

    C = C_in;
    G = G_in;
    m = m_in;

}



/**
@brief Call to update the memory address of the matrix C
@param C_in Input matrix defined by
*/
void
CGaussianState::Update_C( matrix &C_in) {

    C = C_in;

}


/**
@brief Call to update the memory address of the matrix G
@param G_in Input matrix defined by
*/
void
CGaussianState::Update_G(matrix &G_in) {

    G = G_in;

}


/**
@brief Call to update the memory address of the vector containing the displacements
@param m_in The new displacement vector
*/
void
CGaussianState::Update_m(matrix &m_in) {

    m = m_in;

}


/**
@brief Call to update the memory address of the matrix stored in covariance_matrix
@param covariance_matrix_in The covariance matrix describing the gaussian state
*/
void
CGaussianState::Update_covariance_matrix( matrix &covariance_matrix_in ) {

    covariance_matrix = covariance_matrix_in;

}



} // PIC
