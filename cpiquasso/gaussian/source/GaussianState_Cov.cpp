#include <iostream>
#include "GaussianState_Cov.h"
#include <memory.h>

namespace pic {

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GaussianState_Cov::GaussianState_Cov() {}



/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix
@param m_in The displacement of the Gaussian state
@return Returns with the instance of the class.
*/
GaussianState_Cov::GaussianState_Cov( matrix &covariance_matrix_in, matrix &m_in) {

    Update_covariance_matrix( covariance_matrix_in );
    Update_m(m_in);

}



/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix (the displacements are set to zeros)
@param m_in The displacement of the Gaussian state
@return Returns with the instance of the class.
*/
GaussianState_Cov::GaussianState_Cov( matrix &covariance_matrix_in) {

    Update_covariance_matrix( covariance_matrix_in );

    // set the displacement to zero
    m = matrix(1,covariance_matrix_in.rows);
    memset(m.get_data(), 0, m.size()*sizeof(Complex16));

}



/**
@brief Call to get a reduced Gaussian state (i.e. the gaussian state represented by a subset of modes of the original gaussian state)
@param modes An instance of PicState_int64 containing the modes to be extracted from the original gaussian state
@return Returns with the reduced Gaussian state
*/
GaussianState_Cov
GaussianState_Cov::getReducedGaussianState( PicState_int64 &modes ) {

    size_t total_number_of_modes = covariance_matrix.rows/2;

    if (total_number_of_modes == 0) {
        std::cout << "There is no covariance matrix to be reduced. Exiting" << std::endl;
        exit(-1);
    }


    // the number of modes to be extracted
    size_t number_of_modes = modes.size();
    if (number_of_modes == total_number_of_modes) {
        return GaussianState_Cov(covariance_matrix, m);
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
    GaussianState_Cov ret(covariance_matrix_reduced, m_reduced);

    //m_reduced.print_matrix();
    //covariance_matrix_reduced.print_matrix();

    return ret;


}




/**
@brief Call to update the memory address of the vector containing the displacements
@param m_in The new displacement vector
*/
void
GaussianState_Cov::Update_m(matrix &m_in) {

    m = m_in;

}


/**
@brief Call to update the memory address of the matrix stored in covariance_matrix
@param covariance_matrix_in The covariance matrix describing the gaussian state
*/
void
GaussianState_Cov::Update_covariance_matrix( matrix &covariance_matrix_in ) {

    covariance_matrix = covariance_matrix_in;

}



} // PIC
