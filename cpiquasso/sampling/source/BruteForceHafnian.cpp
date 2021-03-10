#include <iostream>
#include "BruteForceHafnian.h"
#include <math.h>



namespace pic {

/**
@brief Constructor of the class.
@param mtx_in The covariance matrix of the Gaussian state.
@return Returns with the instance of the class.
*/
BruteForceHafnian::BruteForceHafnian( matrix &mtx_in ) {

    mtx = mtx_in;
    dim = mtx.rows;
    dim_over_2 = dim/2;

}


/**
@brief Default destructor of the class.
*/
BruteForceHafnian::~BruteForceHafnian() {

}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
Complex16
BruteForceHafnian::calculate() {


    if ( mtx.rows != mtx.cols) {
        std::cout << "The input matrix should be square shaped, bu matrix with " << mtx.rows << " rows and with " << mtx.cols << " columns was given" << std::endl;
        std::cout << "Returning zero" << std::endl;
        return Complex16(0,0);
    }

    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return Complex16(1,0);
    }
    else if (mtx.rows == 2) {
        return mtx[1];
    }
    else if (mtx.rows % 2 != 0) {
        // the hafnian of odd shaped matrix is 0 by definition
        return Complex16(0.0, 0.0);
    }


    Complex16 hafnian( 0.0, 0.0 );


    // create initial logical row indices to start task iterations
    PicVector<char> row_logicals(dim,0);
    for (size_t idx=0; idx<dim_over_2; idx++) {
        row_logicals[idx] = 1;
    }

    // spawning tasks over different row configurations
    SpawnTask(row_logicals, std::move(dim_over_2), hafnian);


    return hafnian;
}



/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in The new covariance matrix
*/
void
BruteForceHafnian::Update_mtx( matrix &mtx_in) {

    mtx = mtx_in;

}





Complex16
BruteForceHafnian::ColPermutationsForGivenRowIndices( const PicVector<short> &row_indices, PicVector<short> &col_indices, size_t&& col_to_iterate) {


    if (col_to_iterate > 1)  {

        // determine the minimal valid column index
        size_t minimal_col_index = row_indices[col_to_iterate-1];

        Complex16 partial_hafnian(0.0,0.0);

        for (int col_idx = col_indices.size()-1; col_idx>=0; col_idx--) {

            size_t current_column = col_indices[col_idx];

            // check whether the column index is not occupied
            if (current_column <= minimal_col_index) {
                break;
            }

            PicVector<short> col_indices_new(col_indices.size()-1);
            size_t jdx=0;
            for (size_t idx=0; idx<col_indices.size(); idx++) {
                if (idx == col_idx) {
                    continue;
                }

                col_indices_new[jdx] = col_indices[idx];
                jdx++;

            }

            Complex16&& partial_hafnian_tmp = ColPermutationsForGivenRowIndices(row_indices, col_indices_new, col_to_iterate-1);
            partial_hafnian = partial_hafnian + partial_hafnian_tmp * mtx[minimal_col_index*mtx.stride + current_column];
//std::cout <<  partial_hafnian << std::endl;

        }

        return partial_hafnian;

    }
    else {
//std::cout <<  mtx[col_indices[0]] << " " << col_indices[0] << std::endl;
        return mtx[col_indices[0]];

    }


}


void
BruteForceHafnian::SpawnTask( PicVector<char>& row_logicals, size_t&& row_to_move, Complex16& hafnian) {

    // calculate the partial hafnian for the given row indices by permutating columns
    PicVector<short> row_indices(dim_over_2);
    PicVector<short> col_indices(dim_over_2);
    size_t row_idx = 0;
    size_t col_idx = 0;
    for (size_t idx=0; idx<dim; idx++) {
        if (row_logicals[idx]) {
            row_indices[row_idx] = idx;
            row_idx++;
        }
        else {
            col_indices[col_idx] = idx;
            col_idx++;
        }
    }

    //calculate partial hafnian
    hafnian = hafnian + ColPermutationsForGivenRowIndices(row_indices, col_indices, std::move(dim_over_2));


    // spawning new iterations with modified row indices

    // determine the number of column whose indices must be greater than the row to be moved
    size_t required_cols = dim - 2*row_to_move + 1;

    // determine the row index to be moved
    size_t row_to_move_index = 0;
    size_t current_row = 0;
    for (size_t idx=0; idx<dim; idx++) {

        if (row_logicals[idx]) {
            current_row++;
        }

        if (current_row == row_to_move) {
            row_to_move_index = idx;
            break;
        }
    }


    // moving the selected row to valid positions
    for (size_t idx=row_to_move_index+1; idx<dim-required_cols; idx++) {

        // check whether idx is occupied by another row or not
        if ( row_logicals[idx] ) {
            break;
        }

        // check whether there will be enough available columns greater than the row indices


        // create new row logical indices
        PicVector<char> row_logicals_new = row_logicals;
        row_logicals_new[row_to_move_index] = 0;
        row_logicals_new[idx] = 1;


        // spawn new tasks to iterate over valid row index combinations
        if (row_to_move>1) { // the very firs row index must always be 0

            SpawnTask(row_logicals_new, row_to_move-1, hafnian);

        }


    }


    return;

}




} // PIC
