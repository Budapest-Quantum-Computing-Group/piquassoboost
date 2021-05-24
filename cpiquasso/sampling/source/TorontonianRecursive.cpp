#ifndef LONG_DOUBLE_CUTOFF
#define LONG_DOUBLE_CUTOFF 50
#endif // LONG_DOUBLE_CUTOFF

#include <iostream>
#include "TorontonianRecursive.h"
#include "TorontonianUtilities.h"
#include "TorontonianRecursive.hpp"

#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>

#ifdef __MPI__
#include <mpi.h>
#endif // MPI





static tbb::spin_mutex my_mutex;
/*
double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {


/**
@brief Constructor of the class.
@param mtx_in A selfadjoint matrix for which the torontonian is calculated. This matrix has to be positive definite matrix with eigenvalues between 0 and 1 (for example a covariance matrix of the Gaussian state.) matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state)
@param occupancy An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
TorontonianRecursive::TorontonianRecursive( matrix &mtx_in ) {
    assert(isSymmetric(mtx_in));

    Update_mtx(mtx_in);

    //mtx = mtx_in;

}


/**
@brief Destructor of the class.
*/
TorontonianRecursive::~TorontonianRecursive() {


}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
double
TorontonianRecursive::calculate() {

    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return 0.0;
    }



    TorontonianRecursive_Tasks<matrix32, Complex32> torontonian_calculator = TorontonianRecursive_Tasks<matrix32, Complex32>(mtx);
    double torontonian = torontonian_calculator.calculate();

    return torontonian;


}


/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void
TorontonianRecursive::Update_mtx( matrix &mtx_in ){
    mtx_orig = mtx_in;

    size_t dim = mtx_in.rows;

    // Calculating B := 1 - A
    mtx = matrix(dim, dim);
    for (size_t idx = 0; idx < dim; idx++) {
        //Complex16 *row_B_idx = B.get_data() + idx * B.stride;
        //Complex16 *row_mtx_pos_idx = mtx.get_data() + positions_of_ones[idx] * mtx.stride;
        for (size_t jdx = 0; jdx < dim; jdx++) {
            mtx[idx * dim + jdx] = -1.0 * mtx_in[idx * mtx_in.stride + jdx];
        }
        mtx[idx * dim + idx] += Complex16(1.0, 0.0);
    }

    // Can scaling be used here since we have to calculate 1-A^Z?
    // It brings a multiplying for each determinant.
    // Should
    //ScaleMatrix();
}








} // PIC
