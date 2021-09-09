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
#include <assert.h>





static tbb::spin_mutex my_mutex;
/*
double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {


/**
@brief Constructor of the class.

@param mtx_in A selfadjoint matrix for which the torontonian is calculated.
    This matrix has to be positive definite matrix with eigenvalues between 0 and 1
    (for example a covariance matrix of the Gaussian state.) matrix.
    ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$
    ordered covariance matrix of the Gaussian state)
@param occupancy An \f$ n \f$ long array describing the number of rows an columns
    to be repeated during the hafnian calculation.
    The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
    (The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
TorontonianRecursive::TorontonianRecursive( matrix_real &mtx_in ) {
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
@param use_extended Logical variable to indicate whether use extended precision for cholesky decomposition (default), or not.
@return Returns with the calculated hafnian
*/
double
TorontonianRecursive::calculate(bool use_extended = true) {

    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return 1.0;
    }


    double torontonian;

    if (use_extended) {
        TorontonianRecursive_Tasks<matrix_real, matrix_real16, long double, long double> torontonian_calculator(mtx);
        torontonian = torontonian_calculator.calculate();
    }
    else {
        TorontonianRecursive_Tasks<matrix_real, matrix_real, double, long double> torontonian_calculator(mtx);
        torontonian = torontonian_calculator.calculate();
    }




    return torontonian;


}


/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void
TorontonianRecursive::Update_mtx( matrix_real &mtx_in ){
    mtx_orig = mtx_in;

    size_t dim = mtx_in.rows;

    // Calculating B := 1 - A
    mtx = matrix_real(dim, dim);
    for (size_t idx = 0; idx < dim; idx++) {
        //Complex16 *row_B_idx = B.get_data() + idx * B.stride;
        //Complex16 *row_mtx_pos_idx = mtx.get_data() + positions_of_ones[idx] * mtx.stride;
        for (size_t jdx = 0; jdx < dim; jdx++) {
            mtx[idx * dim + jdx] = -1.0 * mtx_in[idx * mtx_in.stride + jdx];
        }
        mtx[idx * dim + idx] += 1.0;
    }
}








} // PIC
