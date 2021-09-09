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
#include "BruteForceLoopHafnian.h"
#include "BruteForceHafnian.h"
#include <math.h>



namespace pic {




/**
@brief Constructor of the class.
@param mtx_in The covariance matrix of the Gaussian state.
@return Returns with the instance of the class.
*/
BruteForceLoopHafnian::BruteForceLoopHafnian( matrix &mtx_in ) {

    mtx = mtx_in;
    dim = mtx.rows;
    dim_over_2 = dim/2;

}


/**
@brief Default destructor of the class.
*/
BruteForceLoopHafnian::~BruteForceLoopHafnian() {

}




/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
Complex16
BruteForceLoopHafnian::calculate() {


    if ( mtx.rows != mtx.cols) {
        std::cout << "The input matrix should be square shaped, bu matrix with " << mtx.rows << " rows and with " << mtx.cols << " columns was given" << std::endl;
        std::cout << "Returning zero" << std::endl;
        return Complex16(0,0);
    }

    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return Complex16(1,0);
    }
    else if (mtx.rows % 2 != 0) {
        // the hafnian of odd shaped matrix is 0 by definition
        return Complex16(0.0, 0.0);
    }





    // calculate contribution without loops
    BruteForceHafnian hafnian_calculator = BruteForceHafnian( mtx );
    Complex16&& hafnian = hafnian_calculator.calculate();


    // iterate over number of loops
    for (size_t num_of_loops = 2; num_of_loops<=dim; num_of_loops= num_of_loops+2) {

        // create initial logical loop indices to start task iterations
        PicVector<char> loop_logicals(dim,0);
            for (size_t idx=0; idx<num_of_loops; idx++) {
            loop_logicals[idx] = 1;
        }

        // spawning tasks over different loop configurations
        SpawnTask(loop_logicals, std::move(num_of_loops), std::move(num_of_loops), hafnian);




    }


    return hafnian;
}



/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in The new covariance matrix
*/
void
BruteForceLoopHafnian::Update_mtx( matrix &mtx_in) {

    mtx = mtx_in;

}





Complex16
BruteForceLoopHafnian::PartialHafnianForGivenLoopIndices( const PicVector<char> &loop_logicals, const size_t num_of_loops) {

    // create the submatrix for the hafnian calculator and calculate the contribution of the loops
    matrix submatrix(dim-num_of_loops, dim-num_of_loops);
    Complex16 loop_contribution(1.0, 0.0);
    size_t row_idx = 0;
    for (size_t idx=0; idx<loop_logicals.size(); idx++) {

        if (loop_logicals[idx]) {
            loop_contribution = loop_contribution * mtx[idx*mtx.stride + idx];
        }
        else {

            size_t col_idx = 0;
            for (size_t jdx=0; jdx<loop_logicals.size(); jdx++) {

                if (!loop_logicals[jdx]) {
                    submatrix[row_idx*submatrix.stride + col_idx] = mtx[idx*mtx.stride + jdx];
                    col_idx++;
                }


            }

            row_idx++;
        }



    }

    if (num_of_loops==dim) {
        return loop_contribution;
    }

    // calculate the hafnian of the remaining part and multiply it with the loop contribution
    BruteForceHafnian hafnian_calculator = BruteForceHafnian( submatrix );
    return loop_contribution*hafnian_calculator.calculate();

}


void
BruteForceLoopHafnian::SpawnTask( PicVector<char>& loop_logicals, size_t&& loop_to_move, const size_t num_of_loops, Complex16& hafnian) {

    // calculate the partial hafnian using the given loop indices
    hafnian = hafnian + PartialHafnianForGivenLoopIndices( loop_logicals, num_of_loops );

    // spawning new iterations with modified loop indices

    // determine the row index to be moved
    size_t loop_to_move_index = 0;
    size_t current_loop = 0;
    for (size_t idx=0; idx<dim; idx++) {

        if (loop_logicals[idx]) {
            current_loop++;
        }

        if (current_loop == loop_to_move) {
            loop_to_move_index = idx;
            break;
        }
    }

    // moving the selected row to valid positions
    for (size_t idx=loop_to_move_index+1; idx<dim; idx++) {

        // check whether idx is occupied by another row or not
        if ( loop_logicals[idx] ) {
            break;
        }

        // check whether there will be enough available columns greater than the row indices


        // create new row logical indices
        PicVector<char> loop_logicals_new = loop_logicals;
        loop_logicals_new[loop_to_move_index] = 0;
        loop_logicals_new[idx] = 1;


        // spawn new tasks to iterate over valid row index combinations
        if (loop_to_move>0) { // the very firs row index must always be 0
            SpawnTask(loop_logicals_new, loop_to_move-1, num_of_loops, hafnian);
        }


    }


    return;

}




} // PIC
