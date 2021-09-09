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

#ifndef BruteForceHafnian_H
#define BruteForceHafnian_H

#include "matrix.h"
#include "PicState.h"
#include "PicVector.hpp"


namespace pic {



/**
@brief Class to calculate the hafnian by brute force method
*/
class BruteForceHafnian {

protected:
    /// The covariance matrix of the Gaussian state.
    matrix mtx;

    size_t dim;
    size_t dim_over_2;


public:


/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
BruteForceHafnian();


/**
@brief Constructor of the class.
@param mtx_in The covariance matrix of the Gaussian state.
@return Returns with the instance of the class.
*/
BruteForceHafnian( matrix &mtx_in );

/**
@brief Default destructor of the class.
*/
virtual ~BruteForceHafnian();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate();


Complex16 ColPermutationsForGivenRowIndices( const PicVector<short> &row_indices, PicVector<short> &col_indices, size_t&& col_to_iterate);


void SpawnTask( PicVector<char>& row_indices, size_t&& row_to_move, Complex16& hafnian);


/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void Update_mtx( matrix &mtx_in);


protected:



}; //BruteForceHafnian




} // PIC

#endif
