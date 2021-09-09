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

#ifndef BruteForceLoopHafnian_H
#define BruteForceLoopHafnian_H


#include "matrix.h"
#include "PicState.h"
#include "PicVector.hpp"


namespace pic {



/**
@brief Class to calculate the loop hafnian by brute force method
*/
class BruteForceLoopHafnian  {

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
BruteForceLoopHafnian();


/**
@brief Constructor of the class.
@param mtx_in The covariance matrix of the Gaussian state.
@return Returns with the instance of the class.
*/
BruteForceLoopHafnian( matrix &mtx_in );



/**
@brief Default destructor of the class.
*/
virtual ~BruteForceLoopHafnian();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate();


Complex16 PartialHafnianForGivenLoopIndices( const PicVector<char> &loop_logicals, const size_t num_of_loops );


void SpawnTask( PicVector<char>& loop_logicals, size_t&& loop_to_move, const size_t num_of_loops, Complex16& hafnian);


/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void Update_mtx( matrix &mtx_in);


protected:



}; //BruteForceLoopHafnian




} // PIC

#endif
