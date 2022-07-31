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

#ifndef PowerTraceHafnian_H
#define PowerTraceHafnian_H

#include "matrix.h"
#include "matrix32.h"


namespace pic {



/**
@brief Class to calculate the hafnian of a complex matrix by the power trace method
*/
template <class complex_type>
class PowerTraceHafnian {

protected:
    /// The input matrix. Must be symmetric
    matrix mtx_orig;
    /** The scaled input matrix for which the calculations are performed.
    If the mean magnitude of the matrix elements is one, the treshold of quad precision can be set to higher values.
    */
    matrix mtx;
    /// The scale factor of the input matric
    double scale_factor;


public:


/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
PowerTraceHafnian();

/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix for which the hafnian is calculated. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state)
@return Returns with the instance of the class.
*/
PowerTraceHafnian( matrix &mtx_in );

/**
@brief Default destructor of the class.
*/
virtual ~PowerTraceHafnian();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate();

/**
@brief Call to calculate the hafnian of a complex matrix
@param start_idx The minimal index evaluated in the exponentially large sum (used to divide calculations between MPI processes)
@param step_idx The index step in the exponentially large sum (used to divide calculations between MPI processes)
@param max_idx The maximal indexe valuated in the exponentially large sum (used to divide calculations between MPI processes)
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate(unsigned long long start_idx, unsigned long long step_idx, unsigned long long max_idx );

/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in A symmetric matrix for which the hafnian is calculated. (For example a covariance matrix of the Gaussian state.)
*/
void Update_mtx( matrix &mtx_in);


protected:

/**
@brief Call to scale the input matrix according to according to Eq (2.11) of in arXiv 1805.12498
*/
virtual void ScaleMatrix();



}; //PowerTraceHafnian


using PowerTraceHafnianLongDouble = PowerTraceHafnian<Complex32>;

} // PIC

#endif
