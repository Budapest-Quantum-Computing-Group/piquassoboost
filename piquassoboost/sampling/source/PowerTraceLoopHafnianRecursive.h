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

#ifndef PowerTraceLoopHafnianRecursive_H
#define PowerTraceLoopHafnianRecursive_H

#include "PowerTraceLoopHafnian.h"
#include "PowerTraceHafnianUtilities.h"
#include "PowerTraceHafnianRecursive.h"
#include "PicState.h"
#include "PicVector.hpp"

#ifndef CPYTHON
#include "tbb/tbb.h"
#endif


namespace pic {

/**
@brief Wrapper class to calculate the loop hafnian of a complex matrix by the recursive power trace method, which also accounts for the repeated occupancy in the covariance matrix.
This class is an interface class betwwen the Python extension and the C++ implementation to relieve python extensions from TBB functionalities.
(CPython does not support static objects with constructors/destructors)
*/
template <class small_scalar_type, class scalar_type>
class PowerTraceLoopHafnianRecursive : public PowerTraceLoopHafnian<small_scalar_type, scalar_type> {


protected:
    /// The diagonal elements of the input matrix
    matrix diag;
    /// An array describing the occupancy to be used to calculate the hafnian. The i-th mode is repeated occupancy[i] times.
    PicState_int64 occupancy;

public:

/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state.)
@param occupancy_in An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
PowerTraceLoopHafnianRecursive( matrix &mtx_in, PicState_int64& occupancy_in );


/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state.)
@param diag_elements_in
@param occupancy_in An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
PowerTraceLoopHafnianRecursive( matrix &mtx_in, matrix &diag_in, PicState_int64& occupancy_in );


/**
@brief Default destructor of the class.
*/
virtual ~PowerTraceLoopHafnianRecursive();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate();


}; //PowerTraceLoopHafnianRecursive

using PowerTraceLoopHafnianRecursiveHybrid = PowerTraceLoopHafnianRecursive<double, long double>;
using PowerTraceLoopHafnianRecursiveDouble = PowerTraceLoopHafnianRecursive<double, double>;
using PowerTraceLoopHafnianRecursiveLongDouble = PowerTraceLoopHafnianRecursive<long double, long double>;

#ifdef __MPFR__
using PowerTraceLoopHafnianRecursiveInf = PowerTraceLoopHafnianRecursive<RationalInf, RationalInf>;
#endif

// relieve Python extension from TBB functionalities
#ifndef CPYTHON

/**
@brief Class to calculate the loop hafnian of a complex matrix by a recursive power trace method. This algorithm is designed to support gaussian boson sampling simulations, it is not a general
purpose loop hafnian calculator. This algorithm accounts for the repeated occupancy in the covariance matrix.
*/
template <class small_scalar_type, class scalar_type>
class PowerTraceLoopHafnianRecursive_Tasks : public PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type> {

protected:
    /// The diagonal elements of the input matrix
    matrix diag;

public:

/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state.)
@param occupancy_in An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
PowerTraceLoopHafnianRecursive_Tasks( matrix &mtx_in, PicState_int64& occupancy_in );


/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state.)
@param diag_elements_in
@param occupancy_in An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
PowerTraceLoopHafnianRecursive_Tasks( matrix &mtx_in, matrix &diag_elements_in, PicState_int64& occupancy_in );


/**
@brief Default destructor of the class.
*/
virtual ~PowerTraceLoopHafnianRecursive_Tasks();


protected:

/**
@brief Call to calculate the partial hafnian for given selected modes and their occupancies
@param selected_modes Selected column pairs over which the iterations are run
@param current_occupancy Current occupancy of the selected column pairs for which the partial hafnian is calculated
@return Returns with the calculated hafnian
*/
cplx_select_t<scalar_type> CalculatePartialHafnian( const PicVector<char>& selected_modes, const  PicState_int64& current_occupancy );


/**
@brief Call to construct matrix \f$ A^Z \f$ (see the text below Eq. (3.20) of arXiv 1805.12498) for the given modes and their occupancy
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param num_of_modes The number of modes (including degeneracies) that have been previously calculated. (it is the sum of values in current_occupancy)
@param scale_factor_AZ The scale factor that has been used to scale the matrix elements of AZ =returned by reference)
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
mtx_select_t<cplx_select_t<small_scalar_type>> CreateAZ( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, const size_t& total_num_of_occupancy, small_scalar_type &scale_factor_AZ );

/**
@brief Call to scale the input matrix according to according to Eq (2.14) of in arXiv 1805.12498
@param mtx_in Input matrix defined by
*/
void ScaleMatrix();


/**
@brief Call to create diagonal elements corresponding to the diagonal elements of the input  matrix used in the loop correction
@param selected_modes Selected columns pairs over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param num_of_modes The number of modes (including degeneracies) that have been previously calculated. (it is the sum of values in current_occupancy)
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
mtx_select_t<cplx_select_t<small_scalar_type>> CreateDiagElements( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, const size_t& num_of_modes );


}; //PowerTraceLoopHafnianRecursive_Tasks


#endif





} // PIC

#endif
