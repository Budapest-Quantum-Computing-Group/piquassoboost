#ifndef PowerTraceLoopHafnianRecursive_H
#define PowerTraceLoopHafnianRecursive_H

#include "PowerTraceLoopHafnian.h"
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
class PowerTraceLoopHafnianRecursive : public PowerTraceLoopHafnian {


protected:
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
@brief Default destructor of the class.
*/
virtual ~PowerTraceLoopHafnianRecursive();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate();


}; //PowerTraceLoopHafnianRecursive



// relieve Python extension from TBB functionalities
#ifndef CPYTHON

/**
@brief Class to calculate the loop hafnian of a complex matrix by a recursive power trace method. This algorithm is designed to support gaussian boson sampling simulations, it is not a general
purpose loop hafnian calculator. This algorithm accounts for the repeated occupancy in the covariance matrix.
*/
class PowerTraceLoopHafnianRecursive_Tasks : public PowerTraceHafnianRecursive_Tasks {


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
Complex32 CalculatePartialHafnian( const PicVector<char>& selected_modes, const  PicState_int64& current_occupancy );



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
matrix CreateDiagElements( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, const size_t& num_of_modes );


}; //PowerTraceLoopHafnianRecursive_Tasks


#endif





} // PIC

#endif
