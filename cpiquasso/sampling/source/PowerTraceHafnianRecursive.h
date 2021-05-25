#ifndef PowerTraceHafnianRecursive_H
#define PowerTraceHafnianRecursive_H

#include "PowerTraceHafnian.h"
#include "PicState.h"
#include "PicVector.hpp"

#ifndef CPYTHON
#include "tbb/tbb.h"
#endif


namespace pic {

/**
@brief Wrapper class to calculate the hafnian of a complex matrix by the recursive power trace method, which also accounts for the repeated occupancy in the covariance matrix.
This class is an interface class betwwen the Python extension and the C++ implementation to relieve python extensions from TBB functionalities.
(CPython does not support static objects with constructors/destructors)
*/
class PowerTraceHafnianRecursive : public PowerTraceHafnian {


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
PowerTraceHafnianRecursive( matrix &mtx_in, PicState_int64& occupancy_in );


/**
@brief Default destructor of the class.
*/
virtual ~PowerTraceHafnianRecursive();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate();


}; //PowerTraceHafnianRecursive



// relieve Python extension from TBB functionalities
#ifndef CPYTHON

/**
@brief Class to calculate the hafnian of a complex matrix by a recursive power trace method. This algorithm is designed to support gaussian boson sampling simulations, it is not a general
purpose hafnian calculator. This algorithm accounts for the repeated occupancy in the covariance matrix.
*/
class PowerTraceHafnianRecursive_Tasks : public PowerTraceHafnian {


protected:
    /// An array describing the occupancy to be used to calculate the hafnian. The i-th mode is repeated occupancy[i] times.
    PicState_int64 occupancy;
    /// The maximal number of spawned tasks living at the same time
    short max_task_num;
    /// The current number of spawned tasks
    int task_num;
    /// mutual exclusion to count the spawned tasks
    tbb::spin_mutex* task_count_mutex;

public:

/**
@brief Nullary constructor of the class.
@return Returns with the instance of the class.
*/
PowerTraceHafnianRecursive_Tasks();

/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state.)
@param occupancy_in An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
PowerTraceHafnianRecursive_Tasks( matrix &mtx_in, PicState_int64& occupancy_in );


/**
@brief Default destructor of the class.
*/
virtual ~PowerTraceHafnianRecursive_Tasks();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
Complex16 calculate();


/**
@brief Call to calculate the hafnian of a complex matrix
@param start_idx The minimal index evaluated in the exponentially large sum (used to divide calculations between MPI processes)
@param step_idx The index step in the exponentially large sum (used to divide calculations between MPI processes)
@param max_idx The maximal indexe valuated in the exponentially large sum (used to divide calculations between MPI processes)
@return Returns with the calculated hafnian
*/
Complex16 calculate(unsigned long long start_idx, unsigned long long step_idx, unsigned long long max_idx );



protected:

/**
@brief Call to run iterations over the selected modes to calculate partial hafnians
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param mode_to_iterate The mode for which the occupancy numbers are iterated
@param priv_addend Therad local storage for the partial hafnians
@param tg Reference to a tbb::task_group
*/
void IterateOverSelectedModes( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, size_t mode_to_iterate, tbb::combinable<ComplexM<long double>>& priv_addend, tbb::task_group &tg );


/**
@brief Call to calculate the partial hafnian for given selected modes and their occupancies
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@return Returns with the calculated partial hafnian
*/
virtual Complex32 CalculatePartialHafnian( const PicVector<char>& selected_modes, const  PicState_int64& current_occupancy );


/**
@brief Call to construct matrix \f$ A^Z \f$ (see the text below Eq. (3.20) of arXiv 1805.12498) for the given modes and their occupancy
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param num_of_modes The number of modes (including degeneracies) that have been previously calculated. (it is the sum of values in current_occupancy)
@param scale_factor_AZ The scale factor that has been used to scale the matrix elements of AZ =returned by reference)
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
virtual matrix CreateAZ( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, const size_t& total_num_of_occupancy, double &scale_factor_AZ );




/**
@brief Call to scale the input matrix according to according to Eq (2.11) of in arXiv 1805.12498
*/
virtual void ScaleMatrix();



}; //PowerTraceHafnianRecursive_Tasks


#endif // CPYTHON





} // PIC

#endif
