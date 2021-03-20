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
@param mtx_in The covariance matrix of the Gaussian state.
@param occupancy An array describing the occupancy to be used to calculate the hafnian. The i-th mode is repeated occupancy[i] times.
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
class PowerTraceHafnianRecursive_Tasks {


protected:
    /// The symmetric matrix for which the hafnian is calculated (i.e. the covariance matrix of the Gaussian state.)/// The input matrix. Must be symmetric
    matrix mtx_orig;
    /** The scaled input matrix for which the calculations are performed.
    If the mean magnitude of the matrix elements is one, the treshold of quad precision can be set to higher values.
    */
    matrix mtx;
    /// The scale factor of the input matric
    double scale_factor;
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
@brief Constructor of the class.
@param mtx_in The covariance matrix of the Gaussian state.
@param occupancy An array describing the occupancy to be used to calculate the hafnian. The i-th mode is repeated occupancy[i] times.
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
virtual Complex16 calculate();


protected:

/**
@brief Call to run iterations over the selected modes to calculate partial hafnians
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param mode_to_iterate The mode for which the occupancy numbers are iterated
@param priv_addend Therad local storage for the partial hafnians
@param tg Reference to a tbb::task_group
*/
void IterateOverSelectedModes( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, size_t mode_to_iterate, tbb::combinable<Complex32>& priv_addend, tbb::task_group &tg );


/**
@brief Call to calculate the partial hafnian for given selected modes and their occupancies
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@return Returns with the calculated hafnian
*/
Complex32 CalculatePartialHafnian( const PicVector<char>& selected_modes, const  PicState_int64& current_occupancy );


/**
@brief Call to construct matrix \f$ A^Z \f$ (see the text below Eq. (3.20) of arXiv 1805.12498) for the given modes and their occupancy
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param num_of_modes The number of modes (including degeneracies) that have been previously calculated. (it is the sum of values in current_occupancy)
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
matrix
CreateAZ( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, const size_t& total_num_of_occupancy );




}; //PowerTraceHafnianRecursive_Tasks


#endif





} // PIC

#endif
