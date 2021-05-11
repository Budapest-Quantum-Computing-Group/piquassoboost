#ifndef TorontonianRecursive_H
#define TorontonianRecursive_H

#include "Torontonian.h"
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
class TorontonianRecursive : public Torontonian {


protected:

public:

/**
@brief Constructor of the class.
@param mtx_in The covariance matrix of the Gaussian state.
@return Returns with the instance of the class.
*/
TorontonianRecursive( matrix &mtx_in );


/**
@brief Default destructor of the class.
*/
virtual ~TorontonianRecursive();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual double calculate();


}; //PowerTraceHafnianRecursive



// relieve Python extension from TBB functionalities
#ifndef CPYTHON

/**
@brief Class to calculate the hafnian of a complex matrix by a recursive power trace method. This algorithm is designed to support gaussian boson sampling simulations, it is not a general
purpose hafnian calculator. This algorithm accounts for the repeated occupancy in the covariance matrix.
*/
class TorontonianRecursive_Tasks : public Torontonian {


protected:
    // number of modes spanning the gaussian state
    size_t num_of_modes;
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
TorontonianRecursive_Tasks();

/**
@brief Constructor of the class.
@param mtx_in The covariance matrix of the Gaussian state.
@return Returns with the instance of the class.
*/
TorontonianRecursive_Tasks( matrix &mtx_in );


/**
@brief Default destructor of the class.
*/
virtual ~TorontonianRecursive_Tasks();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
double calculate();


/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
double calculate(unsigned long long start_idx, unsigned long long step_idx, unsigned long long max_idx );



protected:

/**
@brief Call to run iterations over the selected modes to calculate partial hafnians
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param mode_to_iterate The mode for which the occupancy numbers are iterated
@param priv_addend Therad local storage for the partial hafnians
@param tg Reference to a tbb::task_group
*/
void IterateOverSelectedModes( const PicVector<char>& selected_modes, int mode_to_iterate, tbb::combinable<long double>& priv_addend, tbb::task_group &tg );


/**
@brief Call to calculate the partial hafnian for given selected modes and their occupancies
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@return Returns with the calculated hafnian
*/
virtual long double CalculatePartialTorontonian( const PicVector<char>& selected_modes );


/**
@brief Call to construct matrix \f$ A^Z \f$ (see the text below Eq. (3.20) of arXiv 1805.12498) for the given modes and their occupancy
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param num_of_modes The number of modes (including degeneracies) that have been previously calculated. (it is the sum of values in current_occupancy)
@param scale_factor_AZ The scale factor that has been used to scale the matrix elements of AZ =returned by reference)
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
matrix
CreateAZ( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, const size_t& total_num_of_occupancy, double &scale_factor_AZ );




}; //TorontonianRecursive_Tasks


#endif // CPYTHON





} // PIC

#endif
