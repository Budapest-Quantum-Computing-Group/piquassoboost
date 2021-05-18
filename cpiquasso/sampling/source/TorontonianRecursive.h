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
    /// number of modes spanning the gaussian state
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
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void Update_mtx( matrix &mtx_in);



protected:

/**
@brief Call to run iterations over the selected modes to calculate partial torontonians
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A^Z.
@param L Matrix conatining partial Cholesky decomposition if the initial matrix to be reused
@param hole_to_iterate The index indicating which hole index should be iterated
@param priv_addend Therad local storage for the partial torontonians
@param tg Reference to a tbb::task_group
*/
void IterateOverSelectedModes( const PicVector<size_t>& selected_modes, int mode_to_iterate, matrix &L, const size_t reuse_index, tbb::combinable<long double>& priv_addend, tbb::task_group &tg );


/**
@brief Call to calculate the partial torontonian for given selected modes and their occupancies
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A^Z.
@param L Matrix conatining partial Cholesky decomposition if the initial matrix to be reused
@param reuse_index Index labeling the highest mode for which previous Cholesky decomposition can be reused.
@return Returns with the calculated torontonian
*/
virtual long double CalculatePartialTorontonian( const PicVector<size_t>& selected_modes, matrix L, const size_t reuse_index );


/**
@brief Call to construct matrix \f$ A^Z \f$ (see the text below Eq. (2) of arXiv 2009.01177) for the given modes
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A^Z.
@param L Matrix conatining partial Cholesky decomposition if the initial matrix to be reused
@param reuse_index Index labeling the highest mode for which previous Cholesky decomposition can be reused.
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
matrix
CreateAZ( const PicVector<size_t>& selected_index_holes, matrix L, const size_t reuse_index );




}; //TorontonianRecursive_Tasks


#endif // CPYTHON





} // PIC

#endif
