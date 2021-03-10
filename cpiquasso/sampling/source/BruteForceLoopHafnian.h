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
