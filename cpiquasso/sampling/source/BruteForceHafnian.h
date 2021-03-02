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
