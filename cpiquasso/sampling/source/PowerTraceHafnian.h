#ifndef PowerTraceHafnian_H
#define PowerTraceHafnian_H

#include "matrix.h"


namespace pic {



/**
@brief Class to calculate the hafnian of a complex matrix by the power trace method
*/
class PowerTraceHafnian {

protected:
    /// The covariance matrix of the Gaussian state.
    matrix mtx;


public:


/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
PowerTraceHafnian();

/**
@brief Constructor of the class.
@param mtx_in The covariance matrix of the Gaussian state.
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
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void Update_mtx( matrix &mtx_in);



}; //PowerTraceHafnian


} // PIC

#endif
