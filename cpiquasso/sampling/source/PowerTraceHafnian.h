#ifndef PowerTraceHafnian_H
#define PowerTraceHafnian_H

#include "matrix.h"
#include "matrix32.h"


namespace pic {



/**
@brief Class to calculate the hafnian of a complex matrix by the power trace method
*/
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
@param mtx_in A symmetric matrix for which the hafnian is calculated. (For example a covariance matrix of the Gaussian state.)
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
virtual void Update_mtx( matrix &mtx_in);



}; //PowerTraceHafnian


} // PIC

#endif
