#ifndef PowerTraceLoopHafnian_H
#define PowerTraceLoopHafnian_H

#include "PowerTraceHafnian.h"


namespace pic {



/**
@brief Class to calculate the loop hafnian of a complex matrix by the power trace method
*/
class PowerTraceLoopHafnian : public PowerTraceHafnian{


public:


/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
PowerTraceLoopHafnian();

/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix for which the hafnian is calculated. (For example a covariance matrix of the Gaussian state.)
@return Returns with the instance of the class.
*/
PowerTraceLoopHafnian( matrix &mtx_in );


/**
@brief Call to calculate the loop hafnian of a complex matrix
@return Returns with the calculated loop hafnian
*/
Complex16 calculate();




/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void Update_mtx( matrix &mtx_in);

protected:

/**
@brief Call to scale the input matrix according to according to Eq (2.14) of in arXiv 1805.12498
@param mtx_in Input matrix defined by
*/
virtual void ScaleMatrix();


/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
virtual matrix32 CalculateLoopCorrection(matrix &diag_elements, matrix& cx_diag_elements, matrix& AZ);



}; //PowerTraceLoopHafnian


} // PIC

#endif