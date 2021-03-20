#ifndef PowerTraceLoopHafnian_H
#define PowerTraceLoopHafnian_H

#include "PowerTraceHafnian.h"


namespace pic {



/**
@brief Class to calculate the loop hafnian of a complex matrix by the power trace method
*/
class PowerTraceLoopHafnian : public PowerTraceHafnian{


public:

// reuse the constructor of the base class
using PowerTraceHafnian::PowerTraceHafnian;

/**
@brief Call to calculate the loop hafnian of a complex matrix
@return Returns with the calculated loop hafnian
*/
Complex16 calculate();


/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
matrix32 calculate_loop_correction(matrix &diag_elements, matrix& cx_diag_elements, matrix& AZ);



}; //PowerTraceLoopHafnian


} // PIC

#endif
