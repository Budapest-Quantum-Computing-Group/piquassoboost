#ifndef PowerTraceHafnianRecursive_H
#define PowerTraceHafnianRecursive_H

#include "PowerTraceHafnian.h"
#include "PicState.h"


namespace pic {



/**
@brief Class to calculate the hafnian of a complex matrix by the power trace method. This algorithm is designed to support gaussian boson sampling simulations, it is not a general
purpose hafnian calculator. This algorithm accounts for the repeated modes in the covariance matrix.
*/
class PowerTraceHafnianRecursive : public PowerTraceHafnian {


protected:
    /// An array describing the modes to be used to calculate the hafnian. The i-th mode is repeated modes[i] times.
    PicState_int64 modes;

public:

/**
@brief Constructor of the class.
@param mtx_in The covariance matrix of the Gaussian state.
@param modes An array describing the modes to be used to calculate the hafnian. The i-th mode is repeated modes[i] times.
@return Returns with the instance of the class.
*/
PowerTraceHafnianRecursive( matrix &mtx_in, PicState_int64& modes_in );


/**
@brief Default destructor of the class.
*/
virtual ~PowerTraceHafnianRecursive();

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate();


/**
@brief ??????????????????
@return Returns with the calculated hafnian
*/
void IterateOverSelectedModes( std::vector<unsigned char>& selected_modes, PicState_int64& filling_factors, size_t mode_to_iterate, Complex16& hafnian );


Complex16 CalculatePartialHafnian( std::vector<unsigned char>& selected_modes, PicState_int64& filling_factors );

/**
@brief ??????????????????
@return Returns with the calculated hafnian
*/
matrix getPermutedMatrix();


/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate_tmp();




protected:


}; //PowerTraceHafnianRecursive


} // PIC

#endif
