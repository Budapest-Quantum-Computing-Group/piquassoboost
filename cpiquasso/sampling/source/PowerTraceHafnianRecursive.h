#ifndef PowerTraceHafnianRecursive_H
#define PowerTraceHafnianRecursive_H

#include "PowerTraceHafnian.h"
#include "PicState.h"


namespace pic {



/**
@brief Class to calculate the hafnian of a complex matrix by the power trace method. This algorithm is designed to support gaussian boson sampling simulations, it is not a general
purpose hafnian calculator. This algorithm accounts for the repeated occupancy in the covariance matrix.
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


/**
@brief ??????????????????
@return Returns with the calculated hafnian
*/
void IterateOverSelectedoccupancy( std::vector<unsigned char>& selected_occupancy, PicState_int64& filling_factors, size_t mode_to_iterate, Complex16& hafnian );


Complex16 CalculatePartialHafnian( std::vector<unsigned char>& selected_occupancy, PicState_int64& filling_factors );


/**
@brief ??????????????????
@return Returns with the calculated hafnian
*/
matrix
CreateAZ( std::vector<unsigned char>& selected_occupancy, PicState_int64& filling_factors, const size_t& total_num_of_occupancy );





/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
virtual Complex16 calculate_tmp();




protected:


}; //PowerTraceHafnianRecursive


} // PIC

#endif
