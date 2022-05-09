#ifndef GlynnPermanentCalculator_H
#define GlynnPermanentCalculator_H

#include "matrix.h"
#include "matrix32.h"
#include "PicState.h"
#include <vector>
#include "PicVector.hpp"
#include <tbb/tbb.h>


namespace pic {

/**
@brief Call to print the elements of a container
@param vec a container
@return Returns with the sum of the elements of the container
*/
template <typename Container>
void print_state( Container state );


/**
 * @brief Interface class representing a Glynn permanent calculator
 * 
 * Specifying template parameters can adjust precision of the calculation (double or long double precision)
 * @tparam matrix_type matrix with given precision inside calculation
 * @tparam precision_type scalar precision inside calculation (double or long double)
 */
template<typename matrix_type, typename precision_type>
class GlynnPermanentCalculator {

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculator();



/**
@brief Call to calculate the permanent via Glynn formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion)
@param mtx The effective scattering matrix of a boson sampling instance
@return Returns with the calculated permanent
*/
Complex16 calculate(matrix &mtx);


}; //GlynnPermanentCalculator






/**
 * @brief Class to calculate a partial permanent via Glynn's formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion)
 * 
 * Specifying template parameters can adjust precision of the calculation (double or long double precision)
 * @tparam matrix_type matrix with given precision inside calculation
 * @tparam precision_type scalar precision inside calculation (double or long double)
 */
template<typename matrix_type, typename precision_type>
class GlynnPermanentCalculatorTask {

public:

    /// 2*mtx used in the recursive calls (The storing of this matrix spare many repeating multiplications)
    matrix_type mtx2;
    /// thread local storage for partial permanents
    tbb::combinable<ComplexM<precision_type>> priv_addend;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculatorTask();


/**
@brief Call to calculate the permanent via Glynn formula. scales with n*2^n
@param mtx The effective scattering matrix of a boson sampling instance
@return Returns with the calculated permanent
*/
Complex16 calculate(matrix &mtx);


/**
 * @brief Method to span parallel tasks via iterative function calls. (new task is spanned by altering one element in the vector of deltas)
 * @param colSum The sum of \f$ \delta_j a_{ij} \f$ in Eq. (S2) of arXiv:1606.05836
 * @param sign The current product \f$ \prod\delta_i $\f
 * @param index_min \f$ \delta_j a_{ij} $\f with \f$ 0<i<index_min $\f are kept contstant, while the signs of \f$ \delta_i \f$  with \f$ i>=idx_min $\f are changed.
 */
void IterateOverDeltas( matrix_type& colSum, int sign, int index_min );


}; // partial permanent_Task


/** alias for glynn permanent calculator with long double precision
 */
using GlynnPermanentCalculatorLongDouble = GlynnPermanentCalculator<pic::matrix32, long double>;

/** alias for glynn permanent calculator with double precision
 */
using GlynnPermanentCalculatorDouble = GlynnPermanentCalculator<pic::matrix, double>;

} // PIC

#endif
