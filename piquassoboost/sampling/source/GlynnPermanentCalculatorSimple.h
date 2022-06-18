#ifndef GlynnPermanentCalculatorSimple_H
#define GlynnPermanentCalculatorSimple_H

#include "matrix.h"
#include "matrix32.h"
#include <vector>

// avoid tbb usage at python interface
#ifndef CPYTHON
#include <tbb/tbb.h>
#endif


namespace pic {

/**
 * @brief Interface class representing a Glynn permanent calculator
 * 
 * Specifying template parameters can adjust precision of the calculation (double or long double precision)
 * @tparam matrix_type matrix with given precision inside calculation
 * @tparam precision_type scalar precision inside calculation (double or long double)
 */
template<typename matrix_type, typename precision_type>
class GlynnPermanentCalculatorSimple {

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculatorSimple();



/**
@brief Call to calculate the permanent via Glynn formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion)
@param mtx The effective scattering matrix of a boson sampling instance
@return Returns with the calculated permanent
*/
Complex16 calculate(matrix &mtx);


}; //GlynnPermanentCalculator


// avoid tbb usage at python interface
#ifndef CPYTHON
template<typename matrix_type, typename precision_type>
class GlynnPermanentCalculatorSimpleTask {
    
public:



    /** tbb thread safe container for partial sums */
    tbb::combinable<ComplexM<precision_type>> partialSums;

    /**
    * @brief Call to calculate the permanent via Glynn formula scaling with n*2^n.
    *        This function contains tbb code.
    * @param mtx The effective scattering matrix of a boson sampling instance
    * @return Returns with the calculated permanent
    */
    Complex16 calculate(matrix &mtx);


}; //GlynnPermanentCalculatorTask

#endif // CPYTHON


/** alias for glynn permanent calculator with long double precision
 */
using GlynnPermanentCalculatorSimpleLongDouble = GlynnPermanentCalculatorSimple<pic::matrix32, long double>;

/** alias for glynn permanent calculator with double precision
 */
using GlynnPermanentCalculatorSimpleDouble = GlynnPermanentCalculatorSimple<pic::matrix, double>;

} // PIC

#endif // GlynnPermanentCalculatorSimple_H
