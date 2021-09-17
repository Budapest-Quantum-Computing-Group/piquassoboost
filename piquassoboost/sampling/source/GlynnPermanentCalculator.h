#ifndef GlynnPermanentCalculator_H
#define GlynnPermanentCalculator_H

#include "matrix.h"
#include "matrix32.h"
#include "PicState.h"
#include <vector>
#include "PicVector.hpp"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif


namespace pic {

/**
@brief Call to print the elements of a container
@param vec a container
@return Returns with the sum of the elements of the container
*/
template <typename Container>
void print_state( Container state );


/**
@brief Interface class representing a Glynn permanent calculator
*/
class GlynnPermanentCalculator {

protected:
    /// Unitary describing a quantum circuit
    matrix mtx;

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





// relieve Python extension from TBB functionalities
#ifndef CPYTHON


/**
@brief Class to calculate a partial permanent via Glynn's formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion) 
*/
class GlynnPermanentCalculatorTask {

public:

    /// Unitary describing a quantum circuit
    matrix mtx;
    /// 2*mtx used in the recursive calls (The storing of thos matrix spare many repeating multiplications)
    matrix32 mtx2;
    /// thread local storage for partial permanents
    tbb::combinable<ComplexM<long double>> priv_addend;

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
@brief Method to span parallel tasks via iterative function calls. (new task is spanned by altering one element in the vector of deltas)
@param colSum The sum of \f$ \delta_j a_{ij} \f$ in Eq. (S2) of arXiv:1606.05836
@param sign The current product \f$ \prod\delta_i $\f
@param index_min \f$ \delta_j a_{ij} $\f with \f$ 0<i<index_min $\f are kept contstant, while the signs of \f$ \delta_i \f$  with \f$ i>=idx_min $\f are changed.
*/
void IterateOverDeltas( matrix32& colSum, int sign, int index_min );


}; // partial permanent_Task


#endif // CPYTHON




} // PIC

#endif
