#ifndef GlynnPermanentCalculatorRecursive_H
#define GlynnPermanentCalculatorRecursive_H

#include "matrix.h"
#include "matrix32.h"
#include "matrix_real.h"
#include "array_int.h"
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
class GlynnPermanentCalculatorRecursive {

protected:
    /// Unitary describing a quantum circuit
    matrix mtx;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculatorRecursive();



/**
@brief Call to calculate the permanent via Glynn formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion)
@param mtx The effective scattering matrix of a boson sampling instance
@param input_state The input state
@param output_state The output state
@return Returns with the calculated permanent
*/
Complex16 calculate(
    matrix &mtx,
    PicState_int64& input_state,
    PicState_int64& output_state
);


}; //GlynnPermanentCalculatorRecursive





// relieve Python extension from TBB functionalities
#ifndef CPYTHON


/**
@brief Class to calculate a partial permanent via Glynn's formula scaling with n*2^n.
(Does not use gray coding, but does the calculation is similar but scalable fashion) 
*/
class GlynnPermanentCalculatorRecursiveTask {

public:

    /// Unitary describing a quantum circuit
    matrix mtx;
    /// 2*mtx used in the recursive calls (The storing of thos matrix
    /// spare many repeating multiplications)
    matrix32 mtx2;
    /// thread local storage for partial permanents
    tbb::combinable<ComplexM<long double>> priv_addend;

    /// numbers describing the row multiplicity
    array_int& row_multiplicities;
    /// numbers describing the column multiplicity
    array_int& col_multiplicities;

    /// limit of the delta values, all same as the row multiplicity except
    /// the first nonzero one which is one smaller
    array_int deltaLimits;
    /// minimal nonzero index of the row_multiplicity
    int minimalIndex;
public:

/**
@brief Default constructor of the class.
@param mtx Unitary describing a quantum circuit
@param row_multiplicities vector describing the row multiplicity
@param col_multiplicities vector describing the column multiplicity
@return Returns with the instance of the class.
*/
GlynnPermanentCalculatorRecursiveTask(
    matrix &mtx,
    array_int& row_multiplicities,
    array_int& col_multiplicities
);


/**
@brief Call to calculate the permanent via Glynn formula. scales with n*2^n
@return Returns with the calculated permanent
*/
Complex16 calculate();


/**
@brief Method to span parallel tasks via iterative function calls.
(new task is spanned by altering one element in the vector of deltas)
@param colSum The sum of \f$ \delta_j a_{ij} \f$ in Eq. (S2) of arXiv:1606.05836
@param sign The current product \f$ \prod\delta_i $\f
@param index_min \f$ \delta_j a_{ij} $\f with \f$ 0<i<index_min $\f are kept constant, while the signs of \f$ \delta_i \f$  with \f$ i>=idx_min $\f are changed.
@param currentMultiplicity multiplicity of the current delta vector
*/
void IterateOverDeltas(
    matrix32& colSum,
    int sign,
    int index_min,
    int currentMultiplicity
);


}; // partial permanent_Task


#endif // CPYTHON




} // PIC

#endif // GlynnPermanentCalculatorRecursive_H
