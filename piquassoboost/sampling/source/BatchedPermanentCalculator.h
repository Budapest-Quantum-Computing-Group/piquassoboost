#ifndef BatchednPermanentCalculator_H
#define BatchednPermanentCalculator_H

#include "matrix.h"
//#include "matrix32.h"
//#include "matrix_real.h"
#include "PicState.h"
//#include <vector>
//#include "PicVector.hpp"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif


#define GlynnRep 0
#define ChinHuh 1
#define GlynnRepSingleDFE 2
#define GlynnRepDualDFE 3
#define GlynnRepMultiSingleDFE 4
#define GlynnRepMultiDualDFE 5


namespace pic {



/**
@brief Class to accumulate multiple matrices and calculate the permanents for them in one shot. Matrices with row repeatings are split into the sum of smaller permanent problems. The purposo of the class is to upstream multiple problems to DFE engines in once. For CPU implementations it has no further advantage
*/
class BatchednPermanentCalculator {

protected:
    /// The matrix describing the interferometer
    matrix interferometer_matrix;
    /// Unitaries describing a quantum circuit
    PicStates input_states;
    /// Unitaries describing a quantum circuit
    PicStates output_states;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
BatchednPermanentCalculator();


/**
@brief Constructor of the class.
@return Returns with the instance of the class.
*/
BatchednPermanentCalculator(matrix& interferometer_matrix_in);


/**
@brief Destructor of the class
*/
~BatchednPermanentCalculator();




/**
@brief Call to calculate the permanent via Glynn formula scaling with n*2^n. (Does not use gray coding, but does the calculation is similar but scalable fashion)
@param mtx An effective scattering matrix of a boson sampling instance
@param input_state An input state
@param output_state An output state
*/
void add( PicState_int64& input_state, PicState_int64& output_state );


/**
@brief Call to calculate the permanents of the accumulated matrices.
@return Returns with the vector of the calculated permanents
*/
std::vector<Complex16> calculate(int lib=0);


/**
@brief Call toclear the list of matrices and other metadata.
*/
void clear();


}; //BatchednPermanentCalculator





/** @brief Creates a matrix from the `interferometerMatrix` corresponding to the parameters `input_state` and `output_state`.
 *         Corresponding rows and columns are multipled based on output and input states.
 *  @param interferometerMatrix Unitary matrix describing a quantum circuit
 *  @param input_state_in The input state
 *  @param output_state_in The output state
 *  @return Returns with the created matrix
 */
matrix
adaptInterferometerGlynnMultiplied(
    matrix& interferometerMatrix,
    PicState_int64 &input_state,
    PicState_int64 &output_state
);


/** @brief Creates a matrix from the `interferometerMatrix` corresponding to 
 *         the parameters `input_state` and `output_state`.
 *         Does not adapt input and ouput states. They have to be adapted explicitly.
 *         Those matrix rows and columns remain in the adapted matrix where the multiplicity
 *         given by the input and ouput states is nonzero.
 *  @param interferometerMatrix Unitary matrix describing a quantum circuit
 *  @param input_state_in The input state
 *  @param output_state_in The output state
 *  @return Returns with the created matrix
 */
matrix
adaptInterferometer(
    matrix& interferometerMatrix,
    PicState_int64 &input_state,
    PicState_int64 &output_state
);




} // PIC

#endif // GlynnPermanentCalculatorRepeated_H
