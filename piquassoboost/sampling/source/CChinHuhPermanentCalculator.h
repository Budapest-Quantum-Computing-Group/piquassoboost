#ifndef CChinHuhPermanentCalculator_H
#define CChinHuhPermanentCalculator_H

#include "matrix.h"
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
@brief Class representing a matrix Chin-Huh permanent calculator
*/
class CChinHuhPermanentCalculator {

protected:
    /// The effective scattering matrix of a boson sampling instance
    matrix mtx;
    /// The input state
    PicState_int64 input_state;
    /// The output state
    PicState_int64 output_state;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
CChinHuhPermanentCalculator();



/**
@brief Call to calculate the permanent of the effective scattering matrix. Assuming that input state, output state and the matrix are
defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
permanent of an effective scattering matrix related to probability of obtaining output state from given
input state.
@param mtx_in The effective scattering matrix of a boson sampling instance
@param input_state_in The input state
@param output_state_in The output state
@return Returns with the calculated permanent
*/
Complex16 calculate(matrix &mtx, PicState_int64 &input_state, PicState_int64 &output_state);





}; //CChinHuhPermanentCalculator





// relieve Python extension from TBB functionalities
#ifndef CPYTHON


/**
@brief Class to calculate a partial permanent of the effective scattering matrix. (The loop cycles whithin this function gets called are spanned by function CChinHuhPermanentCalculator::calculate)
*/
class PartialPermanentTask {

public:
    ///
    PicVector<int> iter_value;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
PartialPermanentTask();


/**
@brief Call to execute the task to calculate the partial permanent, and spwans additional tasks to calculate partial permanents. The calculated partial permanents are stored within
thread local storages.
@param The current tbb::task_group object from which tasks are spawned
*/
void execute(matrix &mtx, PicState_int64 &input_state, PicVector<int>& input_state_inidices, PicState_int64 &output_state, tbb::combinable<Complex16>& priv_addend, tbb::task_group &g);



}; // partial permanent_Task


#endif // CPYTHON





} // PIC

#endif
