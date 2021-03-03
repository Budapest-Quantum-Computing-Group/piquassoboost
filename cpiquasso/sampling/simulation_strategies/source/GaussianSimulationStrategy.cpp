#include <iostream>
#include "GaussianSimulationStrategy.h"
#include "PowerTraceHafnian.h"
#include <math.h>
#include <tbb/tbb.h>
#include <chrono>


namespace pic {

    double rand_nums[40] = {0.929965, 0.961441, 0.46097, 0.090787, 0.137104, 0.499059, 0.951187, 0.373533, 0.634074, 0.0886671, 0.0856861, 0.999702, 0.419755, 0.376557, 0.947568, 0.705106, 0.0520666, 0.45318,
            0.874288, 0.656594, 0.287817, 0.484918, 0.854716, 0.31408, 0.516911, 0.374158, 0.0124914, 0.878496, 0.322593, 0.699271, 0.0583747, 0.56629, 0.195314, 0.00059639, 0.443711, 0.652659, 0.350379, 0.839752, 0.710161, 0.28553};
    int rand_num_idx = 0;

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategy::GaussianSimulationStrategy() {

    // seeding the random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);

}


/**
@brief Constructor of the class.
@param interferometer_matrix_in The matrix describing the interferometer
@return Returns with the instance of the class.
*/
GaussianSimulationStrategy::GaussianSimulationStrategy( matrix &covariance_matrix_in ) {

    Update_covariance_matrix( covariance_matrix_in );

    // seeding the random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);
}


/**
@brief Destructor of the class
*/
GaussianSimulationStrategy::~GaussianSimulationStrategy() {
}

/**
@brief Call to update the memor address of the stored matrix iinterferometer_matrix
@param covariance_matrix_in The covariance matrix describing the gaussian state
*/
void
GaussianSimulationStrategy::Update_covariance_matrix( matrix &covariance_matrix_in ) {

    covariance_matrix = covariance_matrix_in;

}



/**
@brief Call to determine the resultant state after traversing through linear interferometer.
@return Returns with the resultant state after traversing through linear interferometer.
*/
std::vector<PicState_int64>
GaussianSimulationStrategy::simulate( int samples_number ) {


    // preallocate the memory for the output states
    std::vector<PicState_int64> samples;

    return samples;
}









} // PIC
