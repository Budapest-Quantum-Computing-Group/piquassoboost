#include <iostream>
#include "ThresholdBosonSampling.h"
#include <math.h>
#include <tbb/tbb.h>

#include "Torontonian.h"

#ifdef __MPI__
#include <mpi.h>
#endif // MPI

#include "dot.h"

#include<stdio.h>
#include<stdlib.h>

//#include "lapacke.h"

extern "C" {

#define LAPACK_ROW_MAJOR               101

/// Definition of the LAPACKE_zgetri function from LAPACKE to calculate the LU decomposition of a matrix
int LAPACKE_zgetrf( int matrix_layout, int n, int m, pic::Complex16* a, int lda, int* ipiv );

/// Definition of the LAPACKE_zgetri function from LAPACKE to calculate the inverse of a matirx
int LAPACKE_zgetri( int matrix_layout, int n, pic::Complex16* a, int lda, const int* ipiv );

}



namespace pic {



/**
@brief Function to calculate factorial of a number.
@param n The input number
@return Returns with the factorial of the number
*/
static double factorial(int64_t n) {



    if ( n == 0 ) return 1;
    if ( n == 1 ) return 1;

    int64_t ret=1;

    for (int64_t idx=2; idx<=n; idx++) {
        ret = ret*idx;
    }

    return (double) ret;


}


/**
@brief Call to calculate sum of integers stored in a PicState
@param vec a container if integers
@return Returns with the sum of the elements of the container
*/
static inline int64_t
sum( PicState_int64 &vec) {

    int64_t ret=0;

    size_t element_num = vec.cols;
    int64_t* data = vec.get_data();
    for (size_t idx=0; idx<element_num; idx++ ) {
        ret = ret + data[idx];
    }
    return ret;
}


/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
ThresholdBosonSampling::ThresholdBosonSampling() :
    GaussianSimulationStrategy()
{}


/**
@brief Constructor of the class.
@param covariance_matrix The covariance matrix describing the gaussian state
@return Returns with the instance of the class.
*/
ThresholdBosonSampling::ThresholdBosonSampling( matrix &covariance_matrix )
    : GaussianSimulationStrategy(covariance_matrix, 1) 
{}


/**
@brief Destructor of the class
*/
ThresholdBosonSampling::~ThresholdBosonSampling()
{}

/**
@brief Call to update the memor address of the stored matrix interferometer_matrix
@param covariance_matrix The covariance matrix describing the gaussian state
*/
void
ThresholdBosonSampling::Update_covariance_matrix( matrix &covariance_matrix ) {
    state.Update_covariance_matrix( covariance_matrix );
}


/**
@brief Call to determine the resultant state after traversing through linear interferometer.
@return Returns with the resultant state after traversing through linear interferometer.
*/
std::vector<PicState_int64>
ThresholdBosonSampling::simulate( int samples_number ) {

    // seed the random generator
    srand ( time( NULL) );

    // preallocate the memory for the output states
    std::vector<PicState_int64> samples;
    samples.reserve(samples_number);
    for (int idx=0; idx < samples_number; idx++) {
        PicState_int64&& sample = getSample();
        if (sample.size() > 0 ) {
            samples.push_back(sample);
        }
    }

    return samples;
}



/**
@brief Call to get one sample from the gaussian state
@return Returns with the a sample from a gaussian state
*/
PicState_int64
ThresholdBosonSampling::getSample() {    
    // get husimi covariance matrix. It is stored in state.get_covariance_matrix()
    // convert the sampled Gaussian state into complex amplitude representation
    state.ConvertToComplexAmplitudes();
    // from now the basis is the creation/annihilation operators in a_1,\dots,a_N, a^\dagger_1,\dots, a^\dagger_N ordering.

    PicState_int64 output_sample(0);
    output_sample.number_of_photons = 0;

    // probability of the sampled state
    double current_state_probability = 1.0;

    // The number of modes is equal to dim_over_2 (becose the covariance matrix conatains p,q quadratires)
    // for loop to sample 1,2,3,...dim_over_2 modes
    // These samplings depends from each other by the chain rule of probabilites (see Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time))
    for (size_t mode_idx=1; mode_idx<=dim_over_2; mode_idx++) {
        // modes to be extracted to get reduced gaussian state
        PicState_int64 indices_2_extract(mode_idx);
        for (size_t idx=0; idx<mode_idx; idx++) {
            indices_2_extract[idx] = idx;
        }

        // reduced covariance matrix in reduced gaussian state to the first mode_idx modes
        // get the reduced gaussian state describing the first mode_idx modes
        GaussianState_Cov reduced_state = state.getReducedGaussianState( indices_2_extract );
        

        // since the determinant can be calculated by LU factorization, which is also necessary to calculate the inverse, we
        // calculate the inverse and the determiant in one shot.
        // Calculate Q matrix defined by \sigma + 0.5 * Id
        // Calculate the determinant of Q
        // Caluclate the inverse of Q
        double Qdet(0.0);
        matrix&& Qinv = calc_Qinv( reduced_state, Qdet );

        // calculate the matrix O(k) defined by Id - \sigma(k)^{-1}
        // calculate the Hamilton matrix A defined by Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time
        // O := 1 - Q^-1
        matrix&& O = calc_HamiltonMatrix( Qinv );


        // create a random double that is used to sample from the probabilities
        double rand_num = (double)rand()/RAND_MAX;

#ifdef __MPI__
            // ensure all the processes gets the same random number
            MPI_Bcast(&rand_num, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif // MPI

        // the chosen index of the probabilities
        size_t chosen_index = 0;


        // Calculate if the photon number on the current mode was zero
        // calculate probabilities whether there are any photons on the mode mode_idx

        // current output variable is used for storing the conditions for the conditional probablities.
        // create array for the new output state
        PicState_int64 current_output0(output_sample.size()+1, 0);
        memcpy(current_output0.get_data(), output_sample.get_data(), output_sample.size()*sizeof(int64_t));

        // set the number of photons in the last mode to 0
        current_output0[mode_idx-1] = 0;

        // calculate the probability associated with observing current_output
        double prob0 = calc_probability(Qinv, Qdet, O, current_output0);

        // sometimes the probability is negative which is coming from a negative hafnian.
        prob0 = prob0 > 0 ? prob0 : 0;

        double current_prob0 = prob0 / current_state_probability;

#ifdef DEBUG
        // Calculate if the photon number on the current mode was one
    
        // current output variable is used for storing the conditions for the conditional probablities.
        // create array for the new output state
        PicState_int64 current_output1(output_sample.size()+1, 0);
        memcpy(current_output1.get_data(), output_sample.get_data(), output_sample.size()*sizeof(int64_t));

        // set the number of photons in the last mode to 1
        current_output1[mode_idx-1] = 1;

        // calculate the probability associated with observing current_output
        double prob1 = calc_probability(Qinv, Qdet, O, current_output1);

        // sometimes the probability is negative which is coming from a negative hafnian.
        prob1 = prob1 > 0 ? prob1 : 0;

        double current_prob1 = prob1 / current_state_probability;

        std::cout << "current_prob0: " <<current_prob0 << " current_prob1: "<<current_prob1<<std::endl;

        assert ( std::abs(prob1 - (current_state_probability - prob0) ) < 0.000000000001 );
#endif // DEBUG

        if (current_prob0 > rand_num){
            current_state_probability = prob0;
            chosen_index = 0;
        }
        else{
            current_state_probability = current_state_probability - prob0;
            chosen_index = 1;
        }


        // The sampled current state:
        PicState_int64 current_output(output_sample.size()+1, 0);

        memcpy(current_output.get_data(), output_sample.get_data(), output_sample.size()*sizeof(int64_t));
        current_output[mode_idx-1] = chosen_index;
        current_output.number_of_photons = output_sample.number_of_photons + chosen_index;

        output_sample = current_output;

    }

    return output_sample;

}


/**
@brief Call to calculate the Hamilton matrix A defined by Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.

O = 1 - Qinv

@param Qinv An instace of matrix class conatining the inverse of matrix Q calculated by method get_Qinv.
@return Returns with the Hamilton matrix A.
*/
matrix
ThresholdBosonSampling::calc_HamiltonMatrix( matrix& Qinv ) {
    // multiply by -1 the elements of Qinv and store the result in the corresponding rows of A
    matrix O(Qinv.rows, Qinv.cols);
    for (size_t row_idx = 0; row_idx<O.rows ; row_idx++) {

        for (size_t col_idx = 0; col_idx<O.cols; col_idx++) {
            O[row_idx*O.stride + col_idx] = -1.0 * Qinv[row_idx*Qinv.stride+col_idx];

        }
        O[row_idx*O.stride + row_idx] += 1.0;

    }
    return O;
}


/**
@brief Call to calculate the probability associated with observing output state given by current_output

The calculation is based on Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.

@param Qinv An instace of matrix class conatining the inverse of matrix Q calculated by method get_Qinv.
@param Qdet The determinant of matrix Q.
@param O Hamilton matrix 
@param current_output The current conditions for which the conditional probability is calculated
@return Returns with the calculated probability
*/
double
ThresholdBosonSampling::calc_probability( matrix& Qinv, const double& Qdet, matrix& O, PicState_int64& current_output ) {
    // calculate the normalization factor defined by the square root of the determinant of matrix Q
    double Normalization = 1.0/sqrt(Qdet);


#ifdef DEBUG
    if (Qdet<0) {
        std::cout << "Determinant of matrix Q is negative" << std::endl;
        exit(-1);
    }
#endif // DEBUG

    // create Matrix O_S according to the main text below Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.
    matrix&& O_S = create_O_S( O, current_output );

    /// Calculate the torontonian of O_S
    Torontonian torontonian_calculator(O_S);
    double torontonian = torontonian_calculator.calculate();

    // calculate the probability associated with the current output
    double prob = Normalization*torontonian;


    return prob;

}



/**
@brief Call to create matrix O_S according to the main text below Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.
@param O Hamilton matrix O
@param current_output The fock representation of the current output for which the probability is calculated
@return Returns with the O_S matrix
*/
matrix
ThresholdBosonSampling::create_O_S( matrix& O, PicState_int64& current_output ) {

    size_t dim_O_S = sum(current_output);
    size_t dim_O = current_output.size();

    matrix O_S(2*dim_O_S, 2*dim_O_S);
    memset(O_S.get_data(), 0, O_S.size()*sizeof(Complex16));

    size_t row_idx_O_S = 0;
    for (size_t idx_output=0; idx_output<current_output.size(); idx_output++) {
        // we inserting element to current row if the current output is 1
        if (current_output[idx_output]) {
            size_t row_offset = row_idx_O_S*O_S.stride;
            size_t row_offset_O = idx_output*O.stride;
            size_t col_idx_O_S = 0;
            // insert column elements to the upper left block
            for (size_t jdx_output=0; jdx_output<current_output.size(); jdx_output++) {
                // we inserting element if the current output is 1
                if (current_output[jdx_output]) {
                    if ( (row_idx_O_S == col_idx_O_S) || (idx_output != jdx_output) ) {
                        O_S[row_offset + col_idx_O_S] = O[row_offset_O + jdx_output];
                    }
                    col_idx_O_S++;
                }
            }

            col_idx_O_S = 0;
            // insert column elements to the upper right block
            for (size_t jdx_output=0; jdx_output<current_output.size(); jdx_output++) {
                // we inserting element if the current output is 1
                if (current_output[jdx_output]) {
                    O_S[row_offset + col_idx_O_S + dim_O_S] = O[row_offset_O + jdx_output + dim_O];
                    col_idx_O_S++;
                }

            }

            row_offset = (row_idx_O_S+dim_O_S)*O_S.stride;
            row_offset_O = (idx_output+dim_O)*O.stride;
            col_idx_O_S = 0;
            // insert column elements to the lower left block
            for (size_t jdx_output=0; jdx_output<current_output.size(); jdx_output++) {
                // we inserting element if the current output is 1
                if (current_output[jdx_output]) {
                    O_S[row_offset + col_idx_O_S] = O[row_offset_O + jdx_output];
                    col_idx_O_S++;
                }
            }

            col_idx_O_S = 0;
            // insert column elements to the lower right block
            for (size_t jdx_output=0; jdx_output<current_output.size(); jdx_output++) {
                // we inserting element if the current output is 1
                if (current_output[jdx_output]) {
                    if ( (row_idx_O_S == col_idx_O_S) || (idx_output != jdx_output) ) {
                        O_S[row_offset + col_idx_O_S + dim_O_S] = O[row_offset_O + jdx_output + dim_O];
                    }
                    col_idx_O_S++;
                }

            }


            row_idx_O_S++;
        }


    }

    return O_S;
}



} // PIC
