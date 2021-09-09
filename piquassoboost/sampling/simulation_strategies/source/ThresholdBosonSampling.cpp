/**
 * Copyright 2021 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include "ThresholdBosonSampling.h"
#include <math.h>
#include <tbb/tbb.h>

#include "TorontonianRecursive.h"
#include "Torontonian.h"
#include "GaussianState_Cov.h"

#ifdef __MPI__
#include <mpi.h>
#endif // MPI

#include "dot.h"

#include<stdio.h>
#include<stdlib.h>


extern "C" {

#define LAPACK_ROW_MAJOR               101

/// Definition of the LAPACKE_dgetrf function from LAPACKE to calculate the LU decomposition of a real matrix
int LAPACKE_dgetrf( int matrix_layout, int n, int m, double* a, int lda, int* ipiv );

/// Definition of the LAPACKE_dgetri function from LAPACKE to calculate the inverse of a real matrix
int LAPACKE_dgetri( int matrix_layout, int n, double* a, int lda, const int* ipiv );

}




namespace pic {



ThresholdMeasurementSubstate::ThresholdMeasurementSubstate( matrix_real& O, double Qdet_sqrt_rec )
    : O(O)
    , Qdet_sqrt_rec(Qdet_sqrt_rec)
{}


/**
@brief Call to calculate sum of integers stored in a PicState
@param vec a container of integers
@return Returns with the sum of the elements of the container
*/
static inline int64_t
sum( PicState_int64& vec) {

    int64_t ret=0;

    size_t element_num = vec.cols;
    int64_t* data = vec.get_data();
    for (size_t idx=0; idx<element_num; idx++ ) {
        ret = ret + data[idx];
    }
    return ret;
}


/**
@brief Constructor of the class.
@param covariance_matrix The covariance matrix describing the gaussian state
@return Returns with the instance of the class.
*/
ThresholdBosonSampling::ThresholdBosonSampling( matrix_real& covariance_matrix_in )
    : covariance_matrix(covariance_matrix_in)
{
#ifdef DEBUG  
    assert(covariance_matrix_real.rows == covariance_matrix_real.cols);
    assert(covariance_matrix_real.rows % 2 == 0);
#endif
    // The number of the input modes stored by the covariance matrix
    number_of_modes = covariance_matrix.rows / 2;
    // Space for storing the threshold measurement specific datas for a sample which are equal in all samples.
    pmfs = std::unordered_map<PicState_int64, double, PicStateHash>();
}


/**
@brief Destructor of the class
*/
ThresholdBosonSampling::~ThresholdBosonSampling()
{}

/**
@brief Call to update the memory address of the stored matrix covariance_matrix
@param covariance_matrix The covariance matrix describing the gaussian state
*/
void
ThresholdBosonSampling::Update_covariance_matrix( matrix_real& covariance_matrix_in ) {
    covariance_matrix = covariance_matrix_in;
}


/**
@brief Call to determine the resultant state after traversing through linear interferometer.
@param samples_number The number of shots for which the output should be determined
@return Returns with the resultant state after traversing through linear interferometer.
*/
std::vector<PicState_int64>
ThresholdBosonSampling::simulate( int samples_number ) {
    // calculate the data which are equal for all samples
    fillSubstates( covariance_matrix, number_of_modes );

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

void ThresholdBosonSampling::fillSubstates( matrix_real& matrix, int mode_number ){

    // initialize the substates vector with mode_number+1 elements for the empty modes ... all modes
    substates.reserve(mode_number+1);

    matrix_real empty_matrix_real(0,0);
    substates.push_back(ThresholdMeasurementSubstate(empty_matrix_real, 1.0));

    for (int mode_idx = 1; mode_idx < mode_number+1; mode_idx++){
        // modes to be extracted to get reduced gaussian state
        PicState_int64 indices_2_extract(mode_idx);
        for (int idx = 0; idx < mode_idx; idx++) {
            indices_2_extract[idx] = idx;
        }

        // reduced covariance matrix in reduced gaussian state to the first mode_idx modes
        // get the reduced gaussian state describing the first mode_idx modes
        matrix_real reduced_matrix = reduce( matrix, indices_2_extract );

        // since the determinant can be calculated by LU factorization, which is also necessary to calculate the inverse, we
        // calculate the inverse and the determiant in one shot.
        // Calculate Q matrix defined by \sigma + 0.5 * Id
        // Calculate the determinant of Q
        // Caluclate the inverse of Q
        double Qdet(1.0);
        matrix_real&& Qinv = calc_Qinv( reduced_matrix, Qdet );

        double Qdet_sqrt_rec = 1.0 / std::sqrt(Qdet);

        //// This depends only on mode number and mode idx

        // calculate the matrix O(k) defined by Id - \sigma(k)^{-1}
        // calculate the Hamilton matrix O defined by Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time
        // O := 1 - Q^-1
        matrix_real&& O = calc_HamiltonMatrix( Qinv );

        substates.push_back( ThresholdMeasurementSubstate(O, Qdet_sqrt_rec) );
    }
}



/**
@brief Call to get one sample from the gaussian state
@return Returns with the a sample from a gaussian state
*/
PicState_int64
ThresholdBosonSampling::getSample() {

    PicState_int64 output_sample(0);
    output_sample.number_of_photons = 0;

    // probability of the sampled state
    double last_probability = 1.0;


    // The number of modes is equal to number_of_modes (because the covariance matrix contains p,q quadratires)
    // for loop to sample 1,2,3,...number_of_modes modes
    // These samplings depends from each other by the chain rule of probabilites (see Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time))
    for (size_t mode_idx=1; mode_idx<=number_of_modes; mode_idx++) {

        // create a random double that is used to sample from the probabilities
        double rand_num = (double)rand()/RAND_MAX;

#ifdef __MPI__
        // ensure all the processes gets the same random number
        MPI_Bcast(&rand_num, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif // MPI

        // the chosen index of the probabilities, initial value: 0
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
        double prob0 = calc_probability_from_cache( current_output0 );

        // sometimes the probability is negative which is coming from a negative hafnian.
        prob0 = prob0 > 0 ? prob0 : 0;

        double current_prob0 = prob0 / last_probability;

#ifdef DEBUG
        // Calculate if the photon number on the current mode was one
    
        // current output variable is used for storing the conditions for the conditional probablities.
        // create array for the new output state
        PicState_int64 current_output1(output_sample.size()+1, 0);
        memcpy(current_output1.get_data(), output_sample.get_data(), output_sample.size()*sizeof(int64_t));

        // set the number of photons in the last mode to 1
        current_output1[mode_idx-1] = 1;

        // calculate the probability associated with observing current_output
        double prob1 = calc_probability_from_cache( current_output1 );

        // sometimes the probability is negative which is coming from a negative hafnian.
        prob1 = prob1 > 0 ? prob1 : 0;

        double current_prob1 = prob1 / last_probability;

        assert ( std::abs(prob1 - (last_probability - prob0) ) < 0.000000000001 );
#endif // DEBUG
        if (current_prob0 > rand_num){
            last_probability = prob0;
            chosen_index = 0;
        }
        else{
            last_probability = last_probability - prob0;
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
matrix_real
ThresholdBosonSampling::calc_HamiltonMatrix( matrix_real& Qinv ) {
    // multiply by -1 the elements of Qinv and store the result in the corresponding rows of A
    matrix_real O(Qinv.rows, Qinv.cols);
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

If the size of the given current_output is smaller then limit_for_using_pmfs then we use cache for having faster probability calculation.
Otherwise we just calculate the probability.

@param current_output The current conditions for which the conditional probability is calculated
@return Returns with the calculated probability
*/
double
ThresholdBosonSampling::calc_probability_from_cache( PicState_int64& current_output ) {

    if (current_output.size() < limit_for_using_pmfs){
        // find whether the current probability was already calculated
        auto current_prob_iter = pmfs.find(current_output);

        // checks whether the pmfs contains the current output already
        if (pmfs.end() != current_prob_iter){
            // return with the stored value from pmfs
            return current_prob_iter->second;
        // otherwise calculate the probability
        }else{
            // Call the normal calc_probability method which does not store the results
            double prob = calc_probability( current_output );

            // Save the current probability into the current output
            pmfs.insert( {current_output, prob} );

            return prob;
        }
    }else{
        return calc_probability( current_output );
    }
}


/**
@brief Call to calculate the probability associated with observing output state given by current_output

The calculation is based on Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.

@param current_output The current conditions for which the conditional probability is calculated
@return Returns with the calculated probability
*/
double
ThresholdBosonSampling::calc_probability( PicState_int64& current_output ) {
    int mode_counter = current_output.size();

    // calculate the normalization factor defined by the square root of the determinant of matrix Q
    const double Qdet_sqrt_rec = substates[mode_counter].Qdet_sqrt_rec;
    
    // get the őrecalculated matrix O 
    matrix_real& O = substates[mode_counter].O;


#ifdef DEBUG
    if (Qdet_sqrt_rec<0) {
        std::cout << "Determinant of matrix Q is negative" << std::endl;
        exit(-1);
    }
#endif // DEBUG

    // create Matrix O_S according to the main text below Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.
    matrix_real&& O_S = create_O_S( O, current_output );

    /// Calculate the torontonian of O_S
    TorontonianRecursive torontonian_calculator(O_S);

    //Torontonian torontonian_calculator(O_S);
    const bool use_extended = true;
    double torontonian = torontonian_calculator.calculate(use_extended);

    // calculate the probability associated with the current output
    double prob = Qdet_sqrt_rec*torontonian;
    pmfs.insert( {current_output, prob} );

    return prob;
}

/**
@brief Call to create matrix O_S according to the main text below Eq. (14) of Ref. Exact simulation of Gaussian boson sampling in polynomial space and exponential time.
@param O Hamilton matrix O
@param current_output The fock representation of the current output for which the probability is calculated
@return Returns with the O_S matrix
*/
matrix_real
ThresholdBosonSampling::create_O_S( matrix_real& O, PicState_int64& current_output ) {

    size_t dim_O_S = sum(current_output);
    size_t dim_O = current_output.size();

    matrix_real O_S(2*dim_O_S, 2*dim_O_S);
    memset(O_S.get_data(), 0, O_S.size()*sizeof(double));

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


// TODO: The reduction could be placed to a separate place, since this is not specific
// to this class.
matrix_real
ThresholdBosonSampling::reduce( matrix_real& matrix, PicState_int64& indices_to_reduce ){

    size_t total_number_of_modes = matrix.rows/2;

    if (total_number_of_modes == 0) {
        std::cout << "ThresholdBosonSampling::reduce: There is no covariance matrix to be reduced. Exiting" << std::endl;
        exit(-1);
    }


    // the number of modes to be extracted
    size_t number_of_modes = indices_to_reduce.size();
    if (number_of_modes == total_number_of_modes) {
        return matrix_real(matrix);
    }
    else if ( number_of_modes > total_number_of_modes) {
        std::cout << "ThresholdBosonSampling::reduce: The number of modes to be extracted is larger than the possible number of modes. Exiting" << std::endl;
        exit(-1);
    }

    // allocate data for the reduced covariance matrix
    matrix_real matrix_reduced(number_of_modes*2, number_of_modes*2);  // the size of the covariance matrix must be the double of the number of modes
    double* matrix_reduced_data = matrix_reduced.get_data();
    double* matrix_data = matrix.get_data();

    size_t mode_idx = 0;
    size_t col_range = 1;
    // loop over the col indices to be transformed (indices are stored in attribute modes)
    while (true) {

        // condition to exit the loop: if there are no further columns then we exit the loop
        if ( mode_idx >= number_of_modes) {
            break;
        }

        // determine contiguous memory slices (column indices) to be extracted
        while (true) {

            // condition to exit the loop: if the difference of successive indices is greater than 1, the end of the contiguous memory slice is determined
            if ( mode_idx+col_range >= number_of_modes || indices_to_reduce[mode_idx+col_range] - indices_to_reduce[mode_idx+col_range-1] != 1 ) {
                break;
            }
            else {
                col_range = col_range + 1;
            }

        }

        // the column index in the matrix from we are bout the extract columns
        size_t col_idx = indices_to_reduce[mode_idx];

        // row-wise loop to extract the q quadrature columns from the covariance matrix
        for(size_t mode_row_idx=0; mode_row_idx<number_of_modes; mode_row_idx++) {

            // col-wise extraction of the q quadratures from the covariance matrix
            size_t cov_reduced_offset = mode_row_idx*matrix_reduced.stride + mode_idx;
            size_t cov_offset = indices_to_reduce[mode_row_idx]*matrix.stride + col_idx;
            memcpy(matrix_reduced_data + cov_reduced_offset, matrix_data + cov_offset , col_range*sizeof(double));

            // col-wise extraction of the p quadratures from the covariance matrix
            cov_reduced_offset = cov_reduced_offset + number_of_modes;
            cov_offset = cov_offset + total_number_of_modes;
            memcpy(matrix_reduced_data + cov_reduced_offset, matrix_data + cov_offset , col_range*sizeof(double));

        }


        // row-wise loop to extract the p quadrature columns from the covariance matrix
        for(size_t mode_row_idx=0; mode_row_idx<number_of_modes; mode_row_idx++) {

            // col-wise extraction of the q quadratures from the covariance matrix
            size_t cov_reduced_offset = (mode_row_idx+number_of_modes)*matrix_reduced.stride + mode_idx;
            size_t cov_offset = (indices_to_reduce[mode_row_idx]+total_number_of_modes)*matrix.stride + col_idx;
            memcpy(matrix_reduced_data + cov_reduced_offset, matrix_data + cov_offset , col_range*sizeof(double));

            // col-wise extraction of the p quadratures from the covariance matrix
            cov_reduced_offset = cov_reduced_offset + number_of_modes;
            cov_offset = cov_offset + total_number_of_modes;
            memcpy(matrix_reduced_data + cov_reduced_offset, matrix_data + cov_offset , col_range*sizeof(double));
        }

        // no displacement here
        /*
        // extract modes from the displacement
        if (m.size() > 0) {
            memcpy(m_reduced_data + mode_idx, m_data + col_idx, col_range*sizeof(Complex16)); // q quadratires
            memcpy(m_reduced_data + mode_idx + number_of_modes, m_data + col_idx + total_number_of_modes, col_range*sizeof(Complex16)); // p quadratures
        }
        */
        mode_idx = mode_idx + col_range;
        col_range = 1;

    }

    // returning the reduced matrix
    return matrix_real( matrix_reduced );
}



/**
@brief Call to calculate the inverse of matrix Q defined by Eq (3) of Ref. arXiv 2010.15595 and the determinant of Q.
Since the determinant can be calculated by LU factorization, which is also necessary to calculate the inverse, we
calculatet the inverse and the determiant in one shot.
@param state An instance of Gaussian state in the Fock representation. (If the Gaussian state is in quadrature representation, than it is transformed into Fock-space representation)
@param Qdet The calculated determinant of the matrix Q is stored into this value.
@return Returns with the Hamilton matrix A.
*/
matrix_real
ThresholdBosonSampling::calc_Qinv( matrix_real& matrix, double& Qdet ) {

    // calculate Q matrix from Eq (3) in arXiv 2010.15595v3)
    matrix_real& Q = matrix;
    for (size_t idx=0; idx<Q.rows; idx++) {
        Q[idx*Q.stride+idx] += 0.5 ;
    }

#ifdef DEBUG
    // for checking the matrix inversion
    matrix_real Qcopy = Q.copy();
#endif // DEBUG


    // calculate A matrix from Eq (4) in arXiv 2010.15595v3)
    matrix_real Qinv = Q; //just to reuse the memory of Q for the inverse

    // calculate the inverse of matrix Q
    int* ipiv = (int*)scalable_aligned_malloc( Q.stride*sizeof(int), CACHELINE);
    LAPACKE_dgetrf( LAPACK_ROW_MAJOR, Q.rows, Q.cols, Q.get_data(), Q.stride, ipiv );

    //  calculate the determinant of Q
    Qdet = 1.0;
    for (size_t idx=0; idx<Q.rows; idx++) {
        if (ipiv[idx] != idx+1) {
            Qdet = -Qdet * Q[idx*Q.stride + idx];
        }
        else {
            Qdet = Qdet * Q[idx*Q.stride + idx];
        }
    }

    int info = LAPACKE_dgetri( LAPACK_ROW_MAJOR, Q.rows, Q.get_data(), Q.stride, ipiv );
    scalable_aligned_free( ipiv );

    if ( info <0 ) {
        std::cout << "inversion was not successful. Exiting" << std::endl;
        exit(-1);
    }

#ifdef DEBUG
    // for checking the inversion
    size_t dim = Q.rows;
    matrix_real product = matrix_real(dim, dim);
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++){
            product[i * product.stride + j] = 0;
            for (size_t k = 0; k < dim; k++){
                product[i * product.stride + j] += Qinv[i * Q.stride + k] * Qcopy[k * Qcopy.stride + j];
            }
        }
        product[i * product.stride + i] -= 1.0;
    }

    for (size_t idx=0; idx<product.size(); idx++) {
        assert( abs(product[idx]) > 1e-9 );
    }
#endif // DEBUG

    return Qinv;
}




} // PIC
