#include <iostream>
#include "GaussianSimulationStrategy.h"
#include "PowerTraceHafnian.h"
#include "PowerTraceLoopHafnian.h"
#include "BruteForceHafnian.h"
#include "BruteForceLoopHafnian.h"
#include <math.h>
#include <tbb/tbb.h>
#include <chrono>
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
GaussianSimulationStrategy::GaussianSimulationStrategy() {


    cutoff = 0;
    max_photons = 0;


}


/**
@brief Constructor of the class.
@param covariance_matrix The covariance matrix describing the gaussian state
@param cutoff the Fock basis truncation.
@param max_photons specifies the maximum number of photons that can be counted in the output samples.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategy::GaussianSimulationStrategy( matrix &covariance_matrix, const size_t& cutoff, const size_t& max_photons ) {


    state = GaussianState_Cov( covariance_matrix, qudratures );
    setCutoff( cutoff );
    setMaxPhotons( max_photons );



    dim = covariance_matrix.rows;
    dim_over_2 = dim/2;


}


/**
@brief Constructor of the class.
@param covariance_matrix The covariance matrix describing the gaussian state
@param displacement The mean (displacement) of the Gaussian state
@param cutoff the Fock basis truncation.
@param max_photons specifies the maximum number of photons that can be counted in the output samples.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategy::GaussianSimulationStrategy( matrix &covariance_matrix, matrix& displacement, const size_t& cutoff, const size_t& max_photons ) {

    state = GaussianState_Cov(covariance_matrix, displacement, qudratures);

    setCutoff( cutoff );
    setMaxPhotons( max_photons );

    dim = covariance_matrix.rows;
    dim_over_2 = dim/2;

    // seed the random generator
    srand ( time ( NULL));


}


/**
@brief Destructor of the class
*/
GaussianSimulationStrategy::~GaussianSimulationStrategy() {
}

/**
@brief Call to update the memor address of the stored matrix iinterferometer_matrix
@param covariance_matrix The covariance matrix describing the gaussian state
*/
void
GaussianSimulationStrategy::Update_covariance_matrix( matrix &covariance_matrix ) {

    state.Update_covariance_matrix( covariance_matrix );




}

/**
@brief Call to set the cutoff of the Fock basis truncation
@param cutoff_in The cutoff of the Fock basis truncation
*/
void
GaussianSimulationStrategy::setCutoff( const size_t& cutoff_in ) {

    cutoff = cutoff_in;

}

/**
@brief Call to set the maximum number of photons that can be counted in the output samples.
@param max_photons_in The maximum number of photons that can be counted in the output samples.
*/
void
GaussianSimulationStrategy::setMaxPhotons( const size_t& max_photons_in ) {

    max_photons = max_photons_in;

}


/**
@brief Call to determine the resultant state after traversing through linear interferometer.
@return Returns with the resultant state after traversing through linear interferometer.
*/
std::vector<PicState_int64>
GaussianSimulationStrategy::simulate( int samples_number ) {

    // seed the random generator
    srand ( time ( NULL));


    // preallocate the memory for the output states
    std::vector<PicState_int64> samples;
    samples.reserve(samples_number);
    for (size_t idx=0; idx < samples_number; idx++) {
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
GaussianSimulationStrategy::getSample() {

    // convert the sampled Gaussian state into complex amplitude representation
    state.ConvertToComplexAmplitudes();

    PicState_int64 output_sample(0);
    output_sample.number_of_photons = 0;

    // probability of the sampled state
    double current_state_probability = 1.0;

    // The number of modes is equal to dim_over_2 (becose the covariance matrix conatains p,q quadratires)
    // for loop to sample 1,2,3,...dim_over_2 modes
    // These samplings depends from each other by the chain rule of probabilites (see Eq (13) in arXiv 2010.15595)
    for (size_t mode_idx=1; mode_idx<=dim_over_2; mode_idx++) {


        // container to store probabilities of getting different photon numbers on output of mode mode_idx
        // it contains maximally cutoff number of photons, the probability of getting higher photn number is stored in the cutoff+1-th element
        matrix_base<double> probabilities(1, cutoff+1);
        memset(probabilities.get_data(), 0, probabilities.size()*sizeof(double));

        // modes to be extracted to get reduced gaussian state
        PicState_int64 indices_2_extract(mode_idx);
        for (size_t idx=0; idx<mode_idx; idx++) {
            indices_2_extract[idx] = idx;
        }

        // get the reduced gaussian state describing the first mode_idx modes
        GaussianState_Cov reduced_state = state.getReducedGaussianState( indices_2_extract );

        // calculate the inverse of matrix Q defined by Eq (3) of Ref. arXiv 2010.15595 and the determinant of Q
        // since the determinant can be calculated by LU factorization, which is also necessary to calculate the inverse, we
        // calculatet the inverse and the determiant in one shot.
        double Qdet(0.0);
        matrix&& Qinv = calc_Qinv( reduced_state, Qdet );

        // calculate the Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
        matrix&& A = calc_HamiltonMatrix( Qinv );

        // get the displacement vector
        matrix m = reduced_state.get_m();




        // get the probabilities for different photon counts on the output mode mode_idx
        tbb::parallel_for( (size_t)0, cutoff, (size_t)1, [&](size_t photon_num) {
        //for (size_t photon_num=0; photon_num<cutoff; photon_num++) {

            // create array for the new output state
            PicState_int64 current_output(output_sample.size()+1, 0);
            memcpy(current_output.get_data(), output_sample.get_data(), output_sample.size()*sizeof(int64_t));


            // set the number of photons in the last mode do be sampled
            current_output[mode_idx-1] = photon_num;

            // calculate the probability associated with observing current_output
            double prob = calc_probability(Qinv, Qdet, A, m, current_output);

            // sometimes the probability is negative which is coming from a negative hafnian.
            probabilities[photon_num] = prob > 0 ? prob : 0;

        //}
        });

        matrix_base<double> probabilities_renormalized(1,probabilities.size());

        // renormalize the calculated probabilities with the probability of the previously chosen state
        double prob_sum = 0.0;
        for (size_t idx=0; idx<probabilities_renormalized.size()-1; idx++) {
            probabilities_renormalized[idx] = probabilities[idx]/current_state_probability;
            prob_sum = prob_sum + probabilities_renormalized[idx];
        }
        probabilities_renormalized[probabilities.size()-1] = 1.0-prob_sum;

/*
if ( prob_sum > 1 ) {
probabilities_renormalized.print_matrix();

FILE *fp = fopen("bad_matrix", "wb");
fwrite( &A.rows, sizeof(size_t), 1, fp);
fwrite( &A.cols, sizeof(size_t), 1, fp);
fwrite( A.get_data(), sizeof(Complex16), A.size(), fp);
fwrite( &output_sample.cols, sizeof(size_t), 1, fp);
fwrite( output_sample.get_data(), sizeof(int64_t), output_sample.size(), fp);
fclose(fp);
exit(-1);

}
*/
        // sample from porbabilities
        size_t chosen_index = sample_from_probabilities( probabilities_renormalized );
        if (chosen_index == cutoff) {
            // when the chosen index corresponds to the cutoff the number of photons cannot be determined, so we return from the sampling
            return PicState_int64(0);
        }

        // The sampled current state:
        PicState_int64 current_output(output_sample.size()+1, 0);
        memcpy(current_output.get_data(), output_sample.get_data(), output_sample.size()*sizeof(int64_t));
        current_output[mode_idx-1] = chosen_index;
        current_output.number_of_photons = output_sample.number_of_photons + chosen_index;

        if (current_output.number_of_photons > max_photons) {
            return PicState_int64(0);
        }


        // update the probability of the current sampled state
        current_state_probability = probabilities[chosen_index];

        output_sample = current_output;

    }

    return output_sample;

}




/**
@brief Call to calculate the inverse of matrix Q defined by Eq (3) of Ref. arXiv 2010.15595
@param state An instance of Gaussian state in the Fock representation. (If the Gaussian state is in quadrature representation, than it is transformed into Fock-space representation)
@return Returns with the Hamilton matrix A.
*/
matrix
GaussianSimulationStrategy::calc_Qinv( GaussianState_Cov& state ) {


    if ( state.get_representation() != complex_amplitudes ) {
        state.ConvertToComplexAmplitudes();
    }


    // calculate Q matrix from Eq (3) in arXiv 2010.15595v3)
    matrix Q = state.get_covariance_matrix();

    for (size_t idx=0; idx<Q.rows; idx++) {
        Q[idx*Q.stride+idx].real( Q[idx*Q.stride+idx].real() + 0.5 );
    }


#ifdef DEBUG
    // for checking the matrix inversion
    matrix Qcopy = Q.copy();
#endif // DEBUG


    // calculate A matrix from Eq (4) in arXiv 2010.15595v3)
    matrix Qinv = Q; //just to reuse the memory of Q for the inverse


    // calculate the inverse of matrix Q
    int* ipiv = (int*)scalable_aligned_malloc( Q.stride*sizeof(int), CACHELINE);
    LAPACKE_zgetrf( LAPACK_ROW_MAJOR, Q.rows, Q.cols, Q.get_data(), Q.stride, ipiv );
    int info = LAPACKE_zgetri( LAPACK_ROW_MAJOR, Q.rows, Q.get_data(), Q.stride, ipiv );
    scalable_aligned_free( ipiv );

    if ( info <0 ) {
        std::cout << "inversion was not successfull. Exiting" << std::endl;
        exit(-1);
    }

#ifdef DEBUG
    // for checking the inversion
    matrix C = dot(Qinv, Qcopy);
    for (size_t idx=0; idx<C.rows; idx++) {
        C[idx.C.stride+idx].real( C[idx.C.stride+idx].real() - 1.0);
    }
    double diff=0.0;
    for (size_t idx=0; idx<C.size(); idx++) {
        assert( abs(C[idx]) > 1e-9);
    }
#endif // DEBUG


    return Qinv;

}


/**
@brief Call to calculate the inverse of matrix Q defined by Eq (3) of Ref. arXiv 2010.15595 and the determinant of Q.
Since the determinant can be calculated by LU factorization, which is also necessary to calculate the inverse, we
calculatet the inverse and the determiant in one shot.
@param state An instance of Gaussian state in the Fock representation. (If the Gaussian state is in quadrature representation, than it is transformed into Fock-space representation)
@param Qdet The calculated determinant of the matrix Q is stored into this value.
@return Returns with the Hamilton matrix A.
*/
matrix
GaussianSimulationStrategy::calc_Qinv( GaussianState_Cov& state, double& Qdet ) {



    if ( state.get_representation() != complex_amplitudes ) {
        state.ConvertToComplexAmplitudes();
    }

    // calculate Q matrix from Eq (3) in arXiv 2010.15595v3)
    matrix Q = state.get_covariance_matrix();
    for (size_t idx=0; idx<Q.rows; idx++) {
        Q[idx*Q.stride+idx].real( Q[idx*Q.stride+idx].real() + 0.5 );
    }

#ifdef DEBUG
    // for checking the matrix inversion
    matrix Qcopy = Q.copy();
#endif // DEBUG


    // calculate A matrix from Eq (4) in arXiv 2010.15595v3)
    matrix Qinv = Q; //just to reuse the memory of Q for the inverse

    // calculate the inverse of matrix Q
    int* ipiv = (int*)scalable_aligned_malloc( Q.stride*sizeof(int), CACHELINE);
    LAPACKE_zgetrf( LAPACK_ROW_MAJOR, Q.rows, Q.cols, Q.get_data(), Q.stride, ipiv );

    //  calculate the determinant of Q
    Complex16 Qdet_cmplx(1.0,0.0);
    for (size_t idx=0; idx<Q.rows; idx++) {
        if (ipiv[idx] != idx+1) {
            Qdet_cmplx = -Qdet_cmplx * Q[idx*Q.stride + idx];
        }
        else {
            Qdet_cmplx = Qdet_cmplx * Q[idx*Q.stride + idx];
        }

    }
    Qdet = Qdet_cmplx.real(); // the determinant of a symmetric matrix is real

    int info = LAPACKE_zgetri( LAPACK_ROW_MAJOR, Q.rows, Q.get_data(), Q.stride, ipiv );
    scalable_aligned_free( ipiv );

    if ( info <0 ) {
        std::cout << "inversion was not successful. Exiting" << std::endl;
        exit(-1);
    }

#ifdef DEBUG
    // for checking the inversion
    matrix C = dot(Qinv, Qcopy);
    for (size_t idx=0; idx<C.rows; idx++) {
        C[idx*C.stride+idx].real( C[idx*C.stride+idx].real() - 1.0);
    }

    for (size_t idx=0; idx<C.size(); idx++) {
        assert( abs(C[idx]) > 1e-9);
    }
#endif // DEBUG

    return Qinv;


}



/**
@brief Call to calculate the Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
@param Qinv An instace of matrix class conatining the inverse of matrix Q calculated by method get_Qinv.
@return Returns with the Hamilton matrix A.
*/
matrix
GaussianSimulationStrategy::calc_HamiltonMatrix( matrix& Qinv ) {

    //calculate A = X (1-Qinv)    X=(0,1;1,0)

    // calculate -XQinv
    // multiply by -1 the elements of Qinv and store the result in the corresponding rows of A
    matrix A(Qinv.rows, Qinv.cols);
    __m256d neg = _mm256_setr_pd(-1.0, -1.0, -1.0, -1.0);
    Complex16* Qinv_data_d = Qinv.get_data();
    Complex16* A_data_d    = A.get_data();
    size_t number_of_modes = Qinv.rows/2;
    for (size_t row_idx = 0; row_idx<number_of_modes ; row_idx++) {

        size_t row_offset1 = row_idx*Qinv.stride;
        size_t row_offset2 = (row_idx+number_of_modes)*Qinv.stride;

         // rows 1:N form (1-Qinv) to rows N+1:2N in A --- effect of X
        for (size_t col_idx = 0; col_idx<Qinv.cols; col_idx=col_idx+2) {

            __m256d Qinv_vec = _mm256_loadu_pd((double*)(Qinv_data_d + row_offset1 + col_idx));
            Qinv_vec = _mm256_mul_pd(Qinv_vec, neg);
            _mm256_storeu_pd((double*)(A_data_d + row_offset2 + col_idx), Qinv_vec);

        }

        // rows N+1:2N form (1-Qinv) to rows 1:N in A --- effect of X
        for (size_t col_idx = 0; col_idx<Qinv.cols; col_idx=col_idx+2) {

            __m256d Qinv_vec = _mm256_loadu_pd((double*)(Qinv_data_d + row_offset2 + col_idx));
            Qinv_vec = _mm256_mul_pd(Qinv_vec, neg);
            _mm256_storeu_pd((double*)(A_data_d + row_offset1 + col_idx), Qinv_vec);

        }



    }

    // calculate X-XQinv
    // add X to the matrix eleemnts of -XQinv
    for (size_t row_idx = 0; row_idx<number_of_modes; row_idx++) {

        A[row_idx*A.stride+row_idx+number_of_modes].real(A[row_idx*A.stride+row_idx+number_of_modes].real() + 1);
        A[(row_idx+number_of_modes)*A.stride+row_idx].real(A[(row_idx+number_of_modes)*A.stride+row_idx].real() + 1);

    }


    return A;

}


/**
@brief Call to calculate the probability associated with observing output state given by current_output
@param Qinv An instace of matrix class conatining the inverse of matrix Q calculated by method get_Qinv.
@param Qdet The determinant of matrix Q.
@param A Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
@param m The displacement \f$ \alpha \f$ defined by Eq (8) of Ref. arXiv 2010.15595
@param current_output The fock representation of the current output for which the probability is calculated
@return Returns with the calculated probability
*/
double
GaussianSimulationStrategy::calc_probability( matrix& Qinv, const double& Qdet, matrix& A, matrix& m, PicState_int64& current_output ) {

    // calculate the normalization factor defined by Eq. (10) in arXiv 2010.15595v3
    double Normalization = 1.0/sqrt(Qdet);

#ifdef DEBUG
    if (Qdet<0) {
        std::cout << "Determinant of matrix Q is negative" << std::endl;
        exit(-1);
    }
#endif

    if (m.size()>0) {

        // calculate Q_inv * conj(alpha)
        matrix tmp(m.size(),1);
        for (size_t row_idx=0; row_idx<m.size(); row_idx++) {
            tmp[row_idx] = Complex16(0.0,0.0);
            size_t row_offset = row_idx*Qinv.stride;

            for (size_t col_idx=0; col_idx<m.size(); col_idx++) {
                tmp[row_idx] = tmp[row_idx] + mult_a_bconj( Qinv[row_offset+col_idx], m[col_idx] );
            }
        }


        // calculate alpha * Qinv * conj(alpha)
        Complex16 inner_prod(0.0,0.0);
        for (size_t idx=0; idx<m.size(); idx++) {
            inner_prod = inner_prod + m[idx]*tmp[idx];
        }

        Normalization = exp(-0.5*inner_prod.real())*Normalization;


    }


    // divide Normalization factor by s_1!...s_m! in Eq (10) of arXiv 2010.15595v3
    for (size_t idx=0;idx<current_output.size(); idx++) {
        Normalization = Normalization/factorial(current_output[idx]);
    }

    // create Matrix A_S according to the main text below Eq (5) of arXiv 2010.15595v3
    matrix&& A_S = create_A_S( A, current_output );

    // calculate the hafnian of A_S
    Complex16 hafnian;
    if (m.size()==0) {
        // gaussian state without displacement
        if (A_S.rows <= 10) {
            BruteForceHafnian hafnian_calculator = BruteForceHafnian(A_S);
            hafnian = hafnian_calculator.calculate();
        }
        else {
            PowerTraceHafnian hafnian_calculator = PowerTraceHafnian(A_S);
            hafnian = hafnian_calculator.calculate();
        }
    }
    else {
        // gaussian state with displacement

        // calculate gamma according to Eq (9) of arXiv 2010.15595v3 and set them into the diagonal of A_S
        diag_correction_of_A_S( A_S, Qinv, m, current_output );

        if (A_S.rows <= 2) {
            BruteForceLoopHafnian hafnian_calculator = BruteForceLoopHafnian(A_S);
            hafnian = hafnian_calculator.calculate();
        }
        else {
            PowerTraceLoopHafnian hafnian_calculator = PowerTraceLoopHafnian(A_S);
            hafnian = hafnian_calculator.calculate();
        }



    }


    // calculate the probability associated with the current output
    double prob = Normalization*hafnian.real();


    return prob;

}

/**
@brief Call to add correction coming from the displacement to the diagonal elements of A_S (see Eq. (11) in arXiv 2010.15595)
@param A_S Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
(The output is returned via this variable)
@param Qinv An instace of matrix class conatining the inverse of matrix Q calculated by method get_Qinv.
@param m The displacement \f$ \alpha \f$ defined by Eq (8) of Ref. arXiv 2010.15595
@param current_output The Fock representation of the current output for which the probability is calculated
*/
void
GaussianSimulationStrategy::diag_correction_of_A_S( matrix& A_S, matrix& Qinv, matrix& m, PicState_int64& current_output ) {

    matrix gamma(Qinv.rows, 1);
    for (size_t row_idx=0; row_idx<Qinv.rows; row_idx++) {

        size_t row_offset = row_idx*Qinv.stride;
        gamma[row_idx] = Complex16(0.0,0.0);

        for (size_t col_idx=0; col_idx<Qinv.rows; col_idx++) {
            gamma[row_idx] = gamma[row_idx] + mult_a_bconj( Qinv[row_offset + col_idx], m[col_idx] );
        }
    }
    // store gamma values into matrix A_S
    size_t num_of_modes = current_output.size();
    size_t num_of_repeated_modes = A_S.rows/2;
    size_t row_idx = 0;
    for (size_t idx=0; idx<num_of_modes; idx++) {
        for (size_t row_repeat=0; row_repeat<current_output[idx]; row_repeat++) {

            A_S[row_idx*A_S.stride + row_idx] = gamma[idx];
            A_S[(row_idx+num_of_repeated_modes)*A_S.stride + row_idx + num_of_repeated_modes] = gamma[idx+num_of_modes];


            row_idx++;
        }
    }


    return;
}



/**
@brief Call to create matrix A_S according to the main text below Eq (5) of arXiv 2010.15595v3
@param A Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
@param current_output The fock representation of the current output for which the probability is calculated
@return Returns with the A_S matrix
*/
matrix
GaussianSimulationStrategy::create_A_S( matrix& A, PicState_int64& current_output ) {

    size_t dim_A_S = sum(current_output);
    size_t dim_A = current_output.size();

    matrix A_S(2*dim_A_S, 2*dim_A_S);
    memset(A_S.get_data(), 0, A_S.size()*sizeof(Complex16));

    size_t row_idx = 0;
    for (size_t idx=0; idx<current_output.size(); idx++) {
        for (size_t row_repeat=0; row_repeat<current_output[idx]; row_repeat++) {

            size_t row_offset = row_idx*A_S.stride;
            size_t row_offset_A = idx*A.stride;
            size_t col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<current_output.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<current_output[jdx]; col_repeat++) {
                    if ( (row_idx == col_idx) || (idx != jdx) ) {
                        A_S[row_offset + col_idx] = A[row_offset_A + jdx];
                    }
                    col_idx++;
                }
            }

            col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<current_output.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<current_output[jdx]; col_repeat++) {
                    A_S[row_offset + col_idx + dim_A_S] = A[row_offset_A + jdx + dim_A];
                    col_idx++;
                }

            }

            row_offset = (row_idx+dim_A_S)*A_S.stride;
            row_offset_A = (idx+dim_A)*A.stride;
            col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<current_output.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<current_output[jdx]; col_repeat++) {
                    A_S[row_offset + col_idx] = A[row_offset_A + jdx];
                    col_idx++;
                }
            }

            col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<current_output.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<current_output[jdx]; col_repeat++) {
                    if ( (row_idx == col_idx) || (idx != jdx) ) {
                        A_S[row_offset + col_idx + dim_A_S] = A[row_offset_A + jdx + dim_A];
                    }
                    col_idx++;
                }

            }


            row_idx++;
        }


    }

    return A_S;

}


/**
@brief Call to sample from a probability array.
@param probabilities Array of probabilities from which the sampling should be taken
@return Returns with the index of the chosen probability value
*/
size_t
GaussianSimulationStrategy::sample_from_probabilities( matrix_base<double>& probabilities ) {

    // create a random double
    double rand_num = (double)rand()/RAND_MAX;

    // determine the random index according to the distribution described by probabilities
    size_t random_index=0;
    double prob_sum = 0.0;
    for (size_t idx=0; idx<probabilities.size(); idx++) {
        prob_sum = prob_sum + probabilities[idx];
        if ( prob_sum >= rand_num) {
            random_index = idx;
            break;
        }
    }

    return random_index;
}



} // PIC
