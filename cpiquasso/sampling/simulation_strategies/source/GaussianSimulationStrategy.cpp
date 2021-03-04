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


    cutoff = 0;
    max_photons = 0;


    // seeding the random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);

}


/**
@brief Constructor of the class.
@param covariance_matrix The covariance matrix describing the gaussian state
@param cutoff the Fock basis truncation.
@param max_photons specifies the maximum number of photons that can be counted in the output samples.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategy::GaussianSimulationStrategy( matrix &covariance_matrix, const size_t& cutoff, const size_t& max_photons ) {


    state = GaussianState_Cov( covariance_matrix );
    setCutoff( cutoff );
    setMaxPhotons( max_photons );

    dim = covariance_matrix.rows;
    dim_over_2 = dim/2;

    hbar = 2.0;

    // seeding the random number generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);
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

    state = GaussianState_Cov(covariance_matrix, displacement);

    setCutoff( cutoff );
    setMaxPhotons( max_photons );

    hbar = 2.0;

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

    // preallocate the memory for the output states
    std::vector<PicState_int64> samples;
    samples.reserve(samples_number);
    for (size_t idx=0; idx < samples_number; idx++) {
        samples.push_back(getSample());
    }

    return samples;
}



/**
@brief Call to get one sample from the gaussian state
@return Returns with the a sample from a gaussian state
*/
PicState_int64
GaussianSimulationStrategy::getSample() {

//std::cout << dim_over_2 << std::endl;

    PicState_int64 sample(dim_over_2,0);

    // The number of modes is equal to dim_over_2 (becose the covariance matrix conatains p,q quadratires)
    // for loop to sample 1,2,3,...dim_over_2 modes
    // These samplings depends from each other by the chain rule of probabilites (see Eq (13) in arXiv 2010.15595)
    for (size_t mode_idx=1; mode_idx<=dim_over_2; mode_idx++) {


        // container to store probabilities of getting different photon numbers on output of mode mode_idx
        // it contains maximally cutoff number of photons, the probability of getting higher photn number is stored in the cutoff+1-th element
        matrix_base<double> probabilities(1, cutoff+1);

        // modes to be extracted to get reduced gaussian state
        PicState_int64 indices_2_extract(mode_idx);
        for (size_t idx=0; idx<mode_idx; idx++) {
            indices_2_extract[idx] = idx;
        }
        indices_2_extract.print_matrix();

        GaussianState_Cov reduced_state = state.getReducedGaussianState( indices_2_extract );




    }

/*
    prev_prob = 1.0
    nmodes = N
    for k in range(nmodes):
        probs1 = np.zeros([cutoff + 1], dtype=np.float64)
        kk = np.arange(k + 1)
        mu_red, V_red = reduced_gaussian(local_mu, cov, kk)

        if approx:
            Q = Qmat(V_red, hbar=hbar)
            A = Amat(Q, hbar=hbar, cov_is_qmat=True)

        for i in range(cutoff):
            indices = result + [i]
            ind2 = indices + indices
            if approx:
                factpref = np.prod(fac(indices))
                mat = reduction(A, ind2)
                probs1[i] = (
                    hafnian(np.abs(mat.real), approx=True, num_samples=approx_samples) / factpref
                )
            else:
                probs1[i] = density_matrix_element(
                    mu_red, V_red, indices, indices, include_prefactor=True, hbar=hbar
                ).real

        if approx:
            probs1 = probs1 / np.sqrt(np.linalg.det(Q).real)

        probs2 = probs1 / prev_prob
        probs3 = np.maximum(
            probs2, np.zeros_like(probs2)
        )  # pylint: disable=assignment-from-no-return
        ssum = np.sum(probs3)
        if ssum < 1.0:
            probs3[-1] = 1.0 - ssum

        # The following normalization of probabilities is needed when approx=True
        if approx:
            if ssum > 1.0:
                probs3 = probs3 / ssum

        result.append(np.random.choice(a=range(len(probs3)), p=probs3))
        if result[-1] == cutoff:
            return -1
        if np.sum(result) > max_photons:
            return -1
        prev_prob = probs1[result[-1]]


*/
    return sample;

}







} // PIC
