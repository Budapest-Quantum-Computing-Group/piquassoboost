#ifndef GaussianSimulationStrategy_H
#define GaussianSimulationStrategy_H

#include "matrix.h"
#include "PicVector.hpp"
#include "PicState.h"
#include "PicStateHash.h"
#include "GaussianState_Cov.h"
#include <unordered_map>
#include <random>


namespace pic {




/**
@brief Call to calculate sum of integers stored in a container
@param vec a container if integers
@return Returns with the sum of the elements of the container
*/
static int64_t sum( PicState_int64 &vec);

/**
@brief Class representing a generalized Cliffords simulation strategy
*/
class GaussianSimulationStrategy {

protected:

    /// object describing the Gaussian state
    GaussianState_Cov state;

    /// cutoff of the Fock basis truncation.
    size_t cutoff;
    /// the maximum number of photons that can be counted in the output samples.
    size_t max_photons;

    size_t dim;
    size_t dim_over_2;

    double hbar;

    /// The number of photons
    int64_t number_of_input_photons;
    /// The individual probability layers of the possible output states
    std::unordered_map<PicState_int64, matrix_base<double>, PicStateHash> pmfs;
    /// The possible output sates organized by keys of the inducing input states
    std::unordered_map<PicState_int64, PicStates, PicStateHash> possible_output_states;
    /// The input state entering the interferometer
    PicState_int64 input_state; // can be aligned?
    /// Map of all the possible substates of input_state (including 0 and the input state), labeled with number of particles in this state
    /// the number of particles is given by the ordinal number of the vector element, i.e. labeled_states[n] gives the substates with n occupied particles.
    std::vector<concurrent_PicStates> labeled_states;
    /// The vector of indices corresponding to values greater than 0 in the input state
    PicVector<int64_t> input_state_inidices;
    /// random number generator
    std::default_random_engine generator;


public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategy();

/**
@brief Constructor of the class. (The displacement is set to zero by this constructor)
@param covariance_matrix_in The covariance matrix describing the gaussian state
@param cutoff the Fock basis truncation.
@param max_photons specifies the maximum number of photons that can be counted in the output samples.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategy( matrix &covariance_matrix_in, const size_t& cutoff, const size_t& max_photons );


/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix describing the gaussian state
@param displacement The mean (displacement) of the Gaussian state
@param cutoff the Fock basis truncation.
@param max_photons specifies the maximum number of photons that can be counted in the output samples.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategy( matrix &covariance_matrix_in, matrix& displacement_in, const size_t& cutoff, const size_t& max_photons );

/**
@brief Destructor of the class
*/
~GaussianSimulationStrategy();

/**
@brief Call to update the memory address of the matrix stored in covariance_matrix
@param covariance_matrix_in The covariance matrix describing the gaussian state
*/
void Update_covariance_matrix( matrix &covariance_matrix_in );


/**
@brief Call to set the cutoff of the Fock basis truncation
@param cutoff_in The cutoff of the Fock basis truncation
*/
void setCutoff( const size_t& cutoff_in );

/**
@brief Call to set the maximum number of photons that can be counted in the output samples.
@param max_photons_in The maximum number of photons that can be counted in the output samples.
*/
void setMaxPhotons( const size_t& max_photons_in );

/**
@brief Call to get samples from the gaussian state
@param samples_number The number of shots for which the output should be determined
@return Returns with the samples of the gaussian state
*/
std::vector<PicState_int64> simulate( int samples_number );

protected:





/**
@brief Call to get one sample from the gaussian state
@return Returns with the a sample from a gaussian state
*/
PicState_int64 getSample();


/**
@brief Call to calculate the inverse of matrix Q defined by Eq (3) of Ref. arXiv 2010.15595
@param state An instance of Gaussian state in the Fock representation. (If the Gaussian state is in quadrature representation, than it is transformed into Fock-space representation)
@return Returns with the Hamilton matrix A.
*/
matrix calc_Qinv( GaussianState_Cov& state );

/**
@brief Call to calculate the inverse of matrix Q defined by Eq (3) of Ref. arXiv 2010.15595 and the determinant of Q.
Since the determinant can be calculated by LU factorization, which is also necessary to calculate the inverse, we
calculatet the inverse and the determiant in one shot.
@param state An instance of Gaussian state in the Fock representation. (If the Gaussian state is in quadrature representation, than it is transformed into Fock-space representation)
@param Qdet The calculated determinant of the matrix Q is stored into this value.
@return Returns with the Hamilton matrix A.
*/
matrix calc_Qinv( GaussianState_Cov& state, double& Qdet );


/**
@brief Call to calculate the Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
@param Qinv An instace of matrix class conatining the inverse of matrix Q calculated by method get_Qinv.
@return Returns with the Hamilton matrix A.
*/
matrix calc_HamiltonMatrix( matrix& Qinv );


/**
@brief Call to calculate the probability associated with observing output state given by current_output
@param Qinv An instace of matrix class conatining the inverse of matrix Q calculated by method get_Qinv.
@param Qdet The determinant of matrix Q.
@param A Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
@param m The displacement \f$ \alpha \f$ defined by Eq (8) of Ref. arXiv 2010.15595
@param current_output The fock representation of the current output for which the probability is calculated
@return Returns with the calculated probability
*/
double calc_probability( matrix& Qinv, const double& Qdet, matrix& A, matrix& m, PicState_int64& current_output );


/**
@brief Call to add correction coming from the displacement to the diagonal elements of A_S (see Eq. (11) in arXiv 2010.15595)
@param A_S Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
(The output is returned via this variable)
@param Qinv An instace of matrix class conatining the inverse of matrix Q calculated by method get_Qinv.
@param m The displacement \f$ \alpha \f$ defined by Eq (8) of Ref. arXiv 2010.15595
@param current_output The Fock representation of the current output for which the probability is calculated
*/
void diag_correction_of_A_S( matrix& A_S, matrix& Qinv, matrix& m, PicState_int64& current_output );

/**
@brief Call to create matrix A_S according to the main text below Eq (5) of arXiv 2010.15595v3
@param A Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
@param current_output The fock representation of the current output for which the probability is calculated
@return Returns with the A_S matrix
*/
matrix create_A_S( matrix& A, PicState_int64& current_output );


/**
@brief Call to sample from a probability array.
@param probabilities Array of probabilities from which the sampling should be taken
@return Returns with the index of the chosen probability value
*/
size_t sample_from_probabilities( matrix_base<double>& probabilities );


}; //GaussianSimulationStrategy





} // PIC

#endif
