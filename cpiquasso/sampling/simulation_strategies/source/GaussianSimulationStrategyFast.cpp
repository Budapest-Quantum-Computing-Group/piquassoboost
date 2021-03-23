#include <iostream>
#include "GaussianSimulationStrategyFast.h"
#include "PowerTraceHafnianRecursive.h"
#include "PowerTraceLoopHafnianRecursive.h"
#include "PowerTraceLoopHafnian.h"
#include "BruteForceHafnian.h"
#include "BruteForceLoopHafnian.h"
#include <math.h>
#include <tbb/tbb.h>
#include <chrono>
#include "dot.h"
//#include "lapacke.h"

extern "C" {

#define LAPACK_ROW_MAJOR               101

/// Definition of the LAPACKE_zgetri function from LAPACKE to calculate the LU decomposition of a matrix
int LAPACKE_zgetrf( int matrix_layout, int n, int m, pic::Complex16* a, int lda, int* ipiv );

/// Definition of the LAPACKE_zgetri function from LAPACKE to calculate the inverse of a matirx
int LAPACKE_zgetri( int matrix_layout, int n, pic::Complex16* a, int lda, const int* ipiv );

}

tbb::spin_mutex mymutex;

namespace pic {

/**
@brief Transforms the covariance matrix in the basis \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ...  a_n^* \f$  into the basis \f$ a_1, a_1^*,a_2, a_2^*, ... a_n, a_n^* \f$ suitable for
the PowerTraceHafnianRecursive algorithm.
@param mtx A covariance matrix in the basis \f$ a_1, a_2, ... a_n,, a_1^*, a_2^*, ...  a_n^* \f$.
@return Returns with the covariance matrix in the basis \f$ a_1, a_1^*,a_2, a_2^*, ... a_n, a_n^* \f$.
*/
matrix
getPermutedMatrix( matrix& mtx) {


    matrix res(mtx.rows, mtx.cols);


    size_t num_of_modes = mtx.rows/2;

    for (size_t row_idx=0; row_idx<num_of_modes; row_idx++ ) {

        size_t row_offset_q_orig = row_idx*mtx.stride;
        size_t row_offset_p_orig = (row_idx+num_of_modes)*mtx.stride;

        size_t row_offset_q_permuted = 2*row_idx*res.stride;
        size_t row_offset_p_permuted = (2*row_idx+1)*res.stride;

        for (size_t col_idx=0; col_idx<num_of_modes; col_idx++ ) {

            res[row_offset_q_permuted + col_idx*2] = mtx[row_offset_q_orig + col_idx];
            res[row_offset_q_permuted + col_idx*2 + 1] = mtx[row_offset_q_orig + num_of_modes + col_idx];

            res[row_offset_p_permuted + col_idx*2] = mtx[row_offset_p_orig + col_idx];
            res[row_offset_p_permuted + col_idx*2 + 1] = mtx[row_offset_p_orig + num_of_modes + col_idx];

        }

    }

    //res.print_matrix();

    return res;

}


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
GaussianSimulationStrategyFast::GaussianSimulationStrategyFast() : GaussianSimulationStrategy() {


}

/**
@brief Constructor of the class. (The displacement is set to zero by this constructor)
@param covariance_matrix_in The covariance matrix describing the gaussian state
@param cutoff the Fock basis truncation.
@param max_photons specifies the maximum number of photons that can be counted in the output samples.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategyFast::GaussianSimulationStrategyFast( matrix &covariance_matrix_in, const size_t& cutoff, const size_t& max_photons ) :
    GaussianSimulationStrategy(covariance_matrix_in, cutoff, max_photons) {


}


/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix describing the gaussian state
@param displacement The mean (displacement) of the Gaussian state
@param cutoff the Fock basis truncation.
@param max_photons specifies the maximum number of photons that can be counted in the output samples.
@return Returns with the instance of the class.
*/
GaussianSimulationStrategyFast::GaussianSimulationStrategyFast( matrix &covariance_matrix_in, matrix& displacement_in, const size_t& cutoff, const size_t& max_photons ) :
    GaussianSimulationStrategy(covariance_matrix_in, displacement_in, cutoff, max_photons) {

}


/**
@brief Destructor of the class
*/
GaussianSimulationStrategyFast::~GaussianSimulationStrategyFast() {
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
GaussianSimulationStrategyFast::calc_probability( matrix& Qinv, const double& Qdet, matrix& A, matrix& m, PicState_int64& current_output ) {

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


    // determine number of modes with nonzero occupancy
    size_t num_of_modes = 0;
    for (size_t idx=0; idx<current_output.size(); idx++) {
        if (current_output[idx]>0) {
            num_of_modes++;
        }
    }

    PicState_int64 selected_modes(num_of_modes,0);
    PicState_int64 occupancy(num_of_modes);
    size_t occupancy_idx = 0;
    for (size_t idx=0; idx<current_output.size(); idx++) {
        if (current_output[idx]>0) {
            occupancy[occupancy_idx] = current_output[idx];
            selected_modes[occupancy_idx] = idx;
            occupancy_idx++;
        }
    }


    // calculate the hafnian of A_S
    Complex16 hafnian;
    if (m.size()==0) {

/*
        // create Matrix A_S according to the main text below Eq (5) of arXiv 2010.15595v3
        matrix&& A_S = create_A_S( A, current_output );
        PowerTraceHafnian hafnian_calculator = PowerTraceHafnian(A_S);
        hafnian = hafnian_calculator.calculate();
*/

        // get rid of modes with zero occupancy
        matrix&& A_selected_modes = ExtractModes( A, selected_modes);

        PowerTraceHafnianRecursive hafnian_calculator_recursive = PowerTraceHafnianRecursive(A_selected_modes, occupancy);
        hafnian = hafnian_calculator_recursive.calculate();
        //Complex16 hafnian2 = hafnian_calculator_recursive.calculate();

/*
{
    tbb::spin_mutex::scoped_lock my_lock{mymutex};
if ( std::abs(hafnian - hafnian2)/std::abs(hafnian) > 0.01) {
std::cout << "hafnaian diff: " << hafnian - hafnian2 << " hafnian: " << hafnian << " hafnian2: " << hafnian2 <<std::endl;
//occupancy.print_matrix();
//selected_modes.print_matrix();
//current_output.print_matrix();

//A.print_matrix();
//A_selected_modes.print_matrix();
//A_S.print_matrix();
//A_S_recursive.print_matrix();

}
else {
    std::cout << " hafnian2: " << hafnian2 <<std::endl;
}
}
*/


    }
    else {
        // gaussian state with displacement
/*
        // create Matrix A_S according to the main text below Eq (5) of arXiv 2010.15595v3
        matrix&& A_S = create_A_S( A, current_output );

        // calculate gamma according to Eq (9) of arXiv 2010.15595v3 and set them into the diagonal of A_S
        diag_correction_of_A_S( A_S, Qinv, m, current_output );


        PowerTraceLoopHafnian hafnian_calculator = PowerTraceLoopHafnian(A_S);
        hafnian = hafnian_calculator.calculate();
*/



        // get rid of modes with zero occupancy
        matrix&& A_selected_modes = ExtractModes( A, selected_modes);

        // calculate gamma according to Eq (9) of arXiv 2010.15595v3 in ordering a_1, a_1^*, a_2, a_2^* ....
        matrix&& gamma = CalcGamma( Qinv, m, selected_modes );

        // add the diagonal correction to the Hamilton's matrix
        diag_correction_of_A( A_selected_modes, gamma );


        PowerTraceLoopHafnianRecursive hafnian_calculator_recursive = PowerTraceLoopHafnianRecursive(A_selected_modes, occupancy);
        hafnian = hafnian_calculator_recursive.calculate();
        //Complex16 hafnian2 = hafnian_calculator_recursive.calculate();

/*
{
    tbb::spin_mutex::scoped_lock my_lock{mymutex};
if ( std::abs(hafnian - hafnian2)/std::abs(hafnian) > 0.01) {
std::cout << "hafnaian diff: " << hafnian - hafnian2 << " hafnian: " << hafnian << " hafnian2: " << hafnian2 <<std::endl;
occupancy.print_matrix();
selected_modes.print_matrix();
//current_output.print_matrix();

//A.print_matrix();
A_selected_modes.print_matrix();
A_S.print_matrix();
//A_S_recursive.print_matrix();

}
else {
    std::cout << " hafnian2: " << hafnian2 <<std::endl;
}
}
*/


    }


    // calculate the probability associated with the current output
    double prob = Normalization*hafnian.real();


    return prob;

}



/**
@brief Call to extract selected modes from the covariance matrix in \$f a_1, a_1^*, a_2, a_2^* ... \f$ ordering.
@param A Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
@param selected_modes An array of labels containing the selected modes
@return Returns with the matrix containing the selected modes
*/
matrix
GaussianSimulationStrategyFast::ExtractModes( matrix& A, PicState_int64& selected_modes ) {

    size_t dim_ret = selected_modes.size();
    size_t dim_A = A.rows/2;

    matrix ret(2*dim_ret, 2*dim_ret);
    for (size_t row_idx=0; row_idx<dim_ret; row_idx++) {

        size_t row_offset = 2*row_idx*ret.stride;
        size_t row_offset_A = selected_modes[row_idx]*A.stride;

        // insert column elements
        for (size_t col_idx=0; col_idx<dim_ret; col_idx++) {
            ret[row_offset + 2*col_idx] = A[row_offset_A + selected_modes[col_idx]];
        }

        // insert column elements
        for (size_t col_idx=0; col_idx<dim_ret; col_idx++) {
            ret[row_offset + 2*col_idx+1] = A[row_offset_A + selected_modes[col_idx] + dim_A];
        }


        row_offset = (2*row_idx+1)*ret.stride;
        row_offset_A = (selected_modes[row_idx]+dim_A)*A.stride;

        // insert column elements
        for (size_t col_idx=0; col_idx<dim_ret; col_idx++) {
            ret[row_offset + 2*col_idx] = A[row_offset_A + selected_modes[col_idx]];
        }

        // insert column elements
        for (size_t col_idx=0; col_idx<dim_ret; col_idx++) {
            ret[row_offset + 2*col_idx+1] = A[row_offset_A + selected_modes[col_idx] + dim_A];
        }


    }

    return ret;

}


/**
@brief Call to add correction coming from the displacement to the diagonal elements of A_S (see Eq. (11) in arXiv 2010.15595)
@param A Hamilton matrix A defined by Eq. (4) of Ref. arXiv 2010.15595 (or Eq (4) of Ref. Craig S. Hamilton et. al, Phys. Rev. Lett. 119, 170501 (2017)).
(The output is returned via this variable)
@param gamma The diagonal correction according to see Eq. (11) in arXiv 2010.15595. Here gamma is ordered as a_1, a_1^* ,a_2, a_2^*, ...
*/
void
GaussianSimulationStrategyFast::diag_correction_of_A( matrix& A, matrix& gamma ) {

    // store gamma values into matrix A
    for (size_t row_idx=0; row_idx<A.rows; row_idx++) {
        A[row_idx*A.stride + row_idx] = gamma[row_idx];
    }


    return;
}

/**
@brief Call to calculate gamma according to Eq (9) of arXiv 2010.15595v3 in ordering a_1, a_1^*, a_2, a_2^* ....
@param Qinv An instace of matrix class containing the inverse of matrix Q calculated by method get_Qinv.
@param m The displacement \f$ \alpha \f$ defined by Eq (8) of Ref. arXiv 2010.15595
@param current_output The Fock representation of the current output for which the probability is calculated
@return Returns with the diagonal corrections.
*/
matrix
GaussianSimulationStrategyFast::CalcGamma( matrix& Qinv, matrix& m, PicState_int64& selected_modes ) {
//std::cout << "GaussianSimulationStrategyFast::diag_correction_of_A " << A.rows << " " << Qinv.rows << " " << Qinv.cols << " " << m.size() << std::endl;
//selected_modes.print_matrix();

    matrix gamma(Qinv.rows, 1);
    //memset(gamma.get_data(), 0, gamma.size()*sizeof(Complex16));
    for (size_t row_idx=0; row_idx<Qinv.rows; row_idx++) {

        size_t row_offset = row_idx*Qinv.stride;
        gamma[row_idx] = Complex16(0.0,0.0);

        for (size_t col_idx=0; col_idx<Qinv.rows; col_idx++) {
            gamma[row_idx] = gamma[row_idx] + mult_a_bconj( Qinv[row_offset + col_idx], m[col_idx] );
        }
    }

    // reorder the gamma matrix into a_1, a_1^*, a_2, a_2^* ...
    matrix gamma_reordered(Qinv.rows, 1);
    size_t num_of_modes = selected_modes.size();
    size_t total_number_of_modes = m.size()/2;
    for (size_t idx=0; idx<num_of_modes; idx++) {
        gamma_reordered[2*idx] = gamma[selected_modes[idx]];
        gamma_reordered[2*idx+1] = gamma[selected_modes[idx]+total_number_of_modes];
    }


    return gamma_reordered;
}

} // PIC
