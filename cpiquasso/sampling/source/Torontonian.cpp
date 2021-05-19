#include "Torontonian.h"
#include "TorontonianUtilities.h"
#include "common_functionalities.h"
#include "tbb/tbb.h"


#include <bitset>

namespace pic {

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
Torontonian::Torontonian(){

}

/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix for which the torontonian is calculated. (For example a covariance matrix of the Gaussian state.)
@return Returns with the instance of the class.
*/
Torontonian::Torontonian( matrix &mtx_in ){
    // in debug mode check the input matrix properties
    Update_mtx( mtx_in );
}

/**
@brief Default destructor of the class.
*/
Torontonian::~Torontonian(){

}

/**
@brief Call to calculate the torontonian of a complex matrix
@return Returns with the calculated torontonian
Calculation based on Eq. (2) of arXiv 2009.01177)
*/
double
Torontonian::calculate(){
    if ( mtx.rows != mtx.cols) {
        std::cout << "The input matrix should be square shaped, but matrix with " << mtx.rows << " rows and with " << mtx.cols << " columns was given" << std::endl;
        std::cout << "Returning zero" << std::endl;
        //return Complex16(0,0);
        return 0.0D;
    }

    if (mtx.rows == 0) {
        // the torontonian of an empty matrix is 1 by definition
        //return Complex16(1,0);
        return 1.0D;
    }
    else if (mtx.rows % 2 != 0) {
        //return Complex16(0.0, 0.0);
        return 0.0D;
    }


#if BLAS==0 // undefined BLAS
    int NumThreads = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS==1 // MKL
    int NumThreads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS==2 //OpenBLAS
    int NumThreads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

    const size_t dim = mtx.rows;


    const size_t dim_over_2 = mtx.rows / 2;
    unsigned long long permutation_idx_max = power_of_2( (unsigned long long) dim_over_2);

    tbb::combinable<RealM<double>> summands{[](){return RealM<double>(0.0);}};


    // for cycle over the permutations n/2 according to Eq (3.24) in arXiv 1805.12498
    tbb::parallel_for( tbb::blocked_range<unsigned long long>(0, permutation_idx_max, 1), [&](tbb::blocked_range<unsigned long long> r ) {

        RealM<double> &summand = summands.local();

        for ( unsigned long long permutation_idx=r.begin(); permutation_idx != r.end(); permutation_idx++) {

            // with unsigned char the type std::vector does not work for me
            //std::vector<unsigned char> bin_rep;
            //std::vector<unsigned char> positions_of_ones;
            std::vector<int> bin_rep;
            std::vector<int> positions_of_ones;
            bin_rep.reserve(dim_over_2);
            positions_of_ones.reserve(dim_over_2);

            for (int i = 1 << (dim_over_2-1); i > 0; i = i / 2) {
                if (permutation_idx & i) {
                    bin_rep.push_back(1);
                    positions_of_ones.push_back((bin_rep.size()-1));
                }
                else {
                    bin_rep.push_back(0);
                }
            }

            size_t number_of_ones = positions_of_ones.size();

            size_t dimension_of_B = 2 * number_of_ones;

            // matrix mtx corresponds to 1 - A^(Z), i.e. to the square matrix constructed from
            matrix B(dimension_of_B, dimension_of_B);
            for (size_t idx = 0; idx < number_of_ones; idx++) {
                for (size_t jdx = 0; jdx < number_of_ones; jdx++) {
                    B[idx*dimension_of_B + jdx]                  =
                        mtx[positions_of_ones[idx]*dim + (positions_of_ones[jdx])];
                    B[idx*dimension_of_B + jdx + number_of_ones] =
                        mtx[positions_of_ones[idx]*dim + (positions_of_ones[jdx]) + dim_over_2];
                    B[(idx + number_of_ones)*dimension_of_B + jdx] =
                        mtx[(positions_of_ones[idx]+dim_over_2)*dim + (positions_of_ones[jdx])];
                    B[(idx + number_of_ones)*dimension_of_B + jdx + number_of_ones] =
                        mtx[(positions_of_ones[idx]+dim_over_2)*dim + (positions_of_ones[jdx]) + dim_over_2];
                }
            }


            double factor =
                (number_of_ones + dim_over_2) % 2
                    ? -1.0D
                    : 1.0D;

            // calculating the determinant of B
            Complex16 determinant;
            if (number_of_ones != 0) {

                determinant = calc_determinant_cholesky_decomposition(B);
            }
            else{
                determinant = 1.0;
            }


            // calculating -1^(number of ones) / sqrt(det(1-A^(Z)))
            double sqrt_determinant = std::sqrt(determinant.real());
            double value = factor / sqrt_determinant;

            summand += value;

        }

    });

    double res = 0.0D;
    summands.combine_each([&res](RealM<double>& a) {
        res = res + a.get();
    });

#if BLAS==0 // undefined BLAS
    omp_set_num_threads(NumThreads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(NumThreads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(NumThreads);
#endif


    return res;
}

/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void
Torontonian::Update_mtx( matrix &mtx_in ){
    mtx_orig = mtx_in;

    size_t dim = mtx_in.rows;

    // Calculating B := 1 - A
    mtx = matrix(dim, dim);
    for (size_t idx = 0; idx < dim; idx++) {
        //Complex16 *row_B_idx = B.get_data() + idx * B.stride;
        //Complex16 *row_mtx_pos_idx = mtx.get_data() + positions_of_ones[idx] * mtx.stride;
        for (size_t jdx = 0; jdx < dim; jdx++) {
            mtx[idx * dim + jdx] = -1.0 * mtx_in[idx * mtx_in.stride + jdx];
        }
        mtx[idx * dim + idx] += Complex16(1.0, 0.0);
    }

    // Can scaling be used here since we have to calculate 1-A^Z?
    // It brings a multiplying for each determinant.
    // Should
    ScaleMatrix();
}



/**
@brief Call to scale the input matrix according to according to Eq (2.11) of in arXiv 1805.12498
@param mtx_in Input matrix defined by
*/
void Torontonian::ScaleMatrix(){

}





} // PIC
