#include "Torontonian.h"
#include "TorontonianUtilities.hpp"

#include "common_functionalities.h"

#include "tbb/tbb.h"


#include <bitset>

/*
    Questions
        scaling?
        templating?
        bitátírás a perm_numberből?

    Remarks
        átírtam a B:= 1-A számítást

    Developments:
        Block matrices based on : http://www.netlib.org/utk/papers/factor/node9.html
        Storing just the lower matrix elements (since it is selfadjoint)
*/


extern "C" {


/// Definition of the LAPACKE_zgetri function from LAPACKE to calculate the LU decomposition of a matrix
int LAPACKE_zgetrf( int matrix_layout, int n, int m, pic::Complex16* a, int lda, int* ipiv );
}

namespace pic {

pic::Complex16 determinant_byLU_decomposition( pic::matrix& mtx ){
    pic::matrix& Q = mtx;

    // calculate the inverse of matrix Q
    int* ipiv = (int*)scalable_aligned_malloc( Q.stride*sizeof(int), CACHELINE);
    LAPACKE_zgetrf( LAPACK_ROW_MAJOR, Q.rows, Q.cols, Q.get_data(), Q.stride, ipiv );

    //  calculate the determinant of Q
    pic::Complex16 Qdet_cmplx(1.0,0.0);
    for (size_t idx=0; idx<Q.rows; idx++) {
        if (ipiv[idx] != idx+1) {
            Qdet_cmplx = -Qdet_cmplx * Q[idx*Q.stride + idx];
        }
        else {
            Qdet_cmplx = Qdet_cmplx * Q[idx*Q.stride + idx];
        }

    }
    //std::cout << "complex det: " << Qdet_cmplx << std::endl;
    return Qdet_cmplx;
}



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
    std::cout << "Torontonian mtx in: "<<std::endl;
    mtx_in.print_matrix();
}

/**
@brief Default destructor of the class.
*/
Torontonian::~Torontonian(){

}

/**
@brief Call to calculate the torontonian of a complex matrix
@return Returns with the calculated torontonian

Calculation based on: https://arxiv.org/pdf/1807.01639.pdf
*/
// do we need here complex numbers?
//Complex16
double
Torontonian::calculate(){
    if ( mtx.rows != mtx.cols) {
        std::cout << "The input matrix should be square shaped, bu matrix with " << mtx.rows << " rows and with " << mtx.cols << " columns was given" << std::endl;
        std::cout << "Returning zero" << std::endl;
        //return Complex16(0,0);
        return 0.0D;
    }

    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        //return Complex16(1,0);
        return 1.0D;
    }
    else if (mtx.rows % 2 != 0) {
        // the hafnian of odd shaped matrix is 0 by definition
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


    // hafnian: thread local storages for the partial hafnians
    //tbb::combinable<Complex32> summands{[](){return Complex32(0.0,0.0);}};

    // torontonian: thread local storages for the partial hafnians
    tbb::combinable<double> summands{[](){return 0.0D;}};



    // for cycle over the permutations n/2 according to Eq (3.24) in arXiv 1805.12498
    tbb::parallel_for( tbb::blocked_range<unsigned long long>(0, permutation_idx_max, 1), [&](tbb::blocked_range<unsigned long long> r ) {

        double &summand = summands.local();
        //Complex32 &summand = summands.local();

        for ( unsigned long long permutation_idx=r.begin(); permutation_idx != r.end(); permutation_idx++) {
//for ( unsigned long long permutation_idx=0; permutation_idx < permutation_idx_max; permutation_idx++) {
/*
    Complex32 summand(0.0,0.0);

    for (unsigned long long permutation_idx = 0; permutation_idx < permutation_idx_max; permutation_idx++) {
*/
            std::cout << permutation_idx << " : " << std::bitset<32>(permutation_idx) <<std::endl;


            // get the binary representation of permutation_idx
            // also get the number of 1's in the representation and their position as i and i + dim_over_2
            // hafnian: also get the number of 1's in the representation and their position as 2*i and 2*i+1 in consecutive slots of the vector bin_rep
            
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

            // matrix mtx corresponds to id - A^(Z), i.e. to the square matrix constructed from
            // the elements of mtx = 1-A indexed by the rows and colums, where the binary representation of
            // permutation_idx was 1
            // details in Eq. (12) https://arxiv.org/pdf/1807.01639.pdf
            // B = (1 - A^(Z))
            // Calculating B^(Z)
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
                //B[idx * dimension_of_B + idx] += Complex16(1.0, 0.0);
                //B[(idx + number_of_ones)*dimension_of_B + idx + number_of_ones] += Complex16(1.0, 0.0);
            }

            B.print_matrix();
            // calculating -1^(number of ones)
            // !!! -1 ^ (number of ones - dim_over_2) ???
            // Do we need complex here???
            /*Complex32 factor = 
                (number_of_ones + dim_over_2) % 2 
                    ? Complex32(1.0, 0.0)
                    : Complex32(-1.0, 0.0);
                    */
            double factor = 
                (number_of_ones + dim_over_2) % 2 
                    ? -1.0D
                    : 1.0D;

            // calculating the determinant of B
            Complex16 determinant;
            if (number_of_ones != 0) {
                // testing purpose (the matrix is not positive definite and selfadjoint)
                //determinant = determinant_byLU_decomposition(B);
                //determinant = calc_determinant_of_selfadjoint_hessenberg_matrix<matrix, Complex16>(B);
                determinant = calc_determinant_cholesky_decomposition<matrix, Complex16>(B);
                // hafnian: calculate trace
                //traces = calc_power_traces<matrix32, Complex32>(B, dim_over_2);
            }
            else{
                determinant = 1.0;
                //memset( traces.get_data(), 0.0, traces.rows*traces.cols*sizeof(Complex32));
            }

            //std::cout<<"Det: "<< determinant.real()<<std::endl;

            // calculating -1^(number of ones) / sqrt(det(1-A^(Z)))
            double sqrt_determinant = std::sqrt(determinant.real());
            double value = factor / sqrt_determinant;

            summand = summand + value;

        }

    });

    // hafnian: the resulting Hafnian of matrix mat

    //Complex32 res(0,0);
    double res = 0.0D;
    summands.combine_each([&res](double a) {
        res = res + a;
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

    std::cout << "Modified matrix:" << std::endl;
    mtx.print_matrix();

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
    //mtx = mtx_orig.copy();
}





} // PIC