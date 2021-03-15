#include <stdio.h>
#include <vector>
#include <random>
#include <time.h>

#include "matrix.h"
#include "PowerTraceHafnian.h"
#include "PowerTraceHafnianRecursive.h"

#include "tbb/tbb.h"


/**
@brief Call to calculate sum of integers stored in a PicState
@param vec a container if integers
@return Returns with the sum of the elements of the container
*/
static inline int64_t
sum( pic::PicState_int64 &vec) {

    int64_t ret=0;

    size_t element_num = vec.cols;
    int64_t* data = vec.get_data();
    for (size_t idx=0; idx<element_num; idx++ ) {
        ret = ret + data[idx];
    }
    return ret;
}



pic::matrix
create_repeated_mtx( pic::matrix& A, pic::PicState_int64& filling_factors ) {

    size_t dim_A_S = sum(filling_factors);
    size_t dim_A = filling_factors.size();

    pic::matrix A_S(2*dim_A_S, 2*dim_A_S);
    size_t row_idx = 0;
    for (size_t idx=0; idx<filling_factors.size(); idx++) {
        for (size_t row_repeat=0; row_repeat<filling_factors[idx]; row_repeat++) {

            size_t row_offset = row_idx*A_S.stride;
            size_t row_offset_A = idx*A.stride;
            size_t col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<filling_factors.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<filling_factors[jdx]; col_repeat++) {
                    A_S[row_offset + col_idx] = A[row_offset_A + jdx];
                    col_idx++;
                }
            }

            col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<filling_factors.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<filling_factors[jdx]; col_repeat++) {
                    A_S[row_offset + col_idx + dim_A_S] = A[row_offset_A + jdx + dim_A];
                    col_idx++;
                }

            }

            row_offset = (row_idx+dim_A_S)*A_S.stride;
            row_offset_A = (idx+dim_A)*A.stride;
            col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<filling_factors.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<filling_factors[jdx]; col_repeat++) {
                    A_S[row_offset + col_idx] = A[row_offset_A + jdx];
                    col_idx++;
                }
            }

            col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<filling_factors.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<filling_factors[jdx]; col_repeat++) {
                    A_S[row_offset + col_idx + dim_A_S] = A[row_offset_A + jdx + dim_A];
                    col_idx++;
                }

            }


            row_idx++;
        }


    }

    return A_S;

}


/**
@brief Transforms the covariance matrix in the basis \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ...  a_n^* \f$  into the basis \f$ a_1, a_1^*,a_2, a_2^*, ... a_n, a_n^* \f$ suitable for
the PowerTraceHafnianRecursive algorithm.
@param mtx A covariance matrix in the basis \f$ a_1, a_2, ... a_n,, a_1^*, a_2^*, ...  a_n^* \f$.
@return Returns with the covariance matrix in the basis \f$ a_1, a_1^*,a_2, a_2^*, ... a_n, a_n^* \f$.
*/
pic::matrix
getPermutedMatrix( pic::matrix& mtx) {


    pic::matrix res(mtx.rows, mtx.cols);


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
@brief Unit test case for the recursive hafnian of complex symmetric matrices
*/
int main() {

    printf("\n\n****************************************\n");
    printf("Test of hafnian of random complex random matrix\n");
    printf("****************************************\n\n\n");

    // initialize random generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<double> distribution(0.0, 1.0);


    // allocate matrix array for the larger matrix
    size_t dim = 8;
    pic::matrix mtx = pic::matrix(dim, dim);

    // fill up matrix with random elements
    for (size_t row_idx = 0; row_idx < dim; row_idx++) {
        for (size_t col_idx = 0; col_idx <= row_idx; col_idx++) {
            double randnum1 = distribution(generator);
            double randnum2 = distribution(generator);
            mtx[row_idx * dim + col_idx] = pic::Complex16(randnum1, randnum2);
            mtx[col_idx* dim + row_idx] = mtx[row_idx * dim + col_idx];
        }
    }


    // array of modes describing the occupancy of the individual modes
    pic::PicState_int64 filling_factors(dim/2);
    filling_factors[0] = 6;
    filling_factors[1] = 6;
    filling_factors[2] = 5;
    filling_factors[3] = 4;


    // matrix containing the repeated rows and columns
    pic::matrix&& mtx_repeated = create_repeated_mtx(mtx, filling_factors);



    // print the matrix on standard output
    //mtx.print_matrix();
    //mtx_repeated.print_matrix();
    //mtx_repeated=getPermutedMatrix(mtx_repeated);

    // hafnian calculated by algorithm PowerTraceHafnian
    tbb::tick_count t0 = tbb::tick_count::now();
    pic::PowerTraceHafnian hafnian_calculator = pic::PowerTraceHafnian( mtx_repeated );
    pic::Complex16 hafnian_powertrace = hafnian_calculator.calculate();
    tbb::tick_count t1 = tbb::tick_count::now();


    // calculate the hafnian by the recursive method

    // now calculated the hafnian of the whole matrix using the value calculated for the submatrix
    pic::matrix &&mtx_permuted = getPermutedMatrix(mtx);
    tbb::tick_count t2 = tbb::tick_count::now();
    pic::PowerTraceHafnianRecursive hafnian_calculator_recursive = pic::PowerTraceHafnianRecursive( mtx_permuted, filling_factors );
    pic::Complex16 hafnian_powertrace_recursive = hafnian_calculator_recursive.calculate();
    tbb::tick_count t3 = tbb::tick_count::now();


    std::cout << "the calculated hafnian with the power trace method: " << hafnian_powertrace << std::endl;
    std::cout << "the calculated hafnian with the recursive powertrace method: " << hafnian_powertrace_recursive << std::endl;


    std::cout << (t1-t0).seconds() << " " << (t3-t2).seconds() << " " << (t1-t0).seconds()/(t3-t2).seconds() << std::endl;




  return 0;

};
