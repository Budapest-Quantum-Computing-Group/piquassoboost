#include <stdio.h>
#include <vector>
#include <random>
#include <time.h>
#include <vector>


#include "matrix.h"
#include "PowerTraceHafnian.h"
#include "PowerTraceHafnianRecursive.h"
#include "RepeatedColumnPairs.h"


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
    memset( A_S.get_data(), 0, A_S.size()*sizeof(pic::Complex16));
    size_t row_idx = 0;
    for (size_t idx=0; idx<filling_factors.size(); idx++) {
        for (size_t row_repeat=0; row_repeat<filling_factors[idx]; row_repeat++) {

            size_t row_offset = row_idx*A_S.stride;
            size_t row_offset_A = idx*A.stride;
            size_t col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<filling_factors.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<filling_factors[jdx]; col_repeat++) {
                    if ( (row_idx == col_idx) || (idx != jdx) ) {
                        A_S[row_offset + col_idx] = A[row_offset_A + jdx];
                    }
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
@brief Unit test case for the recursive hafnian of complex symmetric matrices
*/
int main() {

    printf("\n\n****************************************\n");
    printf("Test of hafnian of random complex random matrix\n");
    printf("****************************************\n\n\n");

    // seed the random generator
    srand ( time ( NULL));


    // allocate matrix array for the larger matrix
    size_t dim = 20;
    pic::matrix mtx = pic::matrix(dim, dim);

    // fill up matrix with random elements
    for (size_t row_idx = 0; row_idx < dim; row_idx++) {
        for (size_t col_idx = 0; col_idx <= row_idx; col_idx++) {
            if ( row_idx == col_idx ) continue;

            double randnum1 = ((double)rand()/RAND_MAX*2 - 1.0)*10;
            double randnum2 = ((double)rand()/RAND_MAX*2 - 1.0)*10;
            mtx[row_idx * dim + col_idx] = pic::Complex16(randnum1, randnum2);
            mtx[col_idx* dim + row_idx] = mtx[row_idx * dim + col_idx];
        }
    }


    // array of modes describing the occupancy of the individual modes
    pic::PicState_int64 filling_factors(dim/2);
    for (size_t idx=0; idx<filling_factors.size(); idx++) {
        filling_factors[idx] = 1;
    }

    filling_factors[0] = 2;
    filling_factors[1] = 1;
    filling_factors[2] = 2;
    filling_factors[3] = 0;
    filling_factors[4] = 2;
    filling_factors[5] = 4;
    filling_factors[6] = 1;
    filling_factors[7] = 2;
    filling_factors[8] = 2;
    filling_factors[9] = 0;

    // matrix containing the repeated rows and columns
    pic::matrix&& mtx_repeated = create_repeated_mtx(mtx, filling_factors);


    pic::PicState_int64 repeated_column_pairs;
    pic::matrix mtx_recursive;
    ConstructMatrixForRecursivePowerTrace(mtx, filling_factors, mtx_recursive, repeated_column_pairs);


    // hafnian calculated by algorithm PowerTraceHafnian
    tbb::tick_count t0 = tbb::tick_count::now();
    pic::PowerTraceHafnian hafnian_calculator = pic::PowerTraceHafnian( mtx_repeated );
    pic::Complex16 hafnian_powertrace = hafnian_calculator.calculate();
    tbb::tick_count t1 = tbb::tick_count::now();

    // calculate the hafnian by the recursive method

    // now calculated the hafnian of the whole matrix using the value calculated for the submatrix
    tbb::tick_count t2 = tbb::tick_count::now();
    pic::PowerTraceHafnianRecursive hafnian_calculator_recursive = pic::PowerTraceHafnianRecursive( mtx_recursive, repeated_column_pairs );
    pic::Complex16 hafnian_powertrace_recursive = hafnian_calculator_recursive.calculate();
    tbb::tick_count t3 = tbb::tick_count::now();


    std::cout << "the calculated hafnian with the power trace method: " << hafnian_powertrace << std::endl;
    std::cout << "the calculated hafnian with the recursive powertrace method: " << hafnian_powertrace_recursive << std::endl;


    std::cout << (t1-t0).seconds() << " " << (t3-t2).seconds() << " " << (t1-t0).seconds()/(t3-t2).seconds() << std::endl;



  return 0;

};
