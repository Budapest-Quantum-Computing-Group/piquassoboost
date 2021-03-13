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
create_repeated_mtx( pic::matrix& A, pic::PicState_int64& current_output ) {

    size_t dim_A_S = sum(current_output);
    size_t dim_A = current_output.size();

    pic::matrix A_S(2*dim_A_S, 2*dim_A_S);
    size_t row_idx = 0;
    for (size_t idx=0; idx<current_output.size(); idx++) {
        for (size_t row_repeat=0; row_repeat<current_output[idx]; row_repeat++) {

            size_t row_offset = row_idx*A_S.stride;
            size_t row_offset_A = idx*A.stride;
            size_t col_idx = 0;
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

    // array of modes describing the number of photons in the individual modes
    pic::PicState_int64 modes(dim/2);
    modes[0] = 1;
    modes[1] = 1;



    // print the matrix on standard output
    mtx.print_matrix();



    // hafnian calculated by algorithm PowerTraceHafnian
    pic::PowerTraceHafnian hafnian_calculator = pic::PowerTraceHafnian( mtx );
    pic::Complex16 hafnian_powertrace = hafnian_calculator.calculate();

    // calculate the hafnian by the recursive method

    // now calculated the hafnian of the whole matrix using the value calculated for the submatrix
    pic::PowerTraceHafnianRecursive hafnian_calculator_recursive = pic::PowerTraceHafnianRecursive( mtx, modes );
    pic::Complex16 hafnian_powertrace_recursive = hafnian_calculator_recursive.calculate();


    std::cout << "the calculated hafnian with the power trace method: " << hafnian_powertrace << std::endl;
    std::cout << "the calculated hafnian with the recursive powertrace method: " << hafnian_powertrace_recursive << std::endl;




  return 0;

};
