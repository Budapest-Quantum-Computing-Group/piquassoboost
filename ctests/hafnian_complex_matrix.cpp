#include <stdio.h>
#include <vector>
#include <random>
#include <time.h>

#include "matrix.h"
#include "PowerTraceHafnian.h"

#include "tbb/tbb.h"


/**
@brief Unit test case for the hafnian of complex symmetric matrices
*/
int main() {

    printf("\n\n****************************************\n");
    printf("Test of hafnian of random complex random matrix\n");
    printf("****************************************\n\n\n");

    // seed the random generator
    srand ( time ( NULL));


    // allocate matrix array
    size_t dim = 4;
    pic::matrix mtx = pic::matrix(dim, dim);

    // fill up matrix with random elements
    for (size_t row_idx = 0; row_idx < dim; row_idx++) {
        for (size_t col_idx = 0; col_idx <= row_idx; col_idx++) {
            double randnum1 = ((double)rand()/RAND_MAX*2 - 1.0);
            double randnum2 = ((double)rand()/RAND_MAX*2 - 1.0);
            mtx[row_idx * dim + col_idx] = pic::Complex16(randnum1, randnum2);
            mtx[col_idx* dim + row_idx] = mtx[row_idx * dim + col_idx];
        }
    }

    // print the matrix on standard output
    mtx.print_matrix();

    // set the expected outcome for the hafnian
    pic::Complex16 hafnian_expected = mtx[1] * mtx[11] + mtx[2] * mtx[7] + mtx[3] * mtx[6];


    // calculate the hafnian by the eigenvalue method
    pic::PowerTraceHafnian hafnian_calculator = pic::PowerTraceHafnian( mtx );
    pic::Complex16 hafnian_eigen = hafnian_calculator.calculate();
    tbb::tick_count t1b = tbb::tick_count::now();


    std::cout << "the calculated hafnian with the eigenvalue method: " << hafnian_eigen << std::endl;
    std::cout << "the calculated hafnian with trivial method: " << hafnian_expected << std::endl;




  return 0;

};
