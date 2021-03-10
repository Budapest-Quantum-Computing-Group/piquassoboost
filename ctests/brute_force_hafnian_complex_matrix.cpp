#include <stdio.h>
#include <random>
#include "matrix.h"
#include "BruteForceHafnian.h"
#include "PowerTraceHafnian.h"
#include "tbb/tbb.h"


/**
@brief Unit test case for the hafnian of complex symmetric matrices: compare brute force method with power trace method
*/
int main() {

    printf("\n\n****************************************************************************\n");
    printf("Test of hafnian of random complex random matrix: compare brute force method with power trace method\n");
    printf("********************************************************************************\n\n\n");

    // initialize random generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);


    // allocate matrix array
    size_t dim = 4;
    pic::matrix mtx = pic::matrix(dim, dim);

    // fill up matrix with random elements
    double max_value = 0.0;
    for (size_t row_idx = 0; row_idx < dim; row_idx++) {
        for (size_t col_idx = 0; col_idx <= row_idx; col_idx++) {
            double randnum1 = distribution(generator);
            double randnum2 = distribution(generator);
            mtx[row_idx * dim + col_idx] = pic::Complex16(randnum1, randnum2);
            mtx[col_idx* dim + row_idx] = mtx[row_idx * dim + col_idx];

            if ( max_value < std::abs((mtx[row_idx * dim + col_idx]))) {
                max_value = std::abs((mtx[row_idx * dim + col_idx]));
            }
        }
    }

    // scale matrix elements to fit into 0<= |A_ij| < 1
    for (size_t idx=0; idx<mtx.size(); idx++) {
        mtx[idx] = mtx[idx]/max_value;
    }

tbb::tick_count t0 = tbb::tick_count::now();
    // set the expected outcome for the hafnian
    pic::Complex16 hafnian_expected = mtx[1] * mtx[11] + mtx[2] * mtx[7] + mtx[3] * mtx[6];
tbb::tick_count t1 = tbb::tick_count::now();

tbb::tick_count t2 = tbb::tick_count::now();
    // calculate the hafnian by the brute force method
    pic::BruteForceHafnian hafnian_calculator = pic::BruteForceHafnian( mtx );
    pic::Complex16 hafnian_brute_forcen = hafnian_calculator.calculate();
tbb::tick_count t3 = tbb::tick_count::now();

tbb::tick_count t4 = tbb::tick_count::now();
    // calculate the hafnian by the eigenvalue method
    pic::PowerTraceHafnian hafnian_calculator_powertrace = pic::PowerTraceHafnian( mtx );
    pic::Complex16 hafnian_power_trace = hafnian_calculator_powertrace.calculate();
tbb::tick_count t5 = tbb::tick_count::now();


std::cout << (t1-t0).seconds() << " " <<(t3-t2).seconds() << " " << (t5-t4).seconds() << std::endl;




    std::cout << "the calculated hafnian with the powertrace method: " << hafnian_power_trace << std::endl;
    std::cout << "the calculated hafnian with the brute force method: " << hafnian_brute_forcen << std::endl;
    std::cout << "the calculated hafnian with trivial method: " << hafnian_expected << std::endl;

    assert( std::abs(hafnian_power_trace-hafnian_expected)<1e-13 );
    std::cout << "Test passed"  << std::endl;




  return 0;

};
