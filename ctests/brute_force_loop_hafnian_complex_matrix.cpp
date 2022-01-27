/**
 * Copyright 2021 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <random>
#include "matrix.h"
#include "BruteForceLoopHafnian.h"
#include "PowerTraceLoopHafnian.h"
#include "tbb/tbb.h"


#ifdef __MPI__
#include <mpi.h>
#endif // MPI

/**
@brief Unit test case for the loop hafnian of complex symmetric matrices: compare brute force method with power trace method
*/
int main() {

    printf("\n\n****************************************************************************\n");
    printf("Test of loop hafnian of random complex random matrix: compare brute force method with power trace method\n");
    printf("********************************************************************************\n\n\n");

    // seed the random generator
    srand ( time ( NULL));

#ifdef __MPI__
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
#endif

    // allocate matrix array
    size_t dim = 4;
    pic::matrix mtx = pic::matrix(dim, dim);

    // fill up matrix with random elements
    double max_value = 0.0;
    for (size_t row_idx = 0; row_idx < dim; row_idx++) {
        for (size_t col_idx = 0; col_idx <= row_idx; col_idx++) {
            double randnum1 = ((double)rand()/RAND_MAX*2 - 1.0);
            double randnum2 = ((double)rand()/RAND_MAX*2 - 1.0);
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
    pic::Complex16 hafnian_expected = mtx[0*mtx.stride + 1] * mtx[2*mtx.stride + 3]
                                    + mtx[0*mtx.stride + 2] * mtx[1*mtx.stride + 3]
                                    + mtx[0*mtx.stride + 3] * mtx[1*mtx.stride + 2]
                                    + mtx[0*mtx.stride + 0] * mtx[1*mtx.stride + 1] * mtx[2*mtx.stride + 3]
                                    + mtx[0*mtx.stride + 1] * mtx[2*mtx.stride + 2] * mtx[3*mtx.stride + 3]
                                    + mtx[0*mtx.stride + 2] * mtx[1*mtx.stride + 1] * mtx[3*mtx.stride + 3]
                                    + mtx[0*mtx.stride + 0] * mtx[2*mtx.stride + 2] * mtx[1*mtx.stride + 3]
                                    + mtx[0*mtx.stride + 0] * mtx[3*mtx.stride + 3] * mtx[1*mtx.stride + 2]
                                    + mtx[0*mtx.stride + 3] * mtx[1*mtx.stride + 1] * mtx[2*mtx.stride + 2]
                                    + mtx[0*mtx.stride + 0] * mtx[1*mtx.stride + 1] * mtx[2*mtx.stride + 2] * mtx[3*mtx.stride + 3];

tbb::tick_count t1 = tbb::tick_count::now();

tbb::tick_count t2 = tbb::tick_count::now();
    // calculate the hafnian by brute force method
    pic::BruteForceLoopHafnian hafnian_calculator = pic::BruteForceLoopHafnian( mtx );
    pic::Complex16 hafnian_brute_forcen = hafnian_calculator.calculate();
tbb::tick_count t3 = tbb::tick_count::now();

tbb::tick_count t4 = tbb::tick_count::now();
    // calculate the hafnian by the eigenvalue method
    pic::PowerTraceLoopHafnian hafnian_calculator_powertrace = pic::PowerTraceLoopHafnian( mtx );
    pic::Complex16 hafnian_power_trace = hafnian_calculator_powertrace.calculate();
tbb::tick_count t5 = tbb::tick_count::now();


std::cout << (t1-t0).seconds() << " " <<(t3-t2).seconds() << " " << (t5-t4).seconds() << std::endl;




    std::cout << "the calculated hafnian with the powertrace method: " << hafnian_power_trace << std::endl;
    std::cout << "the calculated hafnian with the brute force method: " << hafnian_brute_forcen << std::endl;
    std::cout << "the calculated hafnian with trivial method: " << hafnian_expected << std::endl;

    assert( std::abs(hafnian_power_trace-hafnian_expected)<1e-13 );
    std::cout << "Test passed"  << std::endl;


#ifdef __MPI__
    // Finalize the MPI environment.
    MPI_Finalize();
#endif



  return 0;

};
