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
#include "dot.h"



/**
@brief Unit test to check matrix multiplication with strides
*/
int main() {

    printf("\n\n****************************************\n");
    printf("Test to calculate strided matrix product\n");
    printf("****************************************\n\n\n");

    // seed the random generator
    srand ( time ( NULL));


    // allocate matrix array for matrix A
    size_t Arows = 4;
    size_t Acols = 5;
    pic::matrix A = pic::matrix(Arows, Acols);

    // fill up matrix with random elements
    for (size_t row_idx = 0; row_idx < A.rows; row_idx++) {
        for (size_t col_idx = 0; col_idx < A.cols; col_idx++) {
            double randnum1 = ((double)rand()/RAND_MAX*2 - 1.0);
            double randnum2 = ((double)rand()/RAND_MAX*2 - 1.0);
            A[row_idx * A.stride + col_idx] = pic::Complex16(randnum1, randnum2);
        }
    }


    // allocate matrix array for matrix A
    size_t Brows = A.cols;
    size_t Bcols = 3;
    size_t Bstride = 6;
    pic::matrix B = pic::matrix(Brows, Bcols, Bstride);

    // fill up matrix with random elements
    for (size_t row_idx = 0; row_idx < B.rows; row_idx++) {
        for (size_t col_idx = 0; col_idx < B.cols; col_idx++) {
            double randnum1 = ((double)rand()/RAND_MAX*2 - 1.0);
            double randnum2 = ((double)rand()/RAND_MAX*2 - 1.0);
            B[row_idx * B.stride + col_idx] = pic::Complex16(randnum1, randnum2);
        }
    }

    // calculate the product of the matrices by piquasso library
    pic::matrix C = dot(A, B);


    // calculate the expexted outcome of the matrix product
    pic::matrix C_expected(Arows, Bcols);
    for (size_t row_idx = 0; row_idx < C_expected.rows; row_idx++) {
        for (size_t col_idx = 0; col_idx < C_expected.cols; col_idx++) {

            C_expected[row_idx*C_expected.stride + col_idx] = pic::Complex16(0.0, 0.0);

            for (size_t idx=0; idx<A.cols; idx++) {
                C_expected[row_idx*C_expected.stride + col_idx] += A[row_idx*A.stride + idx] * B[idx*B.stride + col_idx];
            }


        }
    }

    // calculate the difference between the results
    pic::Complex16 diff(0.0, 0.0);
    for ( size_t idx=0; idx<C.size(); idx++) {
        diff += C[idx] - C_expected[idx];
    }

    std::cout << "the difference from the expected result is" << std::abs(diff) << std::endl;
    assert( std::abs(diff)<1e-13 );
    std::cout << "Test passed"  << std::endl;



  return 0;

};
