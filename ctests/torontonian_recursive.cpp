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

#include <vector>

#include <random>
#include <chrono>
#include <string>

#include "TorontonianUtilities.h"
#include "Torontonian.h"
#include "TorontonianRecursive.h"


#include "matrix_real.h"
#include "matrix.h"

#include "dot.h"

#include "matrix_helper.hpp"
#include "constants_tests.h"

#ifdef __MPI__
#include <mpi.h>
#endif // MPI


/**
@brief Unit test to compare torontonian calculators implemented in piqausso boost
*/
int main(){

    constexpr size_t dim = 40;

#ifdef __MPI__
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
#endif


    // create random matrix to calculate the torontonian
    pic::matrix_real matrix = pic::get_random_density_matrix_real<pic::matrix_real, double>(dim);


#ifdef __MPI__
    // ensure that each MPI process gets the same input matrix from rank 0
    void* syncronized_data = (void*)matrix.get_data();
    MPI_Bcast(syncronized_data, matrix.size()*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    // This is a hack for being able to compare the results of the
    // classic algorithm currently implemented with the recursive algoritms
    pic::matrix matrix_complex(dim, dim);
    for (int row_idx = 0; row_idx < dim; row_idx++){
        for (int col_idx = 0; col_idx < dim; col_idx++){
            matrix_complex[row_idx * matrix_complex.stride + col_idx] = matrix[row_idx * matrix.stride + col_idx];
        }
    }



    // create class instance for torontonian calculator
    tbb::tick_count t0 = tbb::tick_count::now();
    pic::Torontonian torontonian_calculator(matrix_complex);
    double result = torontonian_calculator.calculate();
    tbb::tick_count t1 = tbb::tick_count::now();

    std::cout << "torontonian calculator: " << result << std::endl;


    // create class instance for recursive torontonian calculator
    tbb::tick_count t2 = tbb::tick_count::now();
    pic::TorontonianRecursive recursive_torontonian_calculator(matrix);
    double result_recursive_extended = recursive_torontonian_calculator.calculate(true);
    tbb::tick_count t3 = tbb::tick_count::now();

    std::cout << "recursive torontonian calculator extended precision: " << result_recursive_extended<< std::endl;


    // create class instance for recursive torontonian calculator
    tbb::tick_count t4 = tbb::tick_count::now();
    recursive_torontonian_calculator = pic::TorontonianRecursive(matrix);
    double result_recursive_basic = recursive_torontonian_calculator.calculate(false);
    tbb::tick_count t5 = tbb::tick_count::now();

    std::cout << "recursive torontonian calculator basic precision: " << result_recursive_basic<< std::endl;

    std::cout << (t1-t0).seconds() << " " << (t3-t2).seconds() << " " << (t1-t0).seconds()/(t3-t2).seconds() << std::endl;

#ifdef __MPI__
    // Finalize the MPI environment.
    MPI_Finalize();
#endif

    return 0;

}
