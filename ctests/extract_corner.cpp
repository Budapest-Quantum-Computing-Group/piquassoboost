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

// Ctest 2021

// undefine NDEBUG macro to be able to perform asserts
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <vector>
#include <random>
#include "constants_tests.h"

#include "tasks_apply_to_C_and_G/extract_corner.h"

#ifdef __MPI__
#include <mpi.h>
#endif // MPI

int test_extract_corner(size_t dim, std::vector<size_t> modes_in);



/**
@brief Unit test case for the column extractation method
*/
int main() {

#ifdef __MPI__
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
#endif


    // Test with dimension 8 and qumodes {1, 3, 6}
    std::vector<size_t> modes_in_1 = {1, 3, 6};
    test_extract_corner(8, modes_in_1);
 
    // Test with dimension 12 and qumodes {0, 1, 3, 4, 6, 7, 8
    std::vector<size_t> modes_in_2 = {0, 1, 3, 4, 6, 7, 8};
    test_extract_corner(12, modes_in_2);
 
    // Test with dimension 4 and qumodes {1, 2, 3}
    std::vector<size_t> modes_in_3 = {1, 2, 3};
    test_extract_corner(4, modes_in_3);
 
    // Test with dimension 3 and qumodes {1}
    std::vector<size_t> modes_in_4 = {1};
    test_extract_corner(3, modes_in_4);
    
    return 0;
};


int test_extract_corner(const size_t dim, std::vector<size_t> modes_in){
    printf("\n\n****************************************\n");
    printf("Test of corner extractation method\n");
    printf("****************************************\n\n\n");

    // initialize random generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<double> distribution(0.0, 1.0);

    // allocate memory for the modes indices
    //std::vector<size_t> modes_in = {1, 3, 6};
    size_t dimY_cut = modes_in.size();
    size_t dimX_expected = modes_in.size();

    // allocate matrix array for input
    // note that here the matrix is already cut to the expected number of rows
    //constexpr size_t dim = 8;
    pic::matrix mtx_in = pic::matrix(dimY_cut, dim);
    
    // fill up matrix with random elements
    for (size_t row_idx = 0; row_idx < dimY_cut; row_idx++) {
        for (size_t col_idx = 0; col_idx < dim; col_idx++) {
            double randnum1 = distribution(generator);
            double randnum2 = distribution(generator);
            mtx_in[row_idx * dim + col_idx] = pic::Complex16(randnum1, randnum2);
        }
    }
        
    // bool array with the logical indexes of the columns of the matrix
    bool cols_logical[dim];
    memset( &(cols_logical[0]), 0, dim*sizeof(bool));
    
    // expected bool array
    bool cols_logical_expected[dim];
    memset( &(cols_logical_expected[0]), 0, dim*sizeof(bool));
    for ( size_t mode : modes_in ){
        cols_logical_expected[mode] = 1;
    }

    // allocate memory for the output matrix
    pic::matrix cols_out = pic::matrix(dimY_cut, dimX_expected);
    
    // allocate memory for the expected results of the matrix
    pic::matrix cols_out_expected = pic::matrix(dimY_cut, dimX_expected);
    
    // fill up the expected matrix with expected numbers from the matrix sent in
    for (size_t row_idx = 0; row_idx < dimY_cut; row_idx++) {
        for (size_t modes_idx = 0; modes_idx < dimX_expected; modes_idx++) {
            cols_out_expected[row_idx * dimX_expected + modes_idx] = mtx_in[row_idx * dim + modes_in[modes_idx]];
        }
    }

    // print the matrices on standard output
    std::cout << "Matrix input:" << std::endl;
    mtx_in.print_matrix();
    std::cout << "Matrix expected:" << std::endl;
    cols_out_expected.print_matrix();
    
    // Create instance of the Extract_Rows class with the given input
    pic::Extract_Corner extract_corner = pic::Extract_Corner(mtx_in, cols_out, modes_in, cols_logical);

    // tbb flow message for the method call
    tbb::flow::continue_msg msg = tbb::flow::continue_msg();

    // Calculation
    extract_corner(msg);
    
    // Intentionally make the test failing most likely
    //cols_out[2] = pic::Complex16(0, 1);
    
    // print the result matrix
    std::cout << "Matrix resulted:" << std::endl;
    cols_out.print_matrix();
    
    // Comparing expected and resulted matrix elements
    for (size_t elem_idx = 0; elem_idx < dimX_expected * dimY_cut; elem_idx++) {
        pic::Complex16 diff = cols_out[elem_idx] - cols_out_expected[elem_idx];
        assert(std::abs(diff) < pic::epsilon);
        //std::cout << diff <<std::endl;
        
        //assert(cols_out[elem_idx] == cols_out_expected[elem_idx]);
    }
    // Comparing expected and resulted logical outputs
    for (size_t col_idx = 0; col_idx < dim; col_idx++){
        assert(cols_logical[col_idx] == cols_logical_expected[col_idx]);
    }

    
    
    std::cout << "Test passed. " << std::endl;


#ifdef __MPI__
    // Finalize the MPI environment.
    MPI_Finalize();
#endif

    return 0;
}

