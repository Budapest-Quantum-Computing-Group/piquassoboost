// Ctest 2021

// undefine NDEBUG macro to be able to perform asserts
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <vector>
#include <random>
#include "constants_tests.h"

#include "tasks_apply_to_C_and_G/extract_corner.h"

/**
@brief Unit test case for the column extractation method
*/
int main() {
    printf("\n\n****************************************\n");
    printf("Test of corner extractation method\n");
    printf("****************************************\n\n\n");

    // initialize random generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<double> distribution(0.0, 1.0);

    // allocate memory for the modes indices
    std::vector<size_t> modes_in = {1, 3, 6};
    size_t dimY_cut = modes_in.size();
    size_t dimX_expected = modes_in.size();

    // allocate matrix array for input
    // note that here the matrix is already cut to the expected number of rows
    constexpr size_t dim = 8;
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
    bool cols_logical[dim] = {0};
    
    // expected bool array
    bool cols_logical_expected[dim] = {0};
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
    return 0;
};

