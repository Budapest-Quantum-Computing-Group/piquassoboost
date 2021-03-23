// Ctest 2021

// undefine NDEBUG macro to be able to perform asserts
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <vector>
#include <random>
#include <algorithm>
#include "constants_tests.h"


#include "tasks_apply_to_C_and_G/insert_transformed_cols.h"



/**
@brief Unit test case for the insert_transformed_cols
*/
int main() {
    printf("\n\n****************************************\n");
    printf("Test of class Insert_Transformed_Cols\n");
    printf("****************************************\n\n\n");

    // initialize random generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<double> distribution(0.0, 1.0);


    // allocate memory for the modes indices
    std::vector<size_t> modes_in = {2, 3, 7};
    
    // allocate matrix array for input
    constexpr size_t dim = 8;
    pic::matrix rows_in = pic::matrix(3, dim);
    
    constexpr size_t dim_corner = 3;

    assert(dim_corner == modes_in.size());
    
    // fill up matrix with random elements
    for (size_t row_idx = 0; row_idx < modes_in.size(); row_idx++) {
        for (size_t col_idx = 0; col_idx < dim; col_idx++) {
            double randnum1 = distribution(generator);
            double randnum2 = distribution(generator);
            rows_in[row_idx * dim + col_idx] = pic::Complex16(randnum1, randnum2);
        }
    }
    

    
    // allocate memory for the output matrix
    pic::matrix mtx_out = pic::matrix(dim, dim);
    
    // allocate memory for the expected results of the matrix
    pic::matrix mtx_out_expected = pic::matrix(dim, dim);
    
    // fill up the expected matrix with expected numbers from the matrix sent in
    for (size_t row_idx = 0; row_idx < dim; row_idx++) {
        for (size_t modes_idx = 0; modes_idx < modes_in.size(); modes_idx++) {
            if (std::count(modes_in.begin(), modes_in.end(), row_idx)){
                // intentionally left empty
            }else{
                // insert transposed the line to the given columns
                mtx_out_expected[row_idx * dim + modes_in[modes_idx]] = rows_in[modes_idx * dim + row_idx];
            }
        }
    }

    // fill up the logical arrays about which column was modified
    bool cols_logical_in[dim] = {0};
    for (size_t mode : modes_in){
        cols_logical_in[mode] = true;
    }
    bool conjugate_elements_in = false;
    
    // print the matrices on standard output
    std::cout << "Rows input:" << std::endl;
    rows_in.print_matrix();
    
    std::cout << "Matrix expected:" << std::endl;
    mtx_out_expected.print_matrix();
    
    // Create instance of the Insert_Transformed_Cols class with the given input
    pic::Insert_Transformed_Cols insert_Transformed_Cols = pic::Insert_Transformed_Cols(rows_in, mtx_out, modes_in, cols_logical_in, conjugate_elements_in);
   
    // tbb flow message for the method call
    tbb::flow::continue_msg msg = tbb::flow::continue_msg();

    // Calculation
    insert_Transformed_Cols(msg);
    
    // Intentionally make the test failing most likely
    //mtx_out[2] = pic::Complex16(0, 1);
    
    // print the result matrix
    std::cout << "Matrix resulted:" << std::endl;
    mtx_out.print_matrix();
    
    // Comparing expected and resulted matrix elements
    for (size_t elem_idx = 0; elem_idx < modes_in.size()*dim; elem_idx++) {
        pic::Complex16 diff = mtx_out[elem_idx] - mtx_out_expected[elem_idx];
        assert(std::abs(diff) < pic::epsilon);
        
        //assert(rows_out[elem_idx] == rows_out_expected[elem_idx]);
    }
    
    std::cout << "Test passed. " << std::endl;

    return 0;
};

