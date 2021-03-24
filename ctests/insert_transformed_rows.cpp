// Ctest 2021

// undefine NDEBUG macro to be able to perform asserts
#ifdef NDEBUG
#undef NDEBUG
#endif


#include <vector>
#include <random>
#include "constants_tests.h"


#include "tasks_apply_to_C_and_G/insert_transformed_rows.h"



/**
@brief Unit test case for the insert_transformed_rows
*/
int main() {
    printf("\n\n****************************************\n");
    printf("Test of insert_transformed_rows\n");
    printf("****************************************\n\n\n");

    // initialize random generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<double> distribution(0.0, 1.0);


    // allocate memory for the modes indices
    std::vector<size_t> modes_in = {0, 1, 3};
    constexpr size_t dim_corner = 3;

    assert(dim_corner == modes_in.size());
        
    // allocate matrix array for input
    constexpr size_t dim = 8;
    pic::matrix rows_in = pic::matrix(dim_corner, dim);
    
    // fill up matrix with random elements
    for (size_t row_idx = 0; row_idx < dim_corner; row_idx++) {
        for (size_t col_idx = 0; col_idx < dim; col_idx++) {
            double randnum1 = distribution(generator);
            double randnum2 = distribution(generator);
            rows_in[row_idx * dim + col_idx] = pic::Complex16(randnum1, randnum2);
        }
    }
    
    pic::matrix corner_in = pic::matrix(dim_corner, dim_corner);
    for (size_t row_idx = 0; row_idx < dim_corner; row_idx++) {
        for (size_t col_idx = 0; col_idx < dim_corner; col_idx++) {
            double randnum1 = distribution(generator);
            double randnum2 = distribution(generator);
            corner_in[row_idx * dim_corner + col_idx] = pic::Complex16(randnum1, randnum2);
        }
    }
    

    
    // allocate memory for the output matrix
    pic::matrix mtx_out = pic::matrix(dim, dim);
    
    // allocate memory for the expected results of the matrix
    pic::matrix mtx_out_expected = pic::matrix(dim, dim);
    
    // fill up the expected matrix with expected numbers from the matrix sent in
    for (size_t modes_idx = 0; modes_idx < dim_corner; modes_idx++) {
        for (size_t col_idx = 0; col_idx < dim; col_idx++) {
            mtx_out_expected[modes_in[modes_idx] * dim + col_idx] = rows_in[modes_idx * dim + col_idx];
        }
    }
    std::cout << "Matrix expected:" << std::endl;
    mtx_out_expected.print_matrix();
    for (size_t modes_idx_row = 0; modes_idx_row < dim_corner; modes_idx_row++) {
        for (size_t modes_idx_col = 0; modes_idx_col < dim_corner; modes_idx_col++) {
            mtx_out_expected[modes_in[modes_idx_row] * dim + modes_in[modes_idx_col]] = corner_in[modes_idx_row * dim_corner + modes_idx_col];
        }
    }

    // Create logical array containing information about the indices to insert in
    bool cols_logical_in[dim] = {0};
    for (size_t mode : modes_in){
        cols_logical_in[mode] = true;
    }
    
    // print the matrices on standard output
    std::cout << "Corner input:" << std::endl;
    corner_in.print_matrix();
    
    std::cout << "Rows input:" << std::endl;
    rows_in.print_matrix();
    
    std::cout << "Matrix expected:" << std::endl;
    mtx_out_expected.print_matrix();
    
    // Create instance of the Insert_Transformed_Rows class with the given input
    pic::Insert_Transformed_Rows insert_Transformed_Rows = pic::Insert_Transformed_Rows(rows_in, corner_in, mtx_out, modes_in, cols_logical_in);
    
    // tbb flow message for the method call
    tbb::flow::continue_msg msg = tbb::flow::continue_msg();

    // Calculation
    insert_Transformed_Rows(msg);
    
    // Intentionally make the test failing most likely
    //mtx_out[2] = pic::Complex16(0, 1);
    
    // print the result matrix
    std::cout << "Matrix resulted:" << std::endl;
    mtx_out.print_matrix();
    
    // Comparing expected and resulted matrix rows according to the given modes sent in
    for (size_t modes_idx = 0; modes_idx < dim_corner; modes_idx++) {
        for (size_t col_idx = 0; col_idx < dim; col_idx++){
            size_t elem_idx = modes_in[modes_idx] * dim + col_idx;
            pic::Complex16 diff = mtx_out_expected[elem_idx] - mtx_out[elem_idx];
            assert(std::abs(diff) < pic::epsilon);
            //std::cout << diff << std::endl;
         
            //assert(mtx_out[elem_idx] == mtx_out_expected[elem_idx]);
        }
    }
    std::cout << "Test passed. " << std::endl;

    return 0;
};

