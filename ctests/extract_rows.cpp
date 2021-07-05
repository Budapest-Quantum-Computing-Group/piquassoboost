// Ctest 2021

// undefine NDEBUG macro to be able to perform asserts
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <vector>
#include <random>
#include "constants_tests.h"


#include "tasks_apply_to_C_and_G/extract_rows.h"



/**
@brief Unit test case for the row extractation method
*/
int main() {
    printf("\n\n****************************************\n");
    printf("Test of row extractation method\n");
    printf("****************************************\n\n\n");

    // initialize random generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<double> distribution(0.0, 1.0);


    // allocate matrix array for input
    constexpr size_t dim = 6;
    pic::matrix mtx_in = pic::matrix(dim, dim);
    // fill up matrix with random elements
    for (size_t row_idx = 0; row_idx < dim; row_idx++) {
        for (size_t col_idx = 0; col_idx < dim; col_idx++) {
            double randnum1 = distribution(generator);
            double randnum2 = distribution(generator);
            mtx_in[row_idx * dim + col_idx] = pic::Complex16(randnum1, randnum2);
        }
    }
    
    // allocate memory for the modes indices
    std::vector<size_t> modes_in = {0, 1, 3};

    // allocate memory for the output matrix
    pic::matrix rows_out = pic::matrix(modes_in.size(), dim);
    
    // allocate memory for the expected results of the matrix
    pic::matrix rows_out_expected = pic::matrix(modes_in.size(), dim);
    
    // fill up the expected matrix with expected numbers from the matrix sent in
    for (size_t modes_idx = 0; modes_idx < modes_in.size(); modes_idx++) {
        for (size_t col_idx = 0; col_idx < dim; col_idx++) {
            rows_out_expected[modes_idx * rows_out_expected.stride + col_idx] = mtx_in[modes_in[modes_idx] * mtx_in.stride + col_idx];
        }
    }

    // print the matrices on standard output
    std::cout << "Matrix input:" << std::endl;
    mtx_in.print_matrix();
    std::cout << "Matrix expected:" << std::endl;
    rows_out_expected.print_matrix();
    
    // Create instance of the Extract_Rows class with the given input
    pic::Extract_Rows extract_rows = pic::Extract_Rows(mtx_in, rows_out, modes_in);

    // tbb flow message for the method call
    tbb::flow::continue_msg msg = tbb::flow::continue_msg();

    // Calculation
    extract_rows(msg);
    
    // Intentionally make the test failing most likely
    //rows_out[2] = pic::Complex16(0, 1);
    
    // print the result matrix
    std::cout << "Matrix resulted:" << std::endl;
    rows_out.print_matrix();
    
    // Comparing expected and resulted matrix elements
    for (size_t elem_idx = 0; elem_idx < modes_in.size()*dim; elem_idx++) {
        pic::Complex16 diff = rows_out[elem_idx] - rows_out_expected[elem_idx];
        assert(std::abs(diff) < pic::epsilon);
    }
    std::cout << "Test passed. " << std::endl;

    return 0;
};

