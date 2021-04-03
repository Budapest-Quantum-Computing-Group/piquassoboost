 // Ctest 2021

// undefine NDEBUG macro to be able to perform asserts
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <random>
#include "constants_tests.h"


#include "SymmetrizationChecker.h"

int test(size_t dim, bool testOfAsymmetric);

/**
@brief Unit test case for the class representing symmetrization check of a given input matrix
*/
int main() {
    printf("\n\n****************************************\n");
    printf("Test cases of the class SymmetrizationChecker\n");
    printf("****************************************\n\n\n");

    // Test case for dimension 1, always symmetric case
    test(1, false);
    
    // Test case for dimension 3, asymmetric case
    test(3, true);

    // Test case for dimension 3, symmetric case
    test(3, false);

    // Test case for dimension 12, asymmetric case
    test(12, true);

    // Test case for dimension 12, symmetric case
    test(12, false);
    
    std::cout << "All test cases passed. " << std::endl;
    return 0;
};


// @param dim Dimnesion of the matrix.
// @param testOfAsymmetric Whether we are in asymmetric case or not.
int test(size_t dim, bool testOfAsymmetric){
    // initialize random generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<double> distribution(0.0, 1.0);


    // allocate matrix array for input
    pic::matrix mtx_in = pic::matrix(dim, dim);
    
    // fill up matrix with random elements as a symmetrized matrix
    for (size_t row_idx = 0; row_idx < dim; row_idx++) {
        for (size_t col_idx = row_idx; col_idx < dim; col_idx++) {
            double randnum1 = distribution(generator);
            double randnum2 = distribution(generator);
            mtx_in[row_idx * dim + col_idx] = pic::Complex16(randnum1, randnum2);
            if (row_idx != col_idx){
                mtx_in[col_idx * dim + row_idx] = pic::Complex16(randnum1, randnum2);
            }
        }
    }

    // allocate memory for the output value, initialized as an invalid value
    int output = 12;

    // print the input matrix on standard output
    //std::cout << "Matrix input:" << std::endl;
    //mtx_in.print_matrix();
    
    
    // Create instance of the SymmetrizationChecker class with the given input
    pic::SymmetrizationChecker checker = pic::SymmetrizationChecker(mtx_in, &output);

    // Test case for the symmetric case:
    // tbb flow message for the method call
    tbb::flow::continue_msg msg = tbb::flow::continue_msg();

    // If we are in asymmetric case, break the symmetric property
    if (testOfAsymmetric){
        // Intentionally break the symmatrization
        mtx_in[1] = pic::Complex16(0, 1);
        mtx_in[dim] = pic::Complex16(1, 0);
    }

    // Calculation
    checker(msg);
    
    // Evaluation of the result
    if (testOfAsymmetric){
        assert(output == 0);
        //std::cout << "Matrix is not symmetric!" << std::endl;
    }else{
        assert(output == 1);
        //std::cout << "Matrix is symmetric!" << std::endl;
    }
}

