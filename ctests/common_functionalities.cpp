 // Ctest 2021

// undefine NDEBUG macro to be able to perform asserts
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <random>
#include "constants_tests.h"


#include "common_functionalities.h"


int mainIsSymmetric();
int testCaseIsSymmetric(size_t dim, bool testOfSymmetric);

int mainIsHermitian();
int testCaseIsHermitian(size_t dim, bool testOfHermitian);


int main(){

    /**
    @brief Unit test cases for the function checking symmetric property of a matrix
    */
    mainIsSymmetric();
    
    
    /**
    @brief Unit test cases for the function checking hermitian property of a matrix
    */
    mainIsHermitian();
    
    return 0;
}



/**
@brief Unit test cases for the function checking symmetric property of a matrix
*/
int mainIsSymmetric() {
    printf("\n\n****************************************\n");
    printf("Test cases of the method isSymmetric\n");
    printf("****************************************\n\n\n");

    // Test case for dimension 1, always symmetric case
    testCaseIsSymmetric(1, true);
    
    // Test case for dimension 3, asymmetric case
    testCaseIsSymmetric(3, false);

    // Test case for dimension 3, symmetric case
    testCaseIsSymmetric(3, true);

    // Test case for dimension 12, asymmetric case
    testCaseIsSymmetric(12, false);

    // Test case for dimension 12, symmetric case
    testCaseIsSymmetric(12, true);
    
    std::cout << "All test cases passed. " << std::endl;
    return 0;
};


// @param dim Dimnesion of the matrix.
// @param testOfSymmetric Whether we are in symmetric case or not.
int testCaseIsSymmetric(size_t dim, bool testOfSymmetric){
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


    // If we are in asymmetric case, break the symmetric property
    if (!testOfSymmetric){
        // Intentionally break the symmatrization
        mtx_in[1] = pic::Complex16(0, 1);
        mtx_in[dim] = pic::Complex16(1, 0);
    }

    // print the input matrix on standard output
    //std::cout << "Matrix input:" << std::endl;
    //mtx_in.print_matrix();
    
    // Calculating and storing the result
    const bool result = pic::isSymmetric(mtx_in, pic::epsilon);
    
    // Evaluation of the result
    if (testOfSymmetric){
        assert(result == true);
    }else{
        assert(result == false);
    }
}


/**
@brief Unit test cases for the function checking Hermitian property of a matrix
*/
int mainIsHermitian() {
    printf("\n\n****************************************\n");
    printf("Test cases of the method isHermitian\n");
    printf("****************************************\n\n\n");

    // Test case for dimension 1, hermitian case
    testCaseIsHermitian(1, false);
    
    // Test case for dimension 1, nonhermitian case
    testCaseIsHermitian(1, true);
    
    // Test case for dimension 3, nonhermitian case
    testCaseIsHermitian(3, true);

    // Test case for dimension 3, hermitian case
    testCaseIsHermitian(3, false);

    // Test case for dimension 12, nonhermitian case
    testCaseIsHermitian(12, true);

    // Test case for dimension 12, hermitian case
    testCaseIsHermitian(12, false);
    
    std::cout << "All test cases passed. " << std::endl;
    return 0;
};


// @param dim Dimnesion of the matrix.
// @param testOfHermitian Whether we are in hermitian matrix generation case or not.
int testCaseIsHermitian(size_t dim, bool testOfHermitian){
    // initialize random generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<double> distribution(0.0, 1.0);


    // allocate matrix array for input
    pic::matrix mtx_in = pic::matrix(dim, dim);
    
    // fill up matrix with random elements as a hermitian matrix
    for (size_t row_idx = 0; row_idx < dim; row_idx++) {
        for (size_t col_idx = row_idx; col_idx < dim; col_idx++) {
            if (row_idx != col_idx){
                // any complex number in the offdiagonal and their conjugate in the opposite side
                double randnum1 = distribution(generator);
                double randnum2 = distribution(generator);
                mtx_in[row_idx * dim + col_idx] = pic::Complex16(randnum1, randnum2);
                mtx_in[col_idx * dim + row_idx] = pic::Complex16(randnum1, -randnum2);
            }else{
                // real numbers in diagonal
                double randnum = distribution(generator);            
                mtx_in[col_idx * dim + row_idx] = pic::Complex16(randnum, 0);
            }
        }
    }


    // If we are in asymmetric case, break the symmetric property
    if (!testOfHermitian){
        // Intentionally break the hermitian property
        if (dim == 1){
            mtx_in[0] = pic::Complex16(0, 1);
        }else{
            mtx_in[dim] = pic::Complex16(1, 0);
            mtx_in[1] = pic::Complex16(0, 1);
        }
    }

    // print the input matrix on standard output
    //std::cout << "Matrix input:" << std::endl;
    //mtx_in.print_matrix();
    
    // Calculating and storing the result
    const bool result = pic::isHermitian(mtx_in, pic::epsilon);
    
    // Print the result
    //std::cout << "Result: " << result << std::endl;
    
    // Evaluation of the result
    if (testOfHermitian){
        assert(result == true);
    }else{
        assert(result == false);
    }
}


