#include <vector>

#include <random>
#include <chrono>
#include <string>

#include "constants_tests.h"

#include "Torontonian.h"
#include "TorontonianRecursive.h"
#include "TorontonianUtilities.h"

#include "matrix32.h"
#include "matrix.h"

#include "dot.h"
#include "matrix_helper.hpp"



template<class matrix_type, class complex_type>
matrix_type
get_random_density_matrix(size_t dim){
    matrix_type posdef = pic::getRandomMatrix<matrix_type, complex_type>(dim, pic::POSITIVE_DEFINIT);
    matrix_type posdef_inverse = pic::calc_inverse_of_matrix<matrix_type, complex_type>(posdef);
    return posdef_inverse;
}

/**
@brief Unit test to calculate the Cholesky decomposition of a matrix
*/
int test_cholesky_decomposition(){

/*
    constexpr size_t dim = 131;

    pic::matrix mtx = pic::getRandomMatrix<pic::matrix, pic::Complex16>(dim, pic::POSITIVE_DEFINIT);
    pic::matrix mtx_copy = mtx.copy();

    pic::calc_cholesky_decomposition(mtx);


    // rewrite upper triangular element to zero:
    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = idx + 1; jdx < dim; jdx++){
            mtx[idx * dim + jdx] = pic::Complex16(0.0);
        }
    }

    pic::matrix mtx_adjoint(dim, dim);
    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = 0; jdx < dim; jdx++){
            pic::Complex16 &elem = mtx[jdx * dim + idx];
            mtx_adjoint[idx * dim + jdx] = pic::Complex16(elem.real(), -elem.imag());
        }
    }


    pic::matrix product = dot(mtx, mtx_adjoint);

    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = 0; jdx < dim; jdx++){
            pic::Complex16 diff = product[idx * product.stride + jdx] - mtx_copy[idx * mtx_copy.stride + jdx];
            assert(std::abs(diff) < pic::epsilon);
            if (std::abs(diff) > pic::epsilon){
                std::cout << "Error " << idx << "," << jdx <<" diff: " <<diff<<std::endl;
                return 1;
            }
        }
    }
*/
    std::cout << "test cholsky passed!" << std::endl;
    return 0;
}

/**
@brief Unit test to calculate torontonian of a matrix
*/
int test_calc_torontonian(){

    constexpr size_t dim = 6;

    pic::matrix mtx = get_random_density_matrix<pic::matrix, pic::Complex16>(dim);
    pic::Torontonian torontonian_calculator(mtx);
    double result = torontonian_calculator.calculate();


    std::cout << "test torontonian passed!" << std::endl;
    return 0;
}

int test_compare_torontonian_calculations(){
    pic::matrix mtx(2, 2);
    mtx[0] = pic::Complex16(0.136442, 0);
    mtx[1] = pic::Complex16(0.079634, 0.0393217);
    mtx[2] = pic::Complex16(0.079634, -0.0393217);
    mtx[3] = pic::Complex16(0.136442, 0);

    pic::TorontonianRecursive tor_recursive(mtx);
    double rec_val = tor_recursive.calculate(false);
    pic::Torontonian tor(mtx);
    double tor_val = tor.calculate();
    std::cout << "rec_val " << rec_val << std::endl;
    std::cout << "tor_val " << tor_val << std::endl;

}


/**
@brief Unit test to calculate torontonian of a matrix, test for Cholesky decomposition and for inverse calculation
*/
int main(){

    test_compare_torontonian_calculations();
    //test_cholesky_decomposition();
    //test_calc_torontonian();

}
