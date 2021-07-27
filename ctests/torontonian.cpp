#include "Torontonian.h"
#include "matrix.h"
#include "matrix_helper.hpp"



/**
@brief Unit test to calculate torontonian of a matrix
*/
void test_calc_torontonian(){

    constexpr size_t dim = 6;

    pic::matrix matrix = pic::get_random_density_matrix_complex<pic::matrix, pic::Complex16>(dim);
    pic::Torontonian torontonian_calculator(matrix);
    torontonian_calculator.calculate();
}



/**
@brief Unit test to calculate torontonian of a matrix, test for Cholesky decomposition and for inverse calculation
*/
int main(){

    test_calc_torontonian();

}
