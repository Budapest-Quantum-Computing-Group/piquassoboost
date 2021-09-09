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
