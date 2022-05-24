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

#ifndef MATRIX_HELPER_INCLUDED
#define MATRIX_HELPER_INCLUDED

#include "constants_tests.h"
#include <random>

namespace pic{



// Calculating inverse of lower triangular matrix
// L * M = Id
// sum(k=0,n) l(i,k)*m(k,j) = id(i,j)
// m(0,0) = 1 / l(0,0)
// l(1,0) * m(0,0) + l(1,1) * m(1,0) = 0 => m(1,0) = - l(1,0) * m(0,0) / l(1,1)
// l(2,0) * m(0,0) + l(2,1) * m(1,0) + l(2,2) * m(2,0) = 0 => m(2,0) = - l(1,0) * m(0,0) / l(1,1)
// m(i,j) = ( -sum(k=j,i-1) l(i,k) * m(k,j) ) / l(j,j)
template<class matrix_type, class complex_type>
matrix_type
calc_inverse_of_lower_triangular_matrix(matrix_type &mtx){
    // assuming mtx.rows == mtx.cols
    size_t dim = mtx.rows;

    // columns and rows are changed into each other
    matrix_type inverse(dim, dim);
    for (size_t row_idx = 0; row_idx < dim; row_idx++){
        for (size_t col_idx = 0; col_idx < row_idx; col_idx++){
            complex_type sum = 0.0;
            for (size_t k = col_idx; k < row_idx; k++){
                sum -= mtx[row_idx*mtx.stride + k] * inverse[k*inverse.stride + col_idx];
            }
            inverse[row_idx*inverse.stride + col_idx] = sum / mtx[row_idx*mtx.stride + row_idx];
        }
        inverse[row_idx * inverse.stride + row_idx] = 1.0 / mtx[row_idx * inverse.stride + row_idx];
        for (size_t col_idx = row_idx + 1; col_idx < dim; col_idx++){
            inverse[row_idx * inverse.stride + col_idx] = 0;
        }
    }
    return inverse;
}


// Calculating inverse of lower triangular matrix
// M * U = Id
// m(i,j) = ( -sum(k=j,i-1) m(i,k)*u(k,j) ) / u(j,j)
template<class matrix_type, class complex_type>
matrix_type
calc_inverse_of_upper_triangular_matrix(matrix_type &mtx){
    // assuming mtx.rows == mtx.cols
    size_t dim = mtx.rows;

    // columns and rows are changed into each other
    matrix_type inverse(dim, dim);
    for (size_t row_idx = 0; row_idx < dim; row_idx++){
        for (size_t col_idx = 0; col_idx < row_idx; col_idx++){
            inverse[row_idx * inverse.stride + col_idx] = 0;
        }
        inverse[row_idx * inverse.stride + row_idx] = 1.0 / mtx[row_idx * inverse.stride + row_idx];
        for (size_t col_idx = row_idx + 1; col_idx < dim; col_idx++){
            complex_type sum = 0.0;
            for (size_t k = row_idx; k < col_idx; k++){
                sum -= inverse[row_idx*inverse.stride + k] * mtx[k*mtx.stride + col_idx];
            }
            inverse[row_idx*inverse.stride + col_idx] = sum / mtx[col_idx*mtx.stride + col_idx];
        }
    }
    return inverse;
}


template<class matrix_type, class complex_type>
matrix_type
calc_inverse_of_matrix(matrix_type &mtx){
    if (mtx.rows != mtx.cols){
        exit(1);
    }
    size_t dim = mtx.rows;
    matrix_type lower(dim, dim);
    matrix_type upper(dim, dim);

    for (size_t i = 0; i < dim; i++){
        for (size_t j = i; j < dim; j++){
            // Summ l(i, k) * u(k, j)
            complex_type sum = 0.0;
            for (size_t k = 0; k < i; k++)
                sum += (lower[i*lower.stride+k] * upper[k*upper.stride+j]);

            // u(i, j)
            upper[i*upper.stride+j] = mtx[i*mtx.stride+j] - sum;
        }

        for (size_t j = i; j < dim; j++){
            if (i == j){
                lower[i*lower.stride+i] = 1; // Diagonal as 1
            }else{
                // Sum l(j, k) * u(k, i)
                complex_type sum = 0.0;
                for (size_t k = 0; k < i; k++){
                    sum += (lower[j*lower.stride+k] * upper[k*upper.stride+i]);
                }
                // l(j, i)
                lower[j*lower.stride+i]= (mtx[j*mtx.stride+i] - sum) / upper[i*upper.stride+i];
            }
        }
    }
    
    matrix_type lower_inverse = calc_inverse_of_lower_triangular_matrix<matrix_type, complex_type>(lower);
    matrix_type upper_inverse = calc_inverse_of_upper_triangular_matrix<matrix_type, complex_type>(upper);

    matrix_type product(mtx.rows, mtx.cols);
    for (size_t i = 0; i < product.rows; i++)
        for (size_t j = 0; j < product.cols; j++){
            complex_type &value = product[i * product.stride + j];
            for (size_t k = 0; k < product.cols; k++){
                value += upper_inverse[i * upper_inverse.stride + k] * lower_inverse[k * lower_inverse.stride + j];
            }
        }

    return product;
}


template<class matrix_type, class complex_type>
matrix_type
get_random_matrix(matrix_type &mtx){
    if (mtx.rows != mtx.cols){
        exit(1);
    }
    size_t dim = mtx.rows;
    matrix_type lower(dim, dim);
    matrix_type upper(dim, dim);

    for (size_t i = 0; i < dim; i++){
        for (size_t j = i; j < dim; j++){
            // Summ l(i, k) * u(k, j)
            complex_type sum = 0.0;
            for (size_t k = 0; k < i; k++)
                sum += (lower[i*lower.stride+k] * upper[k*upper.stride+j]);

            // u(i, j)
            upper[i*upper.stride+j] = mtx[i*mtx.stride+j] - sum;
        }

        for (size_t j = i; j < dim; j++){
            if (i == j){
                lower[i*lower.stride+i] = 1; // Diagonal as 1
            }else{
                // Sum l(j, k) * u(k, i)
                complex_type sum = 0.0;
                for (size_t k = 0; k < i; k++){
                    sum += (lower[j*lower.stride+k] * upper[k*upper.stride+i]);
                }
                // l(j, i)
                lower[j*lower.stride+i]= (mtx[j*mtx.stride+i] - sum) / upper[i*upper.stride+i];
            }
        }
    }
    
    matrix_type lower_inverse = calc_inverse_of_lower_triangular_matrix<matrix_type, complex_type>(lower);
    matrix_type upper_inverse = calc_inverse_of_upper_triangular_matrix<matrix_type, complex_type>(upper);

    matrix_type product = dot(upper_inverse, lower_inverse);
    return product;
}


// returns a random matrix of the given type:
// type is specified at definition of RandomMatrixType
template<class matrix_type, class complex_type>
matrix_type 
getRandomComplexMatrix(size_t n, pic::RandomMatrixType type){
    matrix_type mtx(n, n);

    // initialize random generator as a standard normal distribution generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<long double> distribution(0.0, 1.0);

    if (type == pic::RANDOM){
        // fill up matrix with fully random elements
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = 0; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                long double randnum2 = distribution(generator);
                mtx[row_idx * n + col_idx] = complex_type(randnum1, randnum2);
            }
        }
    }else if (type == pic::SYMMETRIC){
        // fill up matrix with fully random elements symmetrically
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                long double randnum2 = distribution(generator);
                mtx[row_idx * n + col_idx] = complex_type(randnum1, randnum2);
                mtx[col_idx * n + row_idx] = complex_type(randnum1, randnum2);
            }
        }
    }else if (type == pic::SELFADJOINT){
        // hermitian case, selfadjoint matrix
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                if (row_idx == col_idx){
                    mtx[row_idx * n + col_idx] = complex_type(randnum1, 0);
                }else{
                    long double randnum2 = distribution(generator);
                    mtx[row_idx * n + col_idx] = complex_type(randnum1, randnum2);
                    mtx[col_idx * n + row_idx] = complex_type(randnum1, -randnum2);
                }
            }
        }
    }else if (type == pic::POSITIVE_DEFINIT){
        // hermitian case, selfadjoint, positive definite matrix
        // if you have a random matrix M then M * M^* gives you a positive definite hermitian matrix
        matrix_type mtx1 = getRandomComplexMatrix<matrix_type, complex_type>(n, pic::RANDOM);
        matrix_type mtx2 = matrix_conjugate_traspose<matrix_type>(mtx1);

        mtx = pic::dot(mtx1, mtx2);

        for (size_t i = 0; i < n; i++){
            for (size_t j = 0; j < n; j++){
                mtx[i * n + j] /= (double)n;
            }    
            mtx[i * n + i] += complex_type(1.0,0.0);
        }
    }else if (type == pic::LOWER_TRIANGULAR){
        // fill up matrix as lower triangular with fully random elements
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = 0; col_idx < row_idx + 1; col_idx++) {
                long double randnum1 = distribution(generator);
                long double randnum2 = distribution(generator);
                mtx[row_idx * n + col_idx] = complex_type(randnum1, randnum2);
            }
            for (size_t col_idx = row_idx + 1; col_idx < n; col_idx++) {
                mtx[row_idx * n + col_idx] = 0.0;
            }
        }
    }else if (type == pic::UPPER_TRIANGULAR){
        // fill up matrix as upper triangular with fully random elements
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = 0; col_idx < row_idx; col_idx++) {
                mtx[row_idx * n + col_idx] = 0.0;
            }
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                long double randnum2 = distribution(generator);
                mtx[row_idx * n + col_idx] = complex_type(randnum1, randnum2);
            }
        }
    }
    return mtx;
}



// returns a random real matrix of the given type:
// type is specified at definition of RandomMatrixType
template<class matrix_type, class scalar_type>
matrix_type 
getRandomRealMatrix(size_t n, pic::RandomMatrixType type){
    matrix_type mtx(n, n);

    // initialize random generator as a standard normal distribution generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<long double> distribution(0.0, 1.0);

    if (type == pic::RANDOM){
        // fill up matrix with fully random elements
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = 0; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                mtx[row_idx * n + col_idx] = scalar_type(randnum1);
            }
        }
    }else if (type == pic::SYMMETRIC){
        // fill up matrix with fully random elements symmetrically
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                mtx[row_idx * n + col_idx] = scalar_type(randnum1);
                mtx[col_idx * n + row_idx] = scalar_type(randnum1);
            }
        }
    }else if (type == pic::SELFADJOINT){
        // selfadjoint == symmetric in case of real matrices
        // hermitian case, selfadjoint matrix
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                if (row_idx == col_idx){
                    mtx[row_idx * n + col_idx] = scalar_type(randnum1);
                }else{
                    mtx[row_idx * n + col_idx] = scalar_type(randnum1);
                    mtx[col_idx * n + row_idx] = scalar_type(randnum1);
                }
            }
        }
    }else if (type == pic::POSITIVE_DEFINIT){
        // hermitian case, selfadjoint, positive definite matrix
        // if you have a random matrix M then M * M^T gives you a positive definite symmetric matrix
        matrix_type mtx1 = getRandomRealMatrix<matrix_type, scalar_type>(n, pic::RANDOM);

        mtx = matrix_type(n, n);
        // calculate mtx1 * mtx1^T
        for (size_t i = 0; i < n; i++){
            for (size_t j = 0; j < n; j++){
                scalar_type &value = mtx[i * mtx.stride + j];
                value = 0;
                for (size_t k = 0; k < n; k++){
                    value += mtx1[i * mtx1.stride + k] * mtx1[j * mtx1.stride + k];
                }
            }
        }

        for (size_t i = 0; i < n; i++){
            for (size_t j = 0; j < n; j++){
                mtx[i * n + j] /= n;
            }    
            mtx[i * n + i] += scalar_type(1.0);
        }
    }else if (type == pic::LOWER_TRIANGULAR){
        // fill up matrix as lower triangular with fully random elements
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = 0; col_idx < row_idx + 1; col_idx++) {
                long double randnum1 = distribution(generator);
                mtx[row_idx * n + col_idx] = scalar_type(randnum1);
            }
            for (size_t col_idx = row_idx + 1; col_idx < n; col_idx++) {
                mtx[row_idx * n + col_idx] = 0.0;
            }
        }
    }else if (type == pic::UPPER_TRIANGULAR){
        // fill up matrix as upper triangular with fully random elements
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = 0; col_idx < row_idx; col_idx++) {
                mtx[row_idx * n + col_idx] = 0.0;
            }
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                mtx[row_idx * n + col_idx] = scalar_type(randnum1);
            }
        }
    }
    return mtx;
}




template<class matrix_type, class complex_type>
matrix_type
get_random_density_matrix_complex(size_t dim){
    matrix_type posdef = getRandomComplexMatrix<matrix_type, complex_type>(dim, pic::POSITIVE_DEFINIT);
    matrix_type posdef_inverse = pic::calc_inverse_of_matrix<matrix_type, complex_type>(posdef);
    return posdef_inverse;
}

template<class matrix_type, class scalar_type>
matrix_type
get_random_density_matrix_real(size_t dim){
    matrix_type posdef = getRandomRealMatrix<matrix_type, scalar_type>(dim, pic::POSITIVE_DEFINIT);
    matrix_type posdef_inverse = pic::calc_inverse_of_matrix<matrix_type, scalar_type>(posdef);
    return posdef_inverse;
}

} // PIC



#endif // MATRIX_HELPER_INCLUDED
