#include "calc_characteristic_polynomial_coeffs_AVX.h"


namespace pic {

/**
@brief AVX kernel to determine the first \f$ k \f$ coefficients of the characteristic polynomial using the Algorithm 2 of LaBudde method.
 See [arXiv:1104.3769](https://arxiv.org/abs/1104.3769v1) for further details.
@param mtx matrix in upper Hessenberg form.
@param highest_order the order of the highest order coefficient to be calculated (k <= n)
@return Returns with the calculated coefficients of the characteristic polynomial.
 */
matrix
calc_characteristic_polynomial_coeffs_AVX(matrix &mtx, size_t highest_order) {

 // the matrix c holds not just the polynomial coefficients, but also auxiliary
 // data. To retrieve the characteristic polynomial coeffients from the matrix c, use
 // this map for characteristic polynomial coefficient c_j:
 // if j = 0, c_0 -> 1
 // if j > 0, c_j -> c[(n - 1) * n + j - 1]


    // check the dimensions of the matrix in debug mode
    assert( mtx.rows == mtx.cols);

    //dimension of the matrix
    size_t dim = mtx.rows;


    // allocate memory for the coefficients c_k of p(\lambda)
    matrix coeffs(dim, dim);
    memset(coeffs.get_data(), 0, dim*dim*sizeof(Complex16));
    double* mtx_data = (double*) mtx.get_data();
    double* coeffs_data = (double*) coeffs.get_data();


    // c^(1)_1 = -\alpha_1
    coeffs[0] = -mtx[0];

    // c^(2)_1 = c^(1)_1 - \alpha_2
    coeffs[dim] = coeffs[0] - mtx[dim+1];

    // c^(2)_2 = \alpha_1\alpha_2 - h_{12}\beta_2
    //coeffs[dim+1] =  mtx[0]*mtx[dim+1] - mtx[1]*mtx[dim];

    __m256d mtx_vec1 = _mm256_loadu_pd(mtx_data);
    __m256d mtx_vec2 = _mm256_loadu_pd(mtx_data+2*dim);
    __m256d mtx_vec3;
    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    mtx_vec3    = _mm256_insertf128_pd(mtx_vec3, _mm256_extractf128_pd(mtx_vec2, 1), 0);
    mtx_vec3    = _mm256_insertf128_pd(mtx_vec3, _mm256_castpd256_pd128(mtx_vec2), 1);

    __m256d vec3    = _mm256_mul_pd(mtx_vec1, mtx_vec3);
    mtx_vec3        = _mm256_permute_pd(mtx_vec3, 0x5);
    mtx_vec3        = _mm256_mul_pd(mtx_vec3, neg);
    __m256d vec4    = _mm256_mul_pd(mtx_vec1, mtx_vec3);
    vec3            = _mm256_hsub_pd(vec3, vec4);
    __m128d res_vec = _mm_sub_pd(_mm256_castpd256_pd128(vec3), _mm256_extractf128_pd(vec3, 1));
    _mm_storeu_pd(coeffs_data+2*(dim+1), res_vec);


//std::cout <<    coeffs[dim+1] << " "  << *((Complex16*)&res_vec[0]) << std::endl;





    // for (i=3:k do)
    for (size_t idx=2; idx<=highest_order-1; idx++) {
        // i = idx + 1

        // calculate the products of matrix elements \beta_i
        // the n-th (0<=n<=idx-2) element of the arary stands for:
        // beta_prods[n] = \beta_i * \beta_i-1 * ... * \beta_{i-n}
        matrix beta_prods(idx,1);
        beta_prods[0] = mtx[idx*dim + idx-1];
        for (size_t prod_idx=1; prod_idx<=idx-1; prod_idx++) {
            beta_prods[prod_idx] = beta_prods[prod_idx-1] * mtx[(idx-prod_idx)*dim + (idx-prod_idx-1)];
        }

        // c^(i)_1 = c^(i-1)_1 - \alpha_i
        coeffs[idx*dim] = coeffs[(idx-1)*dim] - mtx[idx*dim + idx];

        // for j=2 : i-1 do
        for (size_t jdx=1; jdx<=idx-1; jdx++) {
            // j = jdx + 1

            // sum = \sum_^{j-2}{m=1} h_{i-m,i} \beta_i*...*\beta_{i-m+1} c^{(i-m-1)}_{j-m-1}  - h_{i-j+1,i}* beta_i*...*beta_{i-j+2}
            Complex16 sum(0.0,0.0);

            // for m=j-2 : 1 do
            for ( size_t mdx=1; mdx<=jdx-1; mdx++) {
                // m = mdx

                // sum = sum + h_{i-m, i} * beta_prod * c^{(i-m-1)}_{j-m-1}
                sum = sum + mtx[(idx-mdx)*dim+idx] * beta_prods[mdx-1] * coeffs[(idx-mdx-1)*dim + jdx-mdx-1];

            }

            // sum = sum + h_{i-j+1,i} * \beta_prod
            sum = sum + mtx[(idx-jdx)*dim + idx] * beta_prods[jdx-1];

            // c^(i)_j = c^(i-1)_j - \alpha_i*c^(i-1)_{j-1} - sum
            coeffs[idx*dim+jdx] = coeffs[(idx-1)*dim+jdx] - mtx[idx*dim+idx] * coeffs[(idx-1)*dim + jdx-1] - sum;
        }

        // sum = \sum_^{i-2}{m=1} h_{i-m,i} \beta_i*...*\beta_{i-m+1} c^{(i-m-1)}_{i-m-1}  - h_{1,i}* beta_i*...*beta_{2}
        Complex16 sum(0.0,0.0);

        // for m=j-2 : 1 do
        for ( size_t mdx=1; mdx<=idx-1; mdx++) {
            // m = mdx

            // sum = sum + h_{i-m, i} * beta_prod * c^{(i-m-1)}_{j-m-1}
            sum = sum + mtx[(idx-mdx)*dim+idx] * beta_prods[mdx-1] * coeffs[(idx-mdx-1)*dim + idx-mdx-1];
        }

        // c^(i)_i = -\alpha_i c^{(i-1)}_{i-1} - sum
        coeffs[idx*dim+idx] = -mtx[idx*dim+idx]*coeffs[(idx-1)*dim+idx-1] - sum - mtx[idx]*beta_prods[beta_prods.size()-1];

    }

    // for i=k+1 : n do
    for (size_t idx = highest_order; idx<dim; idx++ ) {
        // i = idx + 1

        // c^(i)_1 = c^(i-1)_1 - \alpha_i
        coeffs[idx*dim] = coeffs[(idx-1)*dim] - mtx[idx*dim + idx];

        // calculate the products of matrix elements \beta_i
        // the n-th (0<=n<=idx-2) element of the arary stands for:
        // beta_prods[n] = \beta_i * \beta_i-1 * ... * \beta_{i-n}

        if (highest_order >= 2) {
            matrix beta_prods(idx,1);
            beta_prods[0] = mtx[idx*dim + idx-1];
            for (size_t prod_idx=1; prod_idx<=idx-1; prod_idx++) {
                beta_prods[prod_idx] = beta_prods[prod_idx-1] * mtx[(idx-prod_idx)*dim + (idx-prod_idx-1)];
            }

            // for j = 2 : k do
            for (size_t jdx=1; jdx<=highest_order-1; jdx++) {
                // j = jdx + 1

                // sum = \sum_^{j-2}{m=1} h_{i-m,i} \beta_i*...*\beta_{i-m+1} c^{(i-m-1)}_{j-m-1}  - h_{i-j+1,i}* beta_i*...*beta_{i-j+2}
                Complex16 sum(0.0,0.0);

                // for m=j-2 : 1 do
                for ( size_t mdx=1; mdx<=jdx-1; mdx++) {
                    // m = mdx

                    // sum = sum + h_{i-m, i} * beta_prod * c^{(i-m-1)}_{j-m-1}
                    sum = sum + mtx[(idx-mdx)*dim+idx] * beta_prods[mdx-1] * coeffs[(idx-mdx-1)*dim + jdx-mdx-1];

                }

                // sum = sum + h_{i-j+1,i} * \beta_prod
                sum = sum + mtx[(idx-jdx)*dim + idx] * beta_prods[jdx-1];

                // c^(i)_j = c^(i-1)_j - \alpha_i*c^(i-1)_{j-1} - sum
                coeffs[idx*dim+jdx] = coeffs[(idx-1)*dim+jdx] - mtx[idx*dim+idx] * coeffs[(idx-1)*dim + jdx-1] - sum;
            }



         }
    }


    return coeffs;
/*
// calculate the vector-matrix product (v^+) * A

    size_t sizeH = v.size();

    // pointers to two successive rows of matrix A
    double* data = (double*)A.get_data();
    double* data2 = data + 2*A.stride; // factor 2 is needed becouse of one complex16 consists of two doubles
    double* v_data = (double*)v.get_data();
    double* vH_times_A_data = (double*)vH_times_A.get_data();



    __m256d neg = _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);

    for (size_t row_idx = 0; row_idx < sizeH-1; row_idx=row_idx+2) {

        // extract two successive components v_i,v_{i+1} of vector v
        __m256d v_vec = _mm256_loadu_pd(v_data+2*row_idx);

        // create vector v_{i+1}, v_{i+1}
        __m128d* v_element = (__m128d*)&v_vec[2];
        __m256d v_vec2 = _mm256_broadcast_pd( v_element );

        // create vector v_i, v_i
        v_element = (__m128d*)&v_vec[0];
        __m256d v_vec1 = _mm256_broadcast_pd( v_element );

        for (size_t kdx = 0; kdx < A.cols-1; kdx = kdx + 2) {
            __m256d A_vec = _mm256_loadu_pd(data+2*kdx);
            __m256d A_vec2 = _mm256_loadu_pd(data2+2*kdx);

            // calculate the multiplications  A_vec*conj(v_vec1)

            // 1 Multiply elements of A_vec and v_vec1
            __m256d vec3 = _mm256_mul_pd(A_vec, v_vec1);

            // 2 Switch the real and imaginary elements of v_vec1
            __m256d v_vec_permuted = _mm256_permute_pd(v_vec1, 0x5);

            // 3 Negate the imaginary elements of cx_vec
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg);

            // 4 Multiply elements of A_vec and the modified v_vec
            __m256d vec4 = _mm256_mul_pd(A_vec, v_vec_permuted);

            // 5 Horizontally subtract the elements in vec3 and vec4
            A_vec  = _mm256_hadd_pd(vec3, vec4);



            // calculate the multiplications  A_vec2*conj(c_vec2)
            // 1 Multiply elements of AZ_vec and cx_vec
            vec3 = _mm256_mul_pd(A_vec2, v_vec2);

            // 2 Switch the real and imaginary elements of v_vec1
            v_vec_permuted = _mm256_permute_pd(v_vec2, 0x5);

            // 3 Negate the imaginary elements of cx_vec
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg);

            // 4 Multiply elements of A_vec and the modified v_vec
            vec4 = _mm256_mul_pd(A_vec2, v_vec_permuted);

            // 5 Horizontally subtract the elements in vec3 and vec4
            A_vec2  = _mm256_hadd_pd(vec3, vec4);

            // add A_vec and A_vec2
            A_vec  = _mm256_add_pd(A_vec, A_vec2);

            // add the result to _tmp
            __m256d _tmp = _mm256_loadu_pd(vH_times_A_data+2*kdx);
            _tmp = _mm256_add_pd(_tmp, A_vec);
            _mm256_storeu_pd(vH_times_A_data+2*kdx, _tmp);

        }



        if (A.cols % 2 == 1) {
            size_t kdx = A.cols-1;
            __m256d A_vec;
            A_vec = _mm256_insertf128_pd(A_vec, _mm_load_pd(data+2*kdx), 0);
            A_vec = _mm256_insertf128_pd(A_vec, _mm_load_pd(data2+2*kdx), 1);

            // calculate the multiplications  A_vec*conj(v_vec)

            // 1 Multiply elements of A_vec and v_vec
            __m256d vec3 = _mm256_mul_pd(A_vec, v_vec);

            // 2 Switch the real and imaginary elements of v_vec1
            __m256d v_vec_permuted = _mm256_permute_pd(v_vec, 0x5);

            // 3 Negate the imaginary elements of cx_vec
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg);

            // 4 Multiply elements of A_vec and the modified v_vec
            __m256d vec4 = _mm256_mul_pd(A_vec, v_vec_permuted);

            // 5 Horizontally subtract the elements in vec3 and vec4
            A_vec  = _mm256_hadd_pd(vec3, vec4);


            // sum elements of A_vec
            __m128d _tmp2 = _mm_add_pd(_mm256_castpd256_pd128(A_vec), _mm256_extractf128_pd(A_vec, 1));


            // add the result to vH_times_A
            __m128d _tmp = _mm_loadu_pd(vH_times_A_data+2*kdx);
            _tmp = _mm_add_pd(_tmp, _tmp2);
            _mm_storeu_pd(vH_times_A_data+2*kdx, _tmp);

        }

        // move data pointers to the next row pair
        data = data + 4*A.stride;
        data2 = data2 + 4*A.stride;


    }


    if (sizeH % 2 == 1) {

        size_t row_idx = sizeH-1;

        // extract the last component of vector v
        __m128d v_vec = _mm_loadu_pd(v_data+2*row_idx);

        // create vector v_i, v_i
        __m128d* v_element = (__m128d*)&v_vec[0];
        __m256d v_vec1 = _mm256_broadcast_pd( v_element );


        for (size_t kdx = 0; kdx < A.cols-1; kdx = kdx + 2) {
            __m256d A_vec = _mm256_loadu_pd(data+2*kdx);

            // calculate the multiplications  A_vec*conj(v_vec1)

            // 1 Multiply elements of A_vec and v_vec1
            __m256d vec3 = _mm256_mul_pd(A_vec, v_vec1);

            // 2 Switch the real and imaginary elements of v_vec1
            __m256d v_vec_permuted = _mm256_permute_pd(v_vec1, 0x5);

            // 3 Negate the imaginary elements of cx_vec
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg);

            // 4 Multiply elements of A_vec and the modified v_vec
            __m256d vec4 = _mm256_mul_pd(A_vec, v_vec_permuted);

            // 5 Horizontally subtract the elements in vec3 and vec4
            A_vec  = _mm256_hadd_pd(vec3, vec4);


            // add the result to _tmp
            __m256d _tmp = _mm256_loadu_pd(vH_times_A_data+2*kdx);
            _tmp = _mm256_add_pd(_tmp, A_vec);
            _mm256_storeu_pd(vH_times_A_data+2*kdx, _tmp);


        }


        if (A.cols % 2 == 1) {
            size_t kdx = A.cols-1;

            __m128d neg = _mm_setr_pd(-1.0, 1.0);
            __m128d A_vec = _mm_loadu_pd(data+2*kdx);

            // 1 Multiply elements of A_vec and v_vec1
            __m128d vec3 = _mm_mul_pd(A_vec, v_vec);

            // 2 Switch the real and imaginary elements of v_vec1
            __m128d v_vec_permuted = _mm_permute_pd(v_vec, 0x5);

            // 3 Negate the imaginary elements of cx_vec
            v_vec_permuted = _mm_mul_pd(v_vec_permuted, neg);

            // 4 Multiply elements of A_vec and the modified v_vec
            __m128d vec4 = _mm_mul_pd(A_vec, v_vec_permuted);

            // 5 Horizontally subtract the elements in vec3 and vec4
            A_vec  = _mm_hadd_pd(vec3, vec4);

            __m128d _tmp = _mm_loadu_pd(vH_times_A_data+2*kdx);
            _tmp = _mm_add_pd(_tmp, A_vec);
            _mm_storeu_pd(vH_times_A_data+2*kdx, _tmp);

        }



    }



*/

    return;

}


} // PIC
