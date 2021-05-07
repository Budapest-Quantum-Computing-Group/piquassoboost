#include "apply_householder_cols_AVX.h"

/*
#include "tbb/tbb.h"
static tbb::spin_mutex my_mutex;
*/

namespace pic {

/**
@brief AVX kernel to apply householder transformation on a matrix A' = (1 - 2*v o v/v^2) A for one specific reflection vector v
@param A matrix on which the householder transformation is applied. (The output is returned via this matrix)
@param v A matrix instance of the reflection vector
*/
void
apply_householder_cols_AVX(matrix &A, matrix &v) {


    size_t sizeH = v.size();

// calculate A^~(1-2vov)

    // pointers to two successive rows of matrix A
    double* data = (double*)A.get_data();
    double* data2 = data + 2*A.stride; // factor 2 is needed becouse of one complex16 consists of two doubles
    double* v_data = (double*)v.get_data();



    __m512d neg = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    __m256d neg2 = _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);

    for (size_t row_idx = 0; row_idx < A.rows-1; row_idx=row_idx+2) {


        __m256d factor_vec = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
        __m256d factor_vec2 = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);

        if ( A.cols > 3 ) {


            for (size_t kdx = 0; kdx < A.cols-3; kdx = kdx + 4) {

                __m512d A_vec  = _mm512_loadu_pd(data+2*kdx);
                __m512d A_vec2 = _mm512_loadu_pd(data2+2*kdx);

                // extract two successive components v_i,v_{i+1} of vector v
                __m512d v_vec = _mm512_loadu_pd(v_data+2*kdx);
 
                // calculate the multiplications  A_vec*v_vec

                __m512d vec3           = _mm512_mul_pd(A_vec, v_vec);
                __m512d v_vec_permuted = _mm512_permute_pd(v_vec, 0x55);
                v_vec_permuted         = _mm512_mul_pd(v_vec_permuted, neg);
                __m512d vec4           = _mm512_mul_pd(A_vec, v_vec_permuted);
                __m256d vec3_256       = _mm512_castpd512_pd256(vec3);
                __m256d vec4_256       = _mm512_castpd512_pd256(vec4);    
                vec3_256               = _mm256_hsub_pd(vec3_256, vec4_256);
                factor_vec             = _mm256_add_pd(factor_vec, vec3_256);
                vec3_256               = _mm512_extractf64x4_pd(vec3, 1);
                vec4_256               = _mm512_extractf64x4_pd(vec4, 1);
                vec3_256               = _mm256_hsub_pd(vec3_256, vec4_256);
                factor_vec             = _mm256_add_pd(factor_vec, vec3_256);



                // calculate the multiplications  A_vec2*v_vec

                vec3        = _mm512_mul_pd(A_vec2, v_vec);
                vec4        = _mm512_mul_pd(A_vec2, v_vec_permuted); 
                vec3_256    = _mm512_castpd512_pd256(vec3);
                vec4_256    = _mm512_castpd512_pd256(vec4);
                vec3_256    = _mm256_hsub_pd(vec3_256, vec4_256);
                factor_vec2 = _mm256_add_pd(factor_vec2, vec3_256);

                vec3_256    = _mm512_extractf64x4_pd(vec3, 1);
                vec4_256    = _mm512_extractf64x4_pd(vec4, 1);
                vec3_256    = _mm256_hsub_pd(vec3_256, vec4_256);
                factor_vec2 = _mm256_add_pd(factor_vec2, vec3_256);

            }
        }

        // sum up the contributions
        __m128d factor = _mm256_castpd256_pd128(factor_vec);
        factor = _mm_add_pd(factor, _mm256_extractf128_pd(factor_vec, 1));

        __m128d factor2 = _mm256_castpd256_pd128(factor_vec2);
        factor2 = _mm_add_pd(factor2, _mm256_extractf128_pd(factor_vec2, 1));

        size_t reminder = A.cols % 4;
        __m256d neg_256 = _mm512_castpd512_pd256(neg);
      
        for (size_t kdx = A.cols-reminder; kdx<A.cols; kdx++) {
 
            __m256d A_vec;
            A_vec = _mm256_insertf128_pd(A_vec, _mm_load_pd(data+2*kdx), 0);
            A_vec = _mm256_insertf128_pd(A_vec, _mm_load_pd(data2+2*kdx), 1);

            // extract the last component of vector v
            __m128d v_vec = _mm_load_pd(v_data+2*kdx);

            // create vector v_i, v_i
            __m128d* v_element = (__m128d*)&v_vec[0];
            __m256d v_vec1 = _mm256_broadcast_pd( v_element );


            // calculate the multiplications  A_vec*v_vec1

            __m256d vec3 = _mm256_mul_pd(A_vec, v_vec1);
            __m256d v_vec_permuted = _mm256_permute_pd(v_vec1, 0x5);
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg_256);
            __m256d vec4 = _mm256_mul_pd(A_vec, v_vec_permuted);
            vec3  = _mm256_hsub_pd(vec3, vec4);
            factor = _mm_add_pd(factor, _mm256_castpd256_pd128(vec3));
            factor2 = _mm_add_pd(factor2, _mm256_extractf128_pd(vec3, 1));

        }


        // multiply factors by two
        __m128d two = _mm_setr_pd(2.0, 2.0);
        factor = _mm_mul_pd(factor, two);
        factor2 = _mm_mul_pd(factor2, two);

        factor_vec = _mm256_broadcast_pd( (__m128d*)&factor[0] );
        factor_vec2 = _mm256_broadcast_pd( (__m128d*)&factor2[0] );


        for (size_t kdx = 0; kdx < sizeH-1; kdx = kdx + 2) {


            // extract two successive components v_i,v_{i+1} of vector v
            __m256d v_vec = _mm256_loadu_pd(v_data+2*kdx);

            // calculate the multiplications  factor_vec*conj(v_vec)

            // 2 Switch the real and imaginary elements of v_vec1
            __m256d v_vec_permuted = _mm256_permute_pd(v_vec, 0x5);
            __m256d vec3 = _mm256_mul_pd(factor_vec, v_vec);
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg2);
            __m256d vec4 = _mm256_mul_pd(factor_vec, v_vec_permuted);
            vec3  = _mm256_hadd_pd(vec3, vec4);

            __m256d A_vec = _mm256_loadu_pd(data+2*kdx);
            A_vec = _mm256_sub_pd(A_vec, vec3);
            _mm256_storeu_pd(data+2*kdx, A_vec);


            // calculate the multiplications  A_vec2*v_vec
            vec3 = _mm256_mul_pd(factor_vec2, v_vec);
            vec4 = _mm256_mul_pd(factor_vec2, v_vec_permuted);
            vec3  = _mm256_hadd_pd(vec3, vec4);

            __m256d A_vec2 = _mm256_loadu_pd(data2+2*kdx);
            A_vec2 = _mm256_sub_pd(A_vec2, vec3);
            _mm256_storeu_pd(data2+2*kdx, A_vec2);

        }


        if (sizeH % 2 == 1) {
            size_t kdx = A.cols-1;

            factor_vec = _mm256_insertf128_pd(factor_vec, factor2, 1 );

            // extract the last component of vector v
            __m128d v_vec = _mm_load_pd(v_data+2*kdx);

            // create vector v_i, v_i
            __m128d* v_element = (__m128d*)&v_vec[0];
            __m256d v_vec1 = _mm256_broadcast_pd( v_element );


            // calculate the multiplications  factor_vec*conj(v_vec1)

            // 2 Switch the real and imaginary elements of v_vec1
            __m256d v_vec_permuted = _mm256_permute_pd(v_vec1, 0x5);
            __m256d vec3 = _mm256_mul_pd(factor_vec, v_vec1);
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg2);
            __m256d vec4 = _mm256_mul_pd(factor_vec, v_vec_permuted);
            vec3  = _mm256_hadd_pd(vec3, vec4);

            __m128d A_vec = _mm_loadu_pd(data+2*kdx);
            A_vec = _mm_sub_pd(A_vec, _mm256_castpd256_pd128(vec3));
            _mm_storeu_pd(data+2*kdx, A_vec);

           __m128d A_vec2 = _mm_loadu_pd(data2+2*kdx);
            A_vec2 = _mm_sub_pd(A_vec2, _mm256_extractf128_pd(vec3, 1));
           _mm_storeu_pd(data2+2*kdx, A_vec2);

        }


        // move data pointers to the next row pair
        data = data + 4*A.stride;
        data2 = data2 + 4*A.stride;


    }


/////////////////////////////////////////////////////////////////////////////////////////////
    if (A.rows % 2 == 1 ) {

        __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
        __m256d neg2 = _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);


        __m256d factor_vec = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);


        for (size_t kdx = 0; kdx < A.cols-1; kdx = kdx + 2) {
            __m256d A_vec = _mm256_loadu_pd(data+2*kdx);

            // extract two successive components v_i,v_{i+1} of vector v
            __m256d v_vec = _mm256_loadu_pd(v_data+2*kdx);

            // calculate the multiplications  A_vec*v_vec

            // 2 Switch the real and imaginary elements of v_vec
            __m256d v_vec_permuted = _mm256_permute_pd(v_vec, 0x5);

            // 1 Multiply elements of A_vec and v_vec
            __m256d vec3 = _mm256_mul_pd(A_vec, v_vec);

            // 3 Negate the imaginary elements of cx_vec
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg);

            // 4 Multiply elements of A_vec and the modified v_vec
            __m256d vec4 = _mm256_mul_pd(A_vec, v_vec_permuted);

            // 5 Horizontally subtract the elements in vec3 and vec4
            vec3  = _mm256_hsub_pd(vec3, vec4);

            // add result of multiplication to factor_vec
            factor_vec = _mm256_add_pd(factor_vec, vec3);



        }

        // sum up the contributions
        __m128d factor = _mm256_castpd256_pd128(factor_vec);
        factor = _mm_add_pd(factor, _mm256_extractf128_pd(factor_vec, 1));



        if (A.cols % 2 == 1) {
            size_t kdx = A.cols-1;
            __m128d neg = _mm_setr_pd(1.0, -1.0);

            __m128d A_vec;
            A_vec = _mm_load_pd(data+2*kdx);

            // extract the last component of vector v
            __m128d v_vec = _mm_load_pd(v_data+2*kdx);


            // calculate the multiplications  A_vec*v_vec

            // 2 Switch the real and imaginary elements of v_vec1
            __m128d v_vec_permuted = _mm_permute_pd(v_vec, 0x5);

            // 1 Multiply elements of A_vec and v_vec1
            __m128d vec3 = _mm_mul_pd(A_vec, v_vec);

            // 3 Negate the imaginary elements of v_vec_permuted
            v_vec_permuted = _mm_mul_pd(v_vec_permuted, neg);

            // 4 Multiply elements of A_vec and the modified v_vec
            __m128d vec4 = _mm_mul_pd(A_vec, v_vec_permuted);

            // 5 Horizontally subtract the elements in vec3 and vec4
            vec3  = _mm_hsub_pd(vec3, vec4);

            // add the result to factor
            factor = _mm_add_pd(factor, vec3);


        }


        // multiply factors by two
        __m128d two = _mm_setr_pd(2.0, 2.0);
        factor = _mm_mul_pd(factor, two);

        factor_vec = _mm256_broadcast_pd( (__m128d*)&factor[0] );


        for (size_t kdx = 0; kdx < sizeH-1; kdx = kdx + 2) {


            // extract two successive components v_i,v_{i+1} of vector v
            __m256d v_vec = _mm256_loadu_pd(v_data+2*kdx);

            // calculate the multiplications  factor_vec*conj(v_vec)

            // 2 Switch the real and imaginary elements of v_vec1
            __m256d v_vec_permuted = _mm256_permute_pd(v_vec, 0x5);

            // 1 Multiply elements of factor_vec and v_vec
            __m256d vec3 = _mm256_mul_pd(factor_vec, v_vec);

            // 3 Negate the imaginary elements of v_vec
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg2);

            // 4 Multiply elements of A_vec and the modified v_vec
            __m256d vec4 = _mm256_mul_pd(factor_vec, v_vec_permuted);

            // 5 Horizontally subtract the elements in vec3 and vec4
            vec3  = _mm256_hadd_pd(vec3, vec4);

            __m256d A_vec = _mm256_loadu_pd(data+2*kdx);
            A_vec = _mm256_sub_pd(A_vec, vec3);
            _mm256_storeu_pd(data+2*kdx, A_vec);


        }


        if (sizeH % 2 == 1) {
            size_t kdx = A.cols-1;
            __m128d neg2 = _mm_setr_pd(-1.0, 1.0);


            // extract the last component of vector v
            __m128d v_vec = _mm_load_pd(v_data+2*kdx);


            // calculate the multiplications  factor*conj(v_vec)

            // 2 Switch the real and imaginary elements of v_vec1
            __m128d v_vec_permuted = _mm_permute_pd(v_vec, 0x5);

            // 1 Multiply elements of A_vec and v_vec1
            __m128d vec3 = _mm_mul_pd(factor, v_vec);

            // 3 Negate the imaginary elements of v_vec_permuted
            v_vec_permuted = _mm_mul_pd(v_vec_permuted, neg2);

            // 4 Multiply elements of A_vec and the modified v_vec
            __m128d vec4 = _mm_mul_pd(factor, v_vec_permuted);

            // 5 Horizontally subtract the elements in vec3 and vec4
            vec3  = _mm_hadd_pd(vec3, vec4);

            __m128d A_vec = _mm_loadu_pd(data+2*kdx);
            A_vec = _mm_sub_pd(A_vec, vec3);
            _mm_storeu_pd(data+2*kdx, A_vec);


    }


}


/*

    // The above code with non-AVX instructions
    // calculate A^~(1-2vov)
    for (size_t idx = 0; idx < A.rows-1; idx=idx+2) {

        Complex16* data_A = A.get_data() + idx*A.stride;
        Complex16* data_A2 = data_A + A.stride;

        Complex16 factor(0.0,0.0);
        Complex16 factor2(0.0,0.0);
        for (size_t v_idx = 0; v_idx < sizeH; v_idx++) {
            factor  = factor  + data_A[v_idx] * v[v_idx];
            factor2 = factor2 + data_A2[v_idx] * v[v_idx];
        }


        factor  = factor*2.0;
        factor2 = factor2*2.0;
        for (size_t jdx=0; jdx<sizeH; jdx++) {
            data_A[jdx] = data_A[jdx] - mult_a_bconj(factor, v[jdx]);
            data_A2[jdx] = data_A2[jdx] - mult_a_bconj(factor2, v[jdx]);
        }


    }


    if (A.rows % 2 == 1 ) {

        Complex16* data_A = A.get_data() + (A.rows-1)*A.stride;

        Complex16 factor(0.0,0.0);
        for (size_t v_idx = 0; v_idx < sizeH; v_idx++) {
            factor = factor + data_A[v_idx] * v[v_idx];
        }

        factor = factor*2.0;
        for (size_t jdx=0; jdx<sizeH; jdx++) {
            data_A[jdx] = data_A[jdx] - mult_a_bconj(factor, v[jdx]);
        }



    }
*/



    return;

}


} // PIC
