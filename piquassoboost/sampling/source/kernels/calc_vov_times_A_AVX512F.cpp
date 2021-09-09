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

#include "calc_vov_times_A_AVX.h"

#include "tbb/tbb.h"
static tbb::spin_mutex my_mutex;

namespace pic {

/**
@brief AVX kernel to
*/
void
calc_vov_times_A_AVX(matrix &A, matrix &v, matrix &vH_times_A) {


    size_t size_v = v.size();


    // pointers to two successive rows of matrix A
    double* data = (double*)A.get_data();
    double* data2 = data + 2*A.stride; // factor 2 is needed becouse of one complex16 consists of two doubles
    double* v_data = (double*)v.get_data();
    double* vH_times_A_data = (double*)vH_times_A.get_data();


    __m512d neg_512 = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    __m256d neg = _mm512_castpd512_pd256(neg_512);//_mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    __m256d two = _mm256_setr_pd(2.0, 2.0, 2.0, 2.0);

    // calculate the vector-vector product v * ((v^+) * A))
    for (size_t row_idx = 0; row_idx < 2*(size_v-1); row_idx = row_idx+4) {


        // extract two successive components v_i,v_{i+1} of vector v
        __m256d v_vec = _mm256_loadu_pd(v_data+row_idx);

        // multiply v_vec by two
        v_vec = _mm256_mul_pd(v_vec, two);

        // create vector v_{i}, v_{i}, v_{i}, v_{i}
        __m512d v_vec1_512 = _mm512_setr_pd(v_vec[0], v_vec[1], v_vec[0], v_vec[1], v_vec[0], v_vec[1], v_vec[0], v_vec[1]);

        // create vector v_{i+1}, v_{i+1}, v_{i+1}, v_{i+1}
        __m512d v_vec2_512 = _mm512_setr_pd(v_vec[2], v_vec[3], v_vec[2], v_vec[3], v_vec[2], v_vec[3], v_vec[2], v_vec[3]);


        if ( A.cols > 3 ) {
            for (size_t kdx = 0; kdx < 2*(A.cols-3); kdx=kdx+8) {

                __m512d vH_times_A_vec = _mm512_loadu_pd(vH_times_A_data + kdx);
 
                // calculate the multiplications  vH_times_A_ve*v_vec1

                __m512d vec3           = _mm512_mul_pd(vH_times_A_vec, v_vec1_512);
                __m512d v_vec_permuted = _mm512_permute_pd(v_vec1_512, 0x55);
                v_vec_permuted         = _mm512_mul_pd(v_vec_permuted, neg_512);
                __m512d vec4           = _mm512_mul_pd(vH_times_A_vec, v_vec_permuted);
                __m512d vec5           = _mm512_permute_pd(vec3, 0xFF);
                vec3                   = _mm512_sub_pd(vec3, vec5);
                vec5                   = _mm512_permute_pd(vec4, 0x0);
                vec4                   = _mm512_sub_pd(vec5, vec4);
                vec3                   = _mm512_add_pd(vec3, vec4);

                // subtract the result from A_vec
                __m512d A_vec_512 = _mm512_loadu_pd(data+kdx);
                A_vec_512 = _mm512_sub_pd(A_vec_512, vec3);
                _mm512_storeu_pd(data+kdx, A_vec_512);


                // calculate the multiplications  vH_times_A_vec*v_vec2
                vec3                   = _mm512_mul_pd(vH_times_A_vec, v_vec2_512);
                v_vec_permuted         = _mm512_permute_pd(v_vec2_512, 0x55);
                v_vec_permuted         = _mm512_mul_pd(v_vec_permuted, neg_512);
                vec4                   = _mm512_mul_pd(vH_times_A_vec, v_vec_permuted);
                vec5                   = _mm512_permute_pd(vec3, 0xFF);
                vec3                   = _mm512_sub_pd(vec3, vec5);
                vec5                   = _mm512_permute_pd(vec4, 0x0);
                vec4                   = _mm512_sub_pd(vec5, vec4);
                vec3                   = _mm512_add_pd(vec3, vec4);

                // subtract the result from A_vec
                A_vec_512 = _mm512_loadu_pd(data2+kdx);
                A_vec_512 = _mm512_sub_pd(A_vec_512, vec3);
                _mm512_storeu_pd(data2+kdx, A_vec_512);


            }
        }


        // create vector v_{i+1}, v_{i+1}
        __m256d v_vec2 = _mm512_castpd512_pd256(v_vec2_512);

        // create vector v_i, v_i
        __m256d v_vec1 = _mm512_castpd512_pd256(v_vec1_512);

        size_t reminder = A.cols % 4;
        if (reminder >= 2) {
        //for (size_t kdx = 0; kdx < 2*(A.cols-1); kdx=kdx+4) {

            size_t kdx = 2*(A.cols-reminder);

            __m256d vH_times_A_vec = _mm256_loadu_pd(vH_times_A_data+kdx);

            // calculate the multiplications  vH_times_A_vec*v_vec1
            __m256d vec3 = _mm256_mul_pd(vH_times_A_vec, v_vec1);
            __m256d v_vec_permuted = _mm256_permute_pd(v_vec1, 0x5);
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg);
            __m256d vec4 = _mm256_mul_pd(vH_times_A_vec, v_vec_permuted);
            vec3  = _mm256_hsub_pd(vec3, vec4);

            // subtract the result from A_vec
            __m256d A_vec = _mm256_loadu_pd(data+kdx);
            A_vec = _mm256_sub_pd(A_vec, vec3);

            // store the result to th memory
            _mm256_storeu_pd(data+kdx, A_vec);



            // calculate the multiplications  vH_times_A_vec*v_vec2
            vec3 = _mm256_mul_pd(vH_times_A_vec, v_vec2);
            v_vec_permuted = _mm256_permute_pd(v_vec2, 0x5);
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg);
            vec4 = _mm256_mul_pd(vH_times_A_vec, v_vec_permuted);
            vec3  = _mm256_hsub_pd(vec3, vec4);

            // subtract the result from A_vec2
            __m256d A_vec2 = _mm256_loadu_pd(data2+kdx);
            A_vec2 = _mm256_sub_pd(A_vec2, vec3);

            // store the result to th memory
            _mm256_storeu_pd(data2+kdx, A_vec2);


            reminder = reminder - 2;

        }


        if ( reminder == 1) {
            size_t kdx = 2*(A.cols-1);

            __m256d vH_times_A_vec = _mm256_broadcast_pd( (__m128d*)(vH_times_A_data+kdx) );

            // calculate the multiplications  vH_times_A_vec*v_vec
            __m256d vec3 = _mm256_mul_pd(vH_times_A_vec, v_vec);
            __m256d v_vec_permuted = _mm256_permute_pd(v_vec, 0x5);
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg);
            __m256d vec4 = _mm256_mul_pd(vH_times_A_vec, v_vec_permuted);
            vec3  = _mm256_hsub_pd(vec3, vec4);

            // subtract the result from A_vec
            __m256d A_vec;
            A_vec = _mm256_insertf128_pd(A_vec, _mm_loadu_pd(data+kdx), 0);
            A_vec = _mm256_insertf128_pd(A_vec, _mm_loadu_pd(data2+kdx), 1);
            A_vec = _mm256_sub_pd(A_vec, vec3);

            // store the result to th memory
            _mm_storeu_pd(data+kdx, _mm256_castpd256_pd128(A_vec));
            _mm_storeu_pd(data2+kdx, _mm256_extractf128_pd(A_vec, 1));


        }


        // move data pointers to the next row pair
        data = data + 4*A.stride;
        data2 = data2 + 4*A.stride;

    }

//////////////////////////////////////////////////////////////////

    if (size_v % 2 == 1 ) {

        size_t row_idx = 2*(v.rows-1);

        __m128d two = _mm_setr_pd(2.0, 2.0);

        // extract two successive components v_i,v_{i+1} of vector v
        __m128d v_vec = _mm_loadu_pd(v_data+row_idx);

        // multiply v_vec1 by two
        v_vec = _mm_mul_pd(v_vec, two);


        // create vector v_{i}, v_{i}, v_{i}, v_{i}
        __m512d v_vec1_512 = _mm512_setr_pd(v_vec[0], v_vec[1], v_vec[0], v_vec[1], v_vec[0], v_vec[1], v_vec[0], v_vec[1]);


        if ( A.cols > 3 ) {
            for (size_t kdx = 0; kdx < 2*(A.cols-3); kdx=kdx+8) {

                __m512d vH_times_A_vec = _mm512_loadu_pd(vH_times_A_data + kdx);
 
                // calculate the multiplications  vH_times_A_ve*v_vec1

                __m512d vec3           = _mm512_mul_pd(vH_times_A_vec, v_vec1_512);
                __m512d v_vec_permuted = _mm512_permute_pd(v_vec1_512, 0x55);
                v_vec_permuted         = _mm512_mul_pd(v_vec_permuted, neg_512);
                __m512d vec4           = _mm512_mul_pd(vH_times_A_vec, v_vec_permuted);
                __m512d vec5           = _mm512_permute_pd(vec3, 0xFF);
                vec3                   = _mm512_sub_pd(vec3, vec5);
                vec5                   = _mm512_permute_pd(vec4, 0x0);
                vec4                   = _mm512_sub_pd(vec5, vec4);
                vec3                   = _mm512_add_pd(vec3, vec4);

                // subtract the result from A_vec
                __m512d A_vec_512 = _mm512_loadu_pd(data+kdx);
                A_vec_512 = _mm512_sub_pd(A_vec_512, vec3);
                _mm512_storeu_pd(data+kdx, A_vec_512);

            }
        }


        // create vector v_i, v_i
        __m256d v_vec1 = _mm512_castpd512_pd256(v_vec1_512);

        size_t reminder = A.cols % 4;
        if (reminder >= 2) {

            size_t kdx = 2*(A.cols-reminder);

            __m256d vH_times_A_vec = _mm256_loadu_pd(vH_times_A_data+kdx);

            // calculate the multiplications  vH_times_A_vec*v_vec1
            __m256d vec3 = _mm256_mul_pd(vH_times_A_vec, v_vec1);
            __m256d v_vec_permuted = _mm256_permute_pd(v_vec1, 0x5);
            v_vec_permuted = _mm256_mul_pd(v_vec_permuted, neg);
            __m256d vec4 = _mm256_mul_pd(vH_times_A_vec, v_vec_permuted);
            vec3  = _mm256_hsub_pd(vec3, vec4);

            // subtract the result from A_vec
            __m256d A_vec = _mm256_loadu_pd(data+kdx);
            A_vec = _mm256_sub_pd(A_vec, vec3);
            _mm256_storeu_pd(data+kdx, A_vec);

            reminder = reminder - 2;

        }

        if ( reminder == 1) {
            size_t kdx = 2*(A.cols-1);

            __m128d neg = _mm_setr_pd(1.0, -1.0);
            __m128d vH_times_A_vec = _mm_loadu_pd(vH_times_A_data+kdx);

            // calculate the multiplications  vH_times_A_vec*v_vec
            __m128d vec3 = _mm_mul_pd(vH_times_A_vec, v_vec);
            __m128d v_vec_permuted = _mm_permute_pd(v_vec, 0x5);
            v_vec_permuted = _mm_mul_pd(v_vec_permuted, neg);
            __m128d vec4 = _mm_mul_pd(vH_times_A_vec, v_vec_permuted);
            vec3  = _mm_hsub_pd(vec3, vec4);

            // subtract the result from A_vec
            __m128d A_vec = _mm_loadu_pd(data+kdx);
            A_vec = _mm_sub_pd(A_vec, vec3);
            _mm_storeu_pd(data+kdx, A_vec);


        }


    }




/*
        // The above code with non-AVX instructions
        // calculate the vector-vector product v * ((v^+) * A))
        for (size_t row_idx = 0; row_idx < size_v-1; row_idx = row_idx+2) {

            size_t offset_data_A =  row_idx * A.stride;
            Complex16* data_A = A.get_data() + offset_data_A;
            Complex16* data_A2 = data_A + A.stride;

            Complex16 factor = v[row_idx]*2.0;
            Complex16 factor2 = v[row_idx+1]*2.0;

            for (size_t kdx = 0; kdx < A.cols-1; kdx=kdx+2) {
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx];
                data_A2[kdx] = data_A2[kdx] - factor2 * vH_times_A[kdx];
                data_A[kdx+1] = data_A[kdx+1] - factor * vH_times_A[kdx+1];
                data_A2[kdx+1] = data_A2[kdx+1] - factor2 * vH_times_A[kdx+1];
            }


            if ( A.cols % 2 == 1) {
                size_t kdx = A.cols-1;
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx];
                data_A2[kdx] = data_A2[kdx] - factor2 * vH_times_A[kdx];
            }


        }



        if (size_v % 2 == 1 ) {

            size_t row_idx = v.rows-1;
            Complex16* data_A = A.get_data() + row_idx * A.stride;

            Complex16 factor = v[row_idx]*2.0;

            for (size_t kdx = 0; kdx < A.cols-1; kdx=kdx+2) {
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx];
                data_A[kdx+1] = data_A[kdx+1] - factor * vH_times_A[kdx+1];
            }


            if ( A.cols % 2 == 1) {
                size_t kdx = A.cols-1;
                data_A[kdx] = data_A[kdx] - factor * vH_times_A[kdx];
            }

        }
*/




    return;

}


} // PIC
