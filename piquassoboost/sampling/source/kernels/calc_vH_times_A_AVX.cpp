#include "calc_vH_times_A_AVX.h"


namespace pic {

/**
@brief AVX kernel to apply householder transformation on a matrix A' = (1 - 2*v o v) A for one specific reflection vector v.
@param A matrix on which the householder transformation is applied.
@param v A matrix instance of the reflection vector
@param vH_times_A preallocated array to hold the data of vH_times_A. The result is returned via this reference.
*/
void
calc_vH_times_A_AVX(matrix &A, matrix &v, matrix &vH_times_A) {


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

        for (size_t kdx = 0; kdx < 2*(A.cols-1); kdx = kdx + 4) {
            __m256d A_vec = _mm256_loadu_pd(data+kdx);
            __m256d A_vec2 = _mm256_loadu_pd(data2+kdx);

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
            __m256d _tmp = _mm256_loadu_pd(vH_times_A_data+kdx);
            _tmp = _mm256_add_pd(_tmp, A_vec);
            _mm256_storeu_pd(vH_times_A_data+kdx, _tmp);

        }



        if (A.cols % 2 == 1) {
            size_t kdx = 2*(A.cols-1);
            __m256d A_vec;
            A_vec = _mm256_insertf128_pd(A_vec, _mm_load_pd(data+kdx), 0);
            A_vec = _mm256_insertf128_pd(A_vec, _mm_load_pd(data2+kdx), 1);

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
            __m128d _tmp = _mm_loadu_pd(vH_times_A_data+kdx);
            _tmp = _mm_add_pd(_tmp, _tmp2);
            _mm_storeu_pd(vH_times_A_data+kdx, _tmp);

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


        for (size_t kdx = 0; kdx < 2*(A.cols-1); kdx = kdx + 4) {
            __m256d A_vec = _mm256_loadu_pd(data+kdx);

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
            __m256d _tmp = _mm256_loadu_pd(vH_times_A_data+kdx);
            _tmp = _mm256_add_pd(_tmp, A_vec);
            _mm256_storeu_pd(vH_times_A_data+kdx, _tmp);


        }


        if (A.cols % 2 == 1) {
            size_t kdx = 2*(A.cols-1);

            __m128d neg = _mm_setr_pd(-1.0, 1.0);
            __m128d A_vec = _mm_loadu_pd(data+kdx);

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

            __m128d _tmp = _mm_loadu_pd(vH_times_A_data+kdx);
            _tmp = _mm_add_pd(_tmp, A_vec);
            _mm_storeu_pd(vH_times_A_data+kdx, _tmp);

        }



    }


/*
// The above code with non-AVX instructions
// calculate the vector-matrix product (v^+) * A
      for (size_t row_idx = 0; row_idx < sizeH-1; row_idx=row_idx+2) {

          size_t offset_A_data =  row_idx * A.stride;
          Complex16* data_A = A.get_data() + offset_A_data;
          Complex16* data_A2 = data_A + A.stride;

          for (size_t j = 0; j < A.cols-1; j = j + 2) {
              vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);
              vH_times_A[j+1] = vH_times_A[j+1] + mult_a_bconj(data_A[j+1], v[row_idx]);
              vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A2[j], v[row_idx+1]);
              vH_times_A[j+1] = vH_times_A[j+1] + mult_a_bconj(data_A2[j+1], v[row_idx+1]);
          }


          if (A.cols % 2 == 1) {
              size_t j = A.cols-1;
              vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);
              vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A2[j], v[row_idx+1]);

          }


      }

if (sizeH % 2 == 1) {

    size_t row_idx = sizeH-1;

    size_t offset_A_data =  row_idx * A.stride;
    Complex16* data_A = A.get_data() + offset_A_data;


    for (size_t j = 0; j < A.cols-1; j = j + 2) {
        vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);
        vH_times_A[j+1] = vH_times_A[j+1] + mult_a_bconj(data_A[j+1], v[row_idx]);
    }


    if (A.cols % 2 == 1) {
         size_t j = A.cols-1;
         vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);

    }



}

*/

    return;

}


} // PIC
