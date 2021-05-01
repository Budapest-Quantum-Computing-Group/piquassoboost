
std::cout << "jjjjjjjjjjjjj" << std::endl;
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


    for (size_t j = 0; j < A.cols; j = j + 1) {
        vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);
        //vH_times_A[j+1] = vH_times_A[j+1] + mult_a_bconj(data_A[j+1], v[row_idx]);
    }


    if (A.cols % 2 == 1) {
         size_t j = A.cols-1;
         vH_times_A[j] = vH_times_A[j] + mult_a_bconj(data_A[j], v[row_idx]);

    }

    

}


/*
    size_t max_idx = cx_diag_elements.size();
    matrix tmp_vec(1, max_idx);
    double* tmp_vec_data = (double*)tmp_vec.get_data();
    double* cx_data = (double*)cx_diag_elements.get_data();
    double* diag_data = (double*)diag_elements.get_data();

    __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

    for (size_t idx=0; idx<num_of_modes; idx++) {


        Complex16 tmp(0.0,0.0);

        for(size_t kdx = 0; kdx<max_idx*2; kdx=kdx+4) {
            __m256d diag_vec = _mm256_load_pd(diag_data+kdx);
            __m256d cx_vec = _mm256_load_pd(cx_data+kdx);

            // Multiply elements of AZ_vec and cx_vec
            __m256d vec3 = _mm256_mul_pd(diag_vec, cx_vec);

            // Switch the real and imaginary elements of vec2
            cx_vec = _mm256_permute_pd(cx_vec, 0x5);

            // Negate the imaginary elements of cx_vec
            cx_vec = _mm256_mul_pd(cx_vec, neg);

            // Multiply elements of AZ_vec and the modified cx_vec
            __m256d vec4 = _mm256_mul_pd(diag_vec, cx_vec);

            // Horizontally subtract the elements in vec3 and vec4
            cx_vec = _mm256_hsub_pd(vec3, vec4);

            // get the higher 128 bit of the register
            __m128d cx_vec_high = _mm256_extractf128_pd(cx_vec, 1);

            // calculate the sum of the numbers
            cx_vec_high = _mm_add_pd(cx_vec_high, _mm256_castpd256_pd128(cx_vec) );

            tmp += *((Complex16*)&cx_vec_high[0]);


        }

        loop_correction[idx] = tmp;


        double* data = (double*)AZ.get_data();

        __m256d _tmp = _mm256_setzero_pd();


        for(size_t kdx = 0; kdx<max_idx*2; kdx=kdx+4) {
            __m256d AZ_vec = _mm256_load_pd(data+kdx);
            __m256d AZ_vec2 = _mm256_load_pd(data+2*AZ.stride+kdx);
            __m256d cx_vec = _mm256_load_pd(cx_data+kdx);

            // Multiply elements of AZ_vec and cx_vec
            __m256d vec3 = _mm256_mul_pd(AZ_vec, cx_vec);
            __m256d vec5 = _mm256_mul_pd(AZ_vec2, cx_vec);

            // Switch the real and imaginary elements of vec2
            cx_vec = _mm256_permute_pd(cx_vec, 0x5);

            // Negate the imaginary elements of cx_vec
            cx_vec = _mm256_mul_pd(cx_vec, neg);

            // Multiply elements of AZ_vec and the modified cx_vec
            __m256d vec4 = _mm256_mul_pd(AZ_vec, cx_vec);
            __m256d vec6 = _mm256_mul_pd(AZ_vec2, cx_vec);

            // Horizontally subtract the elements in vec3 and vec4
            AZ_vec  = _mm256_hsub_pd(vec3, vec4);
            AZ_vec2 = _mm256_hsub_pd(vec5, vec6);

            // get the higher 128 bit of the register
            __m128d AZ_vec_high = _mm256_extractf128_pd(AZ_vec, 1);

            // substitute lower 128 bits of register AZ_vec2 into the higher 128 bits of AZ_vec2
            AZ_vec = _mm256_insertf128_pd(AZ_vec, _mm256_castpd256_pd128(AZ_vec2), 1);

            // insert previous higher 128 bits of register AZ_vec into lower 128bits of AZ_vec2
            AZ_vec2 = _mm256_insertf128_pd(AZ_vec2, AZ_vec_high, 0);

            // sum up the elements of vector AZ_ec and AZ_vec2
            AZ_vec = _mm256_add_pd(AZ_vec, AZ_vec2);

            _tmp = _mm256_add_pd(_tmp, AZ_vec );



        }

        _mm256_storeu_pd(tmp_vec_data, _tmp);

        for (size_t jdx=2; jdx<max_idx; jdx=jdx+2) {
            data = data + 4*AZ.stride;

            __m256d _tmp = _mm256_setzero_pd();
            __m128d _tmp2 = _mm_setzero_pd();
            __m128d _tmp3 = _mm_setzero_pd();

            size_t start_idx = jdx-2;
            for(size_t kdx = 2*start_idx; kdx<max_idx*2; kdx=kdx+4) {
                __m256d AZ_vec = _mm256_load_pd(data + kdx);
                __m256d AZ_vec2 = _mm256_load_pd(data+2*AZ.stride+kdx);
                __m256d cx_vec = _mm256_load_pd(cx_data + kdx);

                // Multiply elements of AZ_vec and cx_vec
                __m256d vec3 = _mm256_mul_pd(AZ_vec, cx_vec);
                __m256d vec5 = _mm256_mul_pd(AZ_vec2, cx_vec);

                // Switch the real and imaginary elements of vec2
                cx_vec = _mm256_permute_pd(cx_vec, 0x5);

                // Negate the imaginary elements of cx_vec
                cx_vec = _mm256_mul_pd(cx_vec, neg);

                // Multiply elements of AZ_vec and the modified cx_vec
                __m256d vec4 = _mm256_mul_pd(AZ_vec, cx_vec);
                __m256d vec6 = _mm256_mul_pd(AZ_vec2, cx_vec);

                // Horizontally subtract the elements in vec3 and vec4
                AZ_vec  = _mm256_hsub_pd(vec3, vec4);
                AZ_vec2 = _mm256_hsub_pd(vec5, vec6);

                // get the higher 128 bit of the register
                __m128d AZ_vec_high = _mm256_extractf128_pd(AZ_vec, 1);

                // substitute lower 128 bits of register AZ_vec2 into the higher 128 bits of AZ_vec2
                AZ_vec = _mm256_insertf128_pd(AZ_vec, _mm256_castpd256_pd128(AZ_vec2), 1);

                // insert previous higher 128 bits of register AZ_vec into lower 128bits of AZ_vec2
                AZ_vec2 = _mm256_insertf128_pd(AZ_vec2, AZ_vec_high, 0);

                // sum up the elements of vector AZ_ec and AZ_vec2
                AZ_vec = _mm256_add_pd(AZ_vec, AZ_vec2);

                _tmp = _mm256_add_pd(_tmp, AZ_vec );
            }

            _mm256_storeu_pd(tmp_vec_data+2*jdx, _tmp);

        }







        memcpy(cx_diag_elements.get_data(), tmp_vec.get_data(), tmp_vec.size()*sizeof(Complex16));

    }

*/
