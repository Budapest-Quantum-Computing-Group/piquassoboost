#include "calc_cholesky_decomposition_AVX.h"

#include "tbb/tbb.h"
static tbb::spin_mutex my_mutex;

namespace pic {

/**
@brief AVX kernel to calculate in-place Cholesky decomposition of a matrix
@param mtx A positive definite hermitian matrix with eigenvalues less then unity.  The decomposed matrix is stored in mtx.
@param reuse_index Labels the row and column from which the Cholesky decomposition should be continued.
@param determinant The determinant of the matrix is calculated and stored in this variable.
(if reuse_index index is greater than 0, than the contributions of the first reuse_index-1 elements of the Cholesky L matrix should be multiplied manually)
*/
void
calc_cholesky_decomposition_AVX(matrix& matrix, size_t reuse_index, Complex32 &determinant) {

    // The above code with non-AVX instructions
       // storing in the same memory the results of the algorithm
    int n = matrix.cols;

    __m256d neg2 = _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);

    double* row_i = (double*)matrix.get_data() + 2*(reuse_index)*matrix.stride;

    // Decomposing a matrix into lower triangular matrices
    for (int idx = reuse_index; idx < n; idx++) {

        Complex16* row_j = matrix.get_data() + reuse_index*matrix.stride;
        Complex16* row_j2 = row_j + matrix.stride;

        for (int j = reuse_index; j < idx-1; j=j+2) {

            //Complex16 sum = 0;
            //Complex16 sum2 = 0;

            __m256d sum1_256 = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
            __m256d sum2_256 = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);

            // Evaluating L(i, j) using L(j, j)
            // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
            for (int kdx = 0; kdx < 2*(j-1); kdx=kdx+4){

                __m256d row_i_256       = _mm256_loadu_pd(row_i+kdx);
                __m256d row_i_permuted  = _mm256_mul_pd(row_i_256, neg2);
                row_i_permuted          = _mm256_permute_pd(row_i_permuted, 0x5);


                // calculate the multiplications  rowi*conj(row_j)
                __m256d row_j_256      = _mm256_loadu_pd((double*)row_j+kdx);
                __m256d vec3           = _mm256_mul_pd(row_i_256, row_j_256);
                __m256d vec4           = _mm256_mul_pd(row_i_permuted, row_j_256);
                vec3                   = _mm256_hadd_pd(vec3, vec4);

                // add the result to sum1
                sum1_256                = _mm256_add_pd(sum1_256, vec3);


                // calculate the multiplications  rowi*conj(row_j2)
                __m256d row_j2_256     = _mm256_loadu_pd((double*)row_j2+kdx);
                vec3                   = _mm256_mul_pd(row_i_256, row_j2_256);
                vec4                   = _mm256_mul_pd(row_i_permuted, row_j2_256);
                vec3                   = _mm256_hadd_pd(vec3, vec4);

                // add the result to sum2
                sum2_256                = _mm256_add_pd(sum2_256, vec3);

            }

            // sum up contributions
            __m128d sum1_128 = _mm256_castpd256_pd128(sum1_256);
            sum1_128         = _mm_add_pd(sum1_128, _mm256_extractf128_pd(sum1_256, 1));

            __m128d sum2_128 = _mm256_castpd256_pd128(sum2_256);
            sum2_128         = _mm_add_pd(sum2_128, _mm256_extractf128_pd(sum2_256, 1));



            __m256d row_i_256      = _mm256_loadu_pd(row_i+2*j);
            __m128d row_i_128      = _mm256_castpd256_pd128(row_i_256);
            row_i_128              = _mm_sub_pd(row_i_128, sum1_128);
            __m128d row_j_128      = _mm_loadu_pd((double*)row_j+2*j);
            __m128d row_j_norm_128 = _mm_mul_pd(row_j_128, row_j_128);
            row_j_norm_128         = _mm_hadd_pd(row_j_norm_128, row_j_norm_128);




            // calculate the multiplications  row_i_128*conj(row_j_128)
            __m128d neg2_128       = _mm256_castpd256_pd128(neg2);
            __m128d row_j_permuted = _mm_permute_pd(row_j_128, 0x5);
            __m128d vec3           = _mm_mul_pd(row_i_128, row_j_128);
            row_j_permuted         = _mm_mul_pd(row_j_permuted, neg2_128);
            __m128d vec4           = _mm_mul_pd(row_i_128, row_j_permuted);
            vec3                   = _mm_hadd_pd(vec3, vec4);

            // calculate vec3/row_j_norm and store it into row_i
            row_i_128              = _mm_div_pd(vec3, row_j_norm_128);

            _mm_storeu_pd(row_i+2*j, row_i_128);
//row_i[j]               = *((Complex16*)&row_i_128[0]);
            //row_i[j] = (row_i[j] - *sum) / row_j[j];

            // calculate sum2 = sum2 + row_i[j] * conj(rowj2[j] )
            __m256d row_j2_256     = _mm256_loadu_pd((double*)row_j2+2*j);
            __m128d row_j2_128     = _mm256_castpd256_pd128(row_j2_256);//_mm_loadu_pd((double*)row_j2+2*j);
            row_j_permuted         = _mm_permute_pd(row_j2_128, 0x5);
            vec3                   = _mm_mul_pd(row_i_128, row_j2_128);
            row_j_permuted         = _mm_mul_pd(row_j_permuted, neg2_128);
            vec4                   = _mm_mul_pd(row_i_128, row_j_permuted);
            vec3                   = _mm_hadd_pd(vec3, vec4);
            sum2_128               = _mm_add_pd(sum2_128, vec3);
            //*sum2 += mult_a_bconj( row_i[j], row_j2[j]);

            row_i_128              = _mm256_extractf128_pd(row_i_256, 1);
            row_i_128              = _mm_sub_pd(row_i_128, sum2_128);


            // calculate the multiplications  row_i_128*conj(row_j2_128)
            row_j2_128              = _mm256_extractf128_pd(row_j2_256, 1);
            __m128d row_j2_norm_128 = _mm_mul_pd(row_j2_128, row_j2_128);
            row_j2_norm_128         = _mm_hadd_pd(row_j2_norm_128, row_j2_norm_128);

            row_j_permuted         = _mm_permute_pd(row_j2_128, 0x5);
            vec3                   = _mm_mul_pd(row_i_128, row_j2_128);
            row_j_permuted         = _mm_mul_pd(row_j_permuted, neg2_128);
            vec4                   = _mm_mul_pd(row_i_128, row_j_permuted);
            vec3                   = _mm_hadd_pd(vec3, vec4);

            // calculate vec3/row_j_norm and store it into row_i
            row_i_128              = _mm_div_pd(vec3, row_j2_norm_128);
            _mm_storeu_pd(row_i+2*j+2, row_i_128);

            row_j = row_j + 2*matrix.stride;
            row_j2 = row_j2 + 2*matrix.stride;

        }



        if ( idx%2 == 1) {
            int j = idx-1;

            row_j = matrix.get_data() + j * matrix.stride;

             __m256d sum1_256 = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);

            // Evaluating L(i, j) using L(j, j)
            // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
            for (int kdx = 0; kdx < 2*(j-1); kdx=kdx+4){

                __m256d row_i_256 = _mm256_loadu_pd(row_i+kdx);

                // calculate the multiplications  rowi*conj(row_j)
                __m256d row_j_256      = _mm256_loadu_pd((double*)row_j+kdx);
                __m256d row_j_permuted = _mm256_permute_pd(row_j_256, 0x5);
                __m256d vec3           = _mm256_mul_pd(row_i_256, row_j_256);
                row_j_permuted         = _mm256_mul_pd(row_j_permuted, neg2);
                __m256d vec4           = _mm256_mul_pd(row_i_256, row_j_permuted);
                vec3                   = _mm256_hadd_pd(vec3, vec4);

                // add the result to sum1
                sum1_256                = _mm256_add_pd(sum1_256, vec3);



            }

            // sum up contributions
            __m128d sum1_128 = _mm256_castpd256_pd128(sum1_256);
            sum1_128         = _mm_add_pd(sum1_128, _mm256_extractf128_pd(sum1_256, 1));

            if (j%2 == 1) {

                __m128d neg2_128 = _mm256_castpd256_pd128(neg2);

                int kdx = 2*(j-1);

                __m128d row_i_128 = _mm_loadu_pd(row_i+kdx);

                // calculate the multiplications  rowi*conj(row_j)
                __m128d row_j_128      = _mm_loadu_pd((double*)row_j+kdx);
                __m128d row_j_permuted = _mm_permute_pd(row_j_128, 0x5);
                __m128d vec3           = _mm_mul_pd(row_i_128, row_j_128);
                row_j_permuted         = _mm_mul_pd(row_j_permuted, neg2_128);
                __m128d vec4           = _mm_mul_pd(row_i_128, row_j_permuted);
                vec3                   = _mm_hadd_pd(vec3, vec4);

                // add the result to sum1
                sum1_128                = _mm_add_pd(sum1_128, vec3);



            }


            Complex16* sum = (Complex16*)&sum1_128;
            Complex16* row_i_c = (Complex16*)row_i;

            row_i_c[j] = (row_i_c[j] - *sum) / row_j[j];


#ifdef DEBUG
            if (matrix.isnan()) {

                std::cout << "matrix is NAN" << std::endl;
                matrix.print_matrix();
                exit(-1);
             }
#endif
        }




        // summation for diagonals
        // L_{j,j}=\sqrt{A_{j,j}-\sum_{k=0}^{j-1}L_{j,k}L_{j,k}^*}

         __m256d sum1_256 = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);

            // Evaluating L(i, j) using L(j, j)
            // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
            for (int kdx = 0; kdx < 2*(idx-1); kdx=kdx+4){

                __m256d row_i_256 = _mm256_loadu_pd(row_i+kdx);

                // calculate the multiplications  rowi*conj(row_i)
                __m256d vec3           = _mm256_mul_pd(row_i_256, row_i_256);
                __m256d vec4           = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
                vec3                   = _mm256_hadd_pd(vec3, vec4);

                // add the result to sum1
                sum1_256                = _mm256_add_pd(sum1_256, vec3);

            }

            // sum up contributions
            __m128d sum1_128 = _mm256_castpd256_pd128(sum1_256);
            sum1_128         = _mm_add_pd(sum1_128, _mm256_extractf128_pd(sum1_256, 1));

            if (idx%2 == 1) {

                int kdx = 2*(idx-1);

                __m128d row_i_128 = _mm_loadu_pd(row_i+kdx);

                // calculate the multiplications  rowi*conj(row_j)
                __m128d vec3           = _mm_mul_pd(row_i_128, row_i_128);
                __m128d vec4           = _mm_setr_pd(0.0, 0.0);
                vec3                   = _mm_hadd_pd(vec3, vec4);

                // add the result to sum1
                sum1_128                = _mm_add_pd(sum1_128, vec3);



            }

        Complex16* sum = (Complex16*)&sum1_128;
        Complex16* row_i_c = (Complex16*)row_i;
        row_i_c[idx] = sqrt(row_i_c[idx] - *sum);
        determinant = determinant * row_i_c[idx];

        row_i = row_i + 2*matrix.stride;
    }

/*
    // The AVX code above is equivalent to this code:

    // storing in the same memory the results of the algorithm
    size_t n = matrix.cols;
    // Decomposing a matrix into lower triangular matrices
    for (int i = reuse_index; i < n; i++) {
        Complex16* row_i = matrix.get_data()+i*matrix.stride;

        Complex16* row_j = matrix.get_data() + reuse_index*matrix.stride;
        Complex16* row_j2 = row_j + matrix.stride;

        for (int j = reuse_index; j < i-1; j=j+2) {

            Complex16 sum = 0;
            Complex16 sum2 = 0;
            // Evaluating L(i, j) using L(j, j)
            // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
            for (int k = 0; k < j-1; k=k+2){
                int k2 = k+1;
                sum += mult_a_bconj( row_i[k], row_j[k]) + mult_a_bconj( row_i[k2], row_j[k2]);
                sum2 += mult_a_bconj( row_i[k], row_j2[k]) + mult_a_bconj( row_i[k2], row_j2[k2]);
            }

            if (j%2 == 1) {

                int k = j-1;

                sum += mult_a_bconj( row_i[k], row_j[k]);
                sum2 += mult_a_bconj( row_i[k], row_j2[k]);

            }

            row_i[j] = (row_i[j] - sum) / row_j[j];

            sum2 += mult_a_bconj( row_i[j], row_j2[j]);
            row_i[j+1] = (row_i[j+1] - sum2) / row_j2[j+1];

            row_j = row_j + 2*matrix.stride;
            row_j2 = row_j2 + 2*matrix.stride;

        }

        if ( i%2 == 1) {
            int j = i-1;

            row_j = matrix.get_data() + j * matrix.stride;

            Complex16 sum = 0;
            // Evaluating L(i, j) using L(j, j)
            // L_{i,j}=\frac{1}{L_{j,j}}(A_{i,j}-\sum_{k=0}^{j-1}L_{i,k}L_{j,k}^*)
            for (int k = 0; k < j; k++){
                sum += mult_a_bconj( row_i[k], row_j[k]);
            }

            row_i[j] = (row_i[j] - sum) / row_j[j];

#ifdef DEBUG
            if (matrix.isnan()) {

                std::cout << "matrix is NAN" << std::endl;
                matrix.print_matrix();
                exit(-1);
             }
#endif
        }


        Complex16 sum = 0;
        // summation for diagonals
        // L_{j,j}=\sqrt{A_{j,j}-\sum_{k=0}^{j-1}L_{j,k}L_{j,k}^*}
        for (int k = 0; k < i-1; k=k+2){
            sum += mult_a_bconj( row_i[k], row_i[k] ) + mult_a_bconj( row_i[k+1], row_i[k+1] );
        }

        if ( i%2 == 1) {
            sum += mult_a_bconj( row_i[i-1], row_i[i-1] );
        }


        row_i[i] = sqrt(row_i[i] - sum);
        row_i = row_i + matrix.stride;


    }
*/
    return;

}


} // PIC
