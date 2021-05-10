#include "get_reflection_vector_AVX.h"


namespace pic {

/**
/@brief Determine the reflection vector for Householder transformation used in the upper Hessenberg transformation algorithm
@param input The strided input vector constructed from the k-th column of the matrix on which the Hessenberg transformation should be applied
@param norm_v_sqr The squared norm of the created reflection matrix that is returned by reference
@return Returns with the calculated reflection vector
 */
matrix
get_reflection_vector_AVX(matrix &input, double &norm_v_sqr) {


  __m256d norm_v_sqr_vec_256 = _mm256_setr_pd(0.0, 0.0, 0.0, 0.0);
  double* input_data = (double*)input.get_data();
  matrix reflect_vector(input.rows,1);
  double* reflect_vector_data = (double*)reflect_vector.get_data();

  size_t sizev = reflect_vector.size();

  for (size_t idx = 0; idx < 2*(sizev-1); idx=idx+4) {

      __m256d element_vec;
      element_vec =  _mm256_insertf128_pd(element_vec, _mm_load_pd(input_data), 0);
      input_data = input_data + 2*input.stride;
      element_vec =  _mm256_insertf128_pd(element_vec, _mm_load_pd(input_data), 1);

      // store data to reflection vector
      _mm256_storeu_pd(reflect_vector_data+idx, element_vec);

      // calculate the square of the elements
      element_vec = _mm256_mul_pd(element_vec, element_vec);

      // add the element magnitudes to norm_v_sqr_vec
      norm_v_sqr_vec_256 = _mm256_add_pd(norm_v_sqr_vec_256, element_vec);

      input_data = input_data + 2*input.stride;

  }

  __m128d norm_v_sqr_vec_128 = _mm_add_pd(_mm256_castpd256_pd128(norm_v_sqr_vec_256), _mm256_extractf128_pd(norm_v_sqr_vec_256, 1));

  if ( reflect_vector.rows % 2 == 1) {
      __m128d element_vec = _mm_load_pd(input_data);

      // store data to reflection vector
      _mm_storeu_pd(reflect_vector_data+2*(reflect_vector.rows-1), element_vec);

      // calculate the square of the elements
      element_vec = _mm_mul_pd(element_vec, element_vec);

      // add the element magnitudes to norm_v_sqr_vec
      norm_v_sqr_vec_128 = _mm_add_pd(norm_v_sqr_vec_128, element_vec);
  }


  __m128d norm_v_sqr_vec = _mm_hadd_pd(norm_v_sqr_vec_128, norm_v_sqr_vec_128);
  norm_v_sqr = norm_v_sqr_vec[0];

  //__m128d sigma_vec = _mm_sqrt_pd(norm_v_sqr_vec);
  double sigma = sqrt(norm_v_sqr);//sigma_vec[0];

  double abs_val = std::sqrt( reflect_vector[0].real()*reflect_vector[0].real() + reflect_vector[0].imag()*reflect_vector[0].imag() );
  norm_v_sqr = 2*(norm_v_sqr + abs_val*sigma);

  if (abs_val != 0.0){
      //double angle = std::arg(reflect_vector[0]); // sigma *= (reflect_vector[0] / std::abs(reflect_vector[0]));
      auto addend = reflect_vector[0]/abs_val*sigma;

      double* pfirst_element = (double*)&reflect_vector[0];
      double* paddend = (double*)&addend;

      __m128d first_element_vec = _mm_loadu_pd(pfirst_element);
      __m128d addend_vec = _mm_loadu_pd(paddend);

      first_element_vec = _mm_add_pd( first_element_vec, addend_vec);

      _mm_storeu_pd(pfirst_element, first_element_vec);
  }
  else {
      reflect_vector[0].real( reflect_vector[0].real() + sigma );
  }



    if (norm_v_sqr == 0.0)
        return reflect_vector;

    // normalize the reflection matrix
    double norm_v = std::sqrt(norm_v_sqr);
    __m512d norm_vec_512 = _mm512_set_pd( norm_v, norm_v, norm_v, norm_v, norm_v, norm_v, norm_v, norm_v );
    __m256d norm_vec = _mm512_castpd512_pd256(norm_vec_512);//_mm256_set_pd( norm_v, norm_v, norm_v, norm_v );
    reflect_vector_data = (double*)reflect_vector.get_data();

    if ( sizev > 3 ) {
    for (size_t idx=0; idx<2*(sizev-3); idx=idx+8) {
        __m512d element_vec = _mm512_load_pd(reflect_vector_data+idx);
        element_vec = _mm512_div_pd(element_vec, norm_vec_512);
        _mm512_storeu_pd(reflect_vector_data+idx, element_vec);
    }
}

    reflect_vector_data = (double*)reflect_vector.get_data();
    size_t reminder = sizev % 4;
    if (reminder >= 2) {
        size_t idx = 2*(sizev - reminder);

        __m256d element_vec = _mm256_load_pd(reflect_vector_data+idx);
        element_vec = _mm256_div_pd(element_vec, norm_vec);
        _mm256_storeu_pd(reflect_vector_data+idx, element_vec);

        reminder = reminder - 2;
    }

    if (reminder == 1) {
        size_t idx = 2*(sizev - 1);

        __m128d norm_vec_128 = _mm256_castpd256_pd128(norm_vec);
        __m128d element_vec = _mm_load_pd(reflect_vector_data + idx );
        element_vec = _mm_div_pd(element_vec, norm_vec_128);
        _mm_storeu_pd(reflect_vector_data+idx, element_vec);
    }

    norm_v_sqr = 1.0;


    return reflect_vector;

/*
  double sigma(0.0);
  norm_v_sqr = 0.0;
  matrix reflect_vector(input.rows,1);
  for (size_t idx = 0; idx < reflect_vector.size(); idx++) {
      Complex16 &element = input[idx*input.stride];
      reflect_vector[idx] =  element;//mtx[(idx + offset) * mtx_size + offset - 1];
      norm_v_sqr = norm_v_sqr + element.real()*element.real() + element.imag()*element.imag(); //adding the squared magnitude
  }
  sigma = sqrt(norm_v_sqr);



  double abs_val = std::sqrt( reflect_vector[0].real()*reflect_vector[0].real() + reflect_vector[0].imag()*reflect_vector[0].imag() );
  norm_v_sqr = 2*(norm_v_sqr + abs_val*sigma);

  if (abs_val != 0.0){
      //double angle = std::arg(reflect_vector[0]); // sigma *= (reflect_vector[0] / std::abs(reflect_vector[0]));
      auto addend = reflect_vector[0]/abs_val*sigma;
      reflect_vector[0].real( reflect_vector[0].real() + addend.real());
      reflect_vector[0].imag( reflect_vector[0].imag() + addend.imag());
  }
  else {
      reflect_vector[0].real( reflect_vector[0].real() + sigma );
  }


  if (norm_v_sqr == 0.0)
      return reflect_vector;

  // normalize the reflection matrix
  double norm_v = std::sqrt(norm_v_sqr);
  for (size_t idx=0; idx<reflect_vector.size(); idx++) {
      reflect_vector[idx] = reflect_vector[idx]/norm_v;
  }

  norm_v_sqr = 1.0;

  return reflect_vector;
  */








}


} // PIC
