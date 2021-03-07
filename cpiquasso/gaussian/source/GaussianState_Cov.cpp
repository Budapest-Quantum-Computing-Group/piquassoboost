#include <iostream>
#include "GaussianState_Cov.h"
#include <memory.h>
#include <immintrin.h>
#include "tbb/tbb.h"


namespace pic {

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GaussianState_Cov::GaussianState_Cov() {}



/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix
@param m_in The displacement of the Gaussian state
@param repr_in The representation type (see enumeration representation)
@return Returns with the instance of the class.
*/
GaussianState_Cov::GaussianState_Cov( matrix &covariance_matrix_in, matrix &m_in, representation repr_in) {

    Update_covariance_matrix( covariance_matrix_in );
    Update_m(m_in);

    repr = repr_in;
}



/**
@brief Constructor of the class.
@param covariance_matrix_in The covariance matrix (the displacements are set to zeros)
@param m_in The displacement of the Gaussian state
@param repr The representation type (see enumeration representation)
@return Returns with the instance of the class.
*/
GaussianState_Cov::GaussianState_Cov( matrix &covariance_matrix_in, representation repr_in) {

    Update_covariance_matrix( covariance_matrix_in );

    // set the displacement to zero
    m = matrix(1,covariance_matrix_in.rows);
    memset(m.get_data(), 0, m.size()*sizeof(Complex16));

    repr = repr_in;

}



/**
@brief Call to get a reduced Gaussian state (i.e. the gaussian state represented by a subset of modes of the original gaussian state)
@param modes An instance of PicState_int64 containing the modes to be extracted from the original gaussian state
@return Returns with the reduced Gaussian state
*/
GaussianState_Cov
GaussianState_Cov::getReducedGaussianState( PicState_int64 &modes ) {

    size_t total_number_of_modes = covariance_matrix.rows/2;

    if (total_number_of_modes == 0) {
        std::cout << "There is no covariance matrix to be reduced. Exiting" << std::endl;
        exit(-1);
    }


    // the number of modes to be extracted
    size_t number_of_modes = modes.size();
    if (number_of_modes == total_number_of_modes) {
        return GaussianState_Cov(covariance_matrix, m, repr);
    }
    else if ( number_of_modes >= total_number_of_modes) {
        std::cout << "The number of modes to be extracted is larger than the posibble number of modes. Exiting" << std::endl;
        exit(-1);
    }



    // allocate data for the reduced covariance matrix
    matrix covariance_matrix_reduced(number_of_modes*2, number_of_modes*2);  // the size of the covariance matrix must be the double of the number of modes
    Complex16* covariance_matrix_reduced_data = covariance_matrix_reduced.get_data();
    Complex16* covariance_matrix_data = covariance_matrix.get_data();

    // allocate data for the reduced displacement
    matrix m_reduced(1, number_of_modes*2);
    Complex16* m_reduced_data = m_reduced.get_data();
    Complex16* m_data = m.get_data();


    size_t mode_idx = 0;
    size_t col_range = 1;
    // loop over the col indices to be transformed (indices are stored in attribute modes)
    while (true) {

        // condition to exit the loop: if there are no further columns then we exit the loop
        if ( mode_idx >= number_of_modes) {
            break;
        }

        // determine contiguous memory slices (column indices) to be extracted
        while (true) {

            // condition to exit the loop: if the difference of successive indices is greater than 1, the end of the contiguous memory slice is determined
            if ( mode_idx+col_range >= number_of_modes || modes[mode_idx+col_range] - modes[mode_idx+col_range-1] != 1 ) {
                break;
            }
            else {
                if (mode_idx+col_range+1 >= number_of_modes) {
                    break;
                }
                col_range = col_range + 1;

            }

        }

        // the column index in the matrix from we are bout the extract columns
        size_t col_idx = modes[mode_idx];

        // row-wise loop to extract the q quadrature columns from the covariance matrix
        for(size_t mode_row_idx=0; mode_row_idx<number_of_modes; mode_row_idx++) {

            // col-wise extraction of the q quadratures from the covariance matrix
            size_t cov_reduced_offset = mode_idx*covariance_matrix_reduced.stride + mode_idx;
            size_t cov_offset = modes[mode_idx]*covariance_matrix.stride + col_idx;
            memcpy(covariance_matrix_reduced_data + cov_reduced_offset, covariance_matrix_data + cov_offset , col_range*sizeof(Complex16));

            // col-wise extraction of the p quadratures from the covariance matrix
            cov_reduced_offset = cov_reduced_offset + number_of_modes;
            cov_offset = cov_offset + total_number_of_modes;
            memcpy(covariance_matrix_reduced_data + cov_reduced_offset, covariance_matrix_data + cov_offset , col_range*sizeof(Complex16));

        }


        // row-wise loop to extract the p quadrature columns from the covariance matrix
        for(size_t mode_row_idx=0; mode_row_idx<number_of_modes; mode_row_idx++) {

            // col-wise extraction of the q quadratures from the covariance matrix
            size_t cov_reduced_offset = (mode_idx+number_of_modes)*covariance_matrix_reduced.stride + mode_idx;
            size_t cov_offset = (modes[mode_idx]+total_number_of_modes)*covariance_matrix.stride + col_idx;
            memcpy(covariance_matrix_reduced_data + cov_reduced_offset, covariance_matrix_data + cov_offset , col_range*sizeof(Complex16));

            // col-wise extraction of the p quadratures from the covariance matrix
            cov_reduced_offset = cov_reduced_offset + number_of_modes;
            cov_offset = cov_offset + total_number_of_modes;
            memcpy(covariance_matrix_reduced_data + cov_reduced_offset, covariance_matrix_data + cov_offset , col_range*sizeof(Complex16));
        }

        // extract modes from the displacement
        memcpy(m_reduced_data + mode_idx, m_data + col_idx, col_range*sizeof(Complex16)); // q quadratires
        memcpy(m_reduced_data + mode_idx + number_of_modes, m_data + col_idx + total_number_of_modes, col_range*sizeof(Complex16)); // p quadratures

        mode_idx = mode_idx + col_range;
        col_range = 1;

    }


    // creating the reduced Gaussian state
    GaussianState_Cov ret(covariance_matrix_reduced, m_reduced, repr);

    //m_reduced.print_matrix();
    //covariance_matrix_reduced.print_matrix();

    return ret;


}




/**
@brief Call to convert the representation of the Gaussian state into complex amplitude representation, so the
displacement would be the expectation value \f$ m = \langle \hat{\xi}_i \rangle_{\rho} \f$ and the covariance matrix
\f$ covariance_matrix = \langle \hat{\xi}_i\hat{\xi}_j \rangle_{\rho}  - m_im_j\f$ .
*/
void
GaussianState_Cov::ConvertToComplexAmplitudes() {


    if (repr==fock_space) {
        return;
    }

    // get the mean values of the creation/annihilation operators from the quadrature displacement
    size_t total_number_of_modes = m.cols/2;
    matrix displacement_a(1, m.cols);

    matrix q(m.get_data(), 1, total_number_of_modes);
    matrix p(m.get_data()+total_number_of_modes, 1, total_number_of_modes);
    for (size_t idx=0; idx < total_number_of_modes; idx++) {

        // set the expectation values for a_1, a_2, .... a_N
        displacement_a[idx] = (q[idx] + Complex16(0.0,1.0)*p[idx])/sqrt(2);

        // set the expectation values for a^\dagger_1, a^\dagger_2, .... a^\dagger_N
        displacement_a[idx+total_number_of_modes] = std::conj(displacement_a[idx]);
    }


    //get the representation of the covariance matrix in the creation/annihilation operator basis
    total_number_of_modes = covariance_matrix.rows/2;

    // get the matrices of the quadratures with strides
    matrix qq( /* pointer to the origin of the data = */ covariance_matrix.get_data(),
                /* number of rows = */ total_number_of_modes,
                /* number of columns = */ total_number_of_modes,
                /* stride = */ 2*total_number_of_modes);

    matrix pp( /* pointer to the origin of the data = */ covariance_matrix.get_data()+total_number_of_modes*covariance_matrix.stride + total_number_of_modes,
                /* number of rows = */ total_number_of_modes,
                /* number of columns = */ total_number_of_modes,
                /* stride = */  2*total_number_of_modes);


    matrix pq( /* pointer to the origin of the data = */ covariance_matrix.get_data()+total_number_of_modes*covariance_matrix.stride,
               /* number of rows = */ total_number_of_modes,
               /* number of columns = */ total_number_of_modes,
               /* stride = */ 2*total_number_of_modes);

    matrix qp( /* pointer to the origin of the data = */ covariance_matrix.get_data()+total_number_of_modes,
               /* number of rows = */  total_number_of_modes,
               /* number of columns = */ total_number_of_modes,
               /* stride = */ 2*total_number_of_modes);

    // now calculate the \Sigma = covariance matrix in the creation/annihilation operator basis (see Eq. (1) in arXiv 2010.15595v3)
    matrix covariance_matrix_a(covariance_matrix.rows, covariance_matrix.cols);

    // construct references to the submatrices of covariance_matrix_a =
    // [ a_i * a^+_j, a_i * a_j;
    // a^+_i * a^+_j, a^+_i * a_j ]

    // a_i * a^+_j
    matrix a_i_ad_j( /* pointer to the origin of the data = */ covariance_matrix_a.get_data(),
                          /* number of rows = */ total_number_of_modes,
                          /* number of columns = */ total_number_of_modes,
                          /* stride = */ 2*total_number_of_modes);


    // a^+_i * a_j
    matrix ad_i_a_j( /* pointer to the origin of the data = */ covariance_matrix_a.get_data()+total_number_of_modes*covariance_matrix.stride + total_number_of_modes,
                               /* number of rows = */ total_number_of_modes,
                               /* number of columns = */ total_number_of_modes,
                               /* stride = */  2*total_number_of_modes);


    // a_i * a_j
    matrix a_i_a_j( /* pointer to the origin of the data = */ covariance_matrix_a.get_data()+total_number_of_modes,
                    /* number of rows = */ total_number_of_modes,
                    /* number of columns = */ total_number_of_modes,
                    /* stride = */ 2*total_number_of_modes);

    // a^+_i * a^+_j
    matrix ad_i_ad_j( /* pointer to the origin of the data = */ covariance_matrix_a.get_data()+total_number_of_modes*covariance_matrix.stride,
                         /* number of rows = */  total_number_of_modes,
                         /* number of columns = */ total_number_of_modes,
                         /* stride = */ 2*total_number_of_modes);


/*
    Complex16* qq_data = qq.get_data();
    Complex16* pp_data = pp.get_data();
    Complex16* qp_data = qp.get_data();
    Complex16* pq_data = pq.get_data();


    Complex16* ad_i_a_j_data  = ad_i_a_j.get_data();
    Complex16* a_i_ad_j_data  = a_i_ad_j.get_data();
    Complex16* a_i_a_j_data   = a_i_a_j.get_data();
    Complex16* ad_i_ad_j_data = ad_i_ad_j.get_data();


tbb::tick_count t2 = tbb::tick_count::now();
    // calculate \Sigma = covariance_matrix_a following Eq. (1) in arXiv 2010.15595v3
    for (size_t row_idx = 0; row_idx<total_number_of_modes; row_idx++) {

        size_t row_offset = row_idx*2*total_number_of_modes; // the stride of the submatrices is 2*total_number_of_modes

        for (size_t col_idx = 0; col_idx<total_number_of_modes; col_idx++) {

            size_t idx = row_offset + col_idx;

            a_i_ad_j_data[idx]  = (qq_data[idx] + pp_data[idx] + Complex16(0.0,1.0) * (pq_data[idx] - qp_data[idx]))/2.0;
            ad_i_a_j_data[idx]  = (qq_data[idx] + pp_data[idx] - Complex16(0.0,1.0) * (pq_data[idx] - qp_data[idx]))/2.0;
            a_i_a_j_data[idx]   = (qq_data[idx] - pp_data[idx] + Complex16(0.0,1.0) * (pq_data[idx] + qp_data[idx]))/2.0;
            ad_i_ad_j_data[idx] = (qq_data[idx] - pp_data[idx] - Complex16(0.0,1.0) * (pq_data[idx] + qp_data[idx]))/2.0;

        }


    }
tbb::tick_count t3 = tbb::tick_count::now();

covariance_matrix_a.print_matrix();
*/


    double* qq_data_d = (double*)qq.get_data();
    double* pp_data_d = (double*)pp.get_data();
    double* qp_data_d = (double*)qp.get_data();
    double* pq_data_d = (double*)pq.get_data();

    double* ad_i_a_j_data_d  = (double*)ad_i_a_j.get_data();
    double* a_i_ad_j_data_d  = (double*)a_i_ad_j.get_data();
    double* a_i_a_j_data_d   = (double*)a_i_a_j.get_data();
    double* ad_i_ad_j_data_d = (double*)ad_i_ad_j.get_data();

/*
tbb::tick_count t0 = tbb::tick_count::now();
*/

    __m256d neg = _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);
    __m256d half = _mm256_setr_pd(0.5, 0.5, 0.5, 0.5);

    // calculate \Sigma = covariance_matrix_a following Eq. (1) in arXiv 2010.15595v3
    for (size_t row_idx = 0; row_idx<total_number_of_modes; row_idx++) {

        size_t row_offset = row_idx*2*total_number_of_modes; // the stride of the submatrices is 2*total_number_of_modes

        for (size_t col_idx = 0; col_idx<total_number_of_modes; col_idx=col_idx+2) {

            size_t idx = 2*(row_offset + col_idx); // each 2 double is one complex

            __m256d qq_vec = _mm256_loadu_pd(qq_data_d+idx);
            __m256d pp_vec = _mm256_loadu_pd(pp_data_d+idx);
            __m256d pq_vec = _mm256_loadu_pd(pq_data_d+idx);
            __m256d qp_vec = _mm256_loadu_pd(qp_data_d+idx);

            //a_i_ad_j_data[idx]  = (qq + pp + Complex16(0.0,1.0) * (pq - qp))/2.0;
            __m256d qq_plus_pp = _mm256_add_pd( qq_vec, pp_vec );
            __m256d pq_minus_qp = _mm256_sub_pd( pq_vec, qp_vec );
            pq_minus_qp = _mm256_permute_pd(pq_minus_qp, 0x5); // Switch the real and imaginary elements of pq_minus_qp
            pq_minus_qp = _mm256_mul_pd(pq_minus_qp, neg); // Negate the real elements of pq_minus_qp

            __m256d res = _mm256_add_pd( qq_plus_pp, pq_minus_qp );
            res = _mm256_mul_pd(res, half); // divide result by 2
            _mm256_storeu_pd(a_i_ad_j_data_d + idx, res);


            //ad_i_a_j_data[idx]  = (qq + pp - Complex16(0.0,1.0) * (pq - qp))/2.0;
            res = _mm256_sub_pd( qq_plus_pp, pq_minus_qp );
            res = _mm256_mul_pd(res, half); // divide result by 2
            _mm256_storeu_pd(ad_i_a_j_data_d + idx, res); // store the result



            //a_i_a_j   = (qq - pp + Complex16(0.0,1.0) * (pq + qp))/2.0;
            __m256d qq_minus_pp = _mm256_sub_pd( qq_vec, pp_vec );
            __m256d pq_plus_qp = _mm256_add_pd( pq_vec, qp_vec );
            pq_plus_qp = _mm256_permute_pd(pq_plus_qp, 0x5); // Switch the real and imaginary elements of pq_plus_qp
            pq_plus_qp = _mm256_mul_pd(pq_plus_qp, neg); // Negate the real elements of pq_plus_qp


            res = _mm256_add_pd( qq_minus_pp, pq_plus_qp );
            res = _mm256_mul_pd(res, half); // divide result by 2
            _mm256_storeu_pd(a_i_a_j_data_d + idx, res);


            //ad_i_ad_j = (qq - pp - Complex16(0.0,1.0) * (pq + qp))/2.0;
            res = _mm256_sub_pd( qq_minus_pp, pq_plus_qp );
            res = _mm256_mul_pd(res, half); // divide result by 2
            _mm256_storeu_pd(ad_i_ad_j_data_d + idx, res); // store the result


        }


        if ( total_number_of_modes % 2 == 1 ) {

            size_t idx = row_offset + total_number_of_modes;

            Complex16 qq_plus_pp = qq[idx] + pp[idx];
            Complex16 pq_minus_qp = pq[idx] - qp[idx];

            a_i_ad_j[idx]  = (qq_plus_pp + Complex16(0.0,1.0) * pq_minus_qp)/2.0;
            ad_i_a_j[idx]  = (qq_plus_pp - Complex16(0.0,1.0) * pq_minus_qp)/2.0;

            Complex16 qq_minus_pp = qq[idx] - pp[idx];
            Complex16 pq_plus_qp = pq[idx] + qp[idx];

            a_i_a_j[idx]   = (qq_minus_pp + Complex16(0.0,1.0) * pq_plus_qp)/2.0;
            ad_i_ad_j[idx] = (qq_minus_pp - Complex16(0.0,1.0) * pq_plus_qp)/2.0;

/*
            a_i_ad_j_data[idx]  = (qq_data[idx] + pp_data[idx] + Complex16(0.0,1.0) * (pq_data[idx] - qp_data[idx]))/2.0;
            ad_i_a_j_data[idx]  = (qq_data[idx] + pp_data[idx] - Complex16(0.0,1.0) * (pq_data[idx] - qp_data[idx]))/2.0;
            a_i_a_j_data[idx]   = (qq_data[idx] - pp_data[idx] + Complex16(0.0,1.0) * (pq_data[idx] + qp_data[idx]))/2.0;
            ad_i_ad_j_data[idx] = (qq_data[idx] - pp_data[idx] - Complex16(0.0,1.0) * (pq_data[idx] + qp_data[idx]))/2.0;
*/
        }


    }
/*
tbb::tick_count t1 = tbb::tick_count::now();

covariance_matrix_a.print_matrix();
*/

/*
std::cout<< (t1-t0).seconds() << " " << (t3-t2).seconds() << " " << (t3-t2).seconds()/(t1-t0).seconds() << std::endl;
exit(-1);
*/

    // strore the calculated covariance matrix
    covariance_matrix = covariance_matrix_a;


    // indicate that the representation was converted to Fock-space
    repr = fock_space;


}





/**
@brief Call to update the memory address of the vector containing the displacements
@param m_in The new displacement vector
*/
void
GaussianState_Cov::Update_m(matrix &m_in) {

    m = m_in;

}


/**
@brief Call to update the memory address of the matrix stored in covariance_matrix
@param covariance_matrix_in The covariance matrix describing the gaussian state
*/
void
GaussianState_Cov::Update_covariance_matrix( matrix &covariance_matrix_in ) {

    covariance_matrix = covariance_matrix_in;

}


/**
@brief Call to get the covariance matrix
@return Returns with a matrix instance containing the covariance matrix..
*/
matrix
GaussianState_Cov::get_covariance_matrix() {

    return covariance_matrix.copy();

}




/**
@brief Call to get the representation type of the Gaussian state.
@return Returns with the representation type of the Gaussian state.
*/
representation
GaussianState_Cov::get_representation() {

    return repr;

}



} // PIC
