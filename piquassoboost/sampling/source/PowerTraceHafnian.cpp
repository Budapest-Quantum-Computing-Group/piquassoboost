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

#ifndef LONG_DOUBLE_CUTOFF
#define LONG_DOUBLE_CUTOFF 40
#endif // LONG_DOUBLE_CUTOFF

#include <iostream>
#include "PowerTraceHafnian.h"
#include "PowerTraceHafnianUtilities.hpp"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>

#ifdef __MPI__
#include <mpi.h>
#endif // MPI

/*
tbb::spin_mutex my_mutex;

double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {



/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
template <class small_scalar_type, class scalar_type>
PowerTraceHafnian<small_scalar_type, scalar_type>::PowerTraceHafnian() {

}


/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix for which the hafnian is calculated. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state)
@return Returns with the instance of the class.
*/
template <class small_scalar_type, class scalar_type>
PowerTraceHafnian<small_scalar_type, scalar_type>::PowerTraceHafnian( matrix &mtx_in ) {
    assert(isSymmetric(mtx_in));

    Update_mtx( mtx_in);
}


/**
@brief Default destructor of the class.
*/
template <class small_scalar_type, class scalar_type>
PowerTraceHafnian<small_scalar_type, scalar_type>::~PowerTraceHafnian() {

}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
template <class small_scalar_type, class scalar_type>
Complex16
PowerTraceHafnian<small_scalar_type, scalar_type>::calculate() {

    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return Complex16(1,0);
    }
    else if (mtx.rows % 2 != 0) {
        // the hafnian of odd shaped matrix is 0 by definition
        return Complex16(0.0, 0.0);
    }


    size_t dim_over_2 = mtx.rows / 2;
    unsigned long long permutation_idx_max = power_of_2( (unsigned long long) dim_over_2);

#ifdef __MPI__
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

#ifdef GLYNN
    Complex16 hafnian = calculate(current_rank, world_size, permutation_idx_max/2);
#else
    Complex16 hafnian = calculate(current_rank+1, world_size, permutation_idx_max);
#endif

    // send the calculated partial hafnian to rank 0
    Complex16* partial_hafnians = new Complex16[world_size];

    MPI_Allgather(&hafnian, 2, MPI_DOUBLE, partial_hafnians, 2, MPI_DOUBLE, MPI_COMM_WORLD);

    hafnian = cplx_select_t<scalar_type>(0.0,0.0);
    for (size_t idx=0; idx<world_size; idx++) {
        hafnian = hafnian + partial_hafnians[idx];
    }

    // release memory on the zero rank
    delete partial_hafnians;


    return hafnian;

#else

    unsigned long long current_rank = 0;
    unsigned long long world_size = 1;

#ifdef GLYNN
    Complex16 hafnian = calculate(current_rank, world_size, permutation_idx_max/2);
#else
    Complex16 hafnian = calculate(current_rank+1, world_size, permutation_idx_max);
#endif

    return hafnian;
#endif



}


/**
@brief Call to calculate the hafnian of a complex matrix
@param start_idx The minimal index evaluated in the exponentially large sum (used to divide calculations between MPI processes)
@param step_idx The index step in the exponentially large sum (used to divide calculations between MPI processes)
@param max_idx The maximal indexe valuated in the exponentially large sum (used to divide calculations between MPI processes)
@return Returns with the calculated hafnian
*/
template <class small_scalar_type, class scalar_type>
Complex16
PowerTraceHafnian<small_scalar_type, scalar_type>::calculate(unsigned long long start_idx, unsigned long long step_idx, unsigned long long max_idx ) {
    using complex_type = cplx_select_t<scalar_type>;
    using matrix_type = mtx_select_t<complex_type>;
    if ( mtx.rows != mtx.cols) {
        std::cout << "The input matrix should be square shaped, but matrix with " << mtx.rows << " rows and with " << mtx.cols << " columns was given" << std::endl;
        std::cout << "Returning zero" << std::endl;
        return Complex16(0,0);
    }

    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return Complex16(1,0);
    }
    else if (mtx.rows % 2 != 0) {
        // the hafnian of odd shaped matrix is 0 by definition
        return Complex16(0.0, 0.0);
    }

#if BLAS==0 // undefined BLAS
    int NumThreads = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS==1 // MKL
    int NumThreads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS==2 //OpenBLAS
    int NumThreads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

    size_t dim = mtx.rows;
    size_t dim_over_2 = mtx.rows / 2;
    //unsigned long long permutation_idx_max = power_of_2( (unsigned long long) dim_over_2);


    // thread local storages for the partial hafnians
    tbb::combinable<complex_type> summands{[](){return complex_type(0.0,0.0);}};

    // for cycle over the permutations n/2 according to Eq (3.24) in arXiv 1805.12498
    tbb::parallel_for( start_idx, max_idx, step_idx, [&](unsigned long long permutation_idx) {


        complex_type &summand = summands.local();

/*
    complex_type summand(0.0,0.0);

    for (unsigned long long permutation_idx = 0; permutation_idx < permutation_idx_max; permutation_idx++) {
*/

#ifdef GLYNN
        mtx_select_t<cplx_select_t<small_scalar_type>> B(dim, dim);
        small_scalar_type scale_factor_B = 0.0;
        bool fact = false;
        //also possible to the pairs of rows and pairs of columns, places them on the opposite halves, and reverses these halves 
        for (size_t idx = 0; idx < dim; idx++) {
            size_t row_offset = (idx ^ 1)*mtx.stride;
            size_t swap_row_offset = idx*B.stride;
            for (size_t jdx = 0; jdx < dim_over_2; jdx++) {
                bool neg = (permutation_idx & (1ULL << jdx)) != 0;
                if (idx == jdx) fact ^= neg;
                if (neg) {
                    ::new (&B[swap_row_offset + jdx*2]) cplx_select_t<small_scalar_type>(-mtx[ row_offset + jdx*2].real(), -mtx[ row_offset + jdx*2].imag());
                    ::new (&B[swap_row_offset + jdx*2+1]) cplx_select_t<small_scalar_type>(-mtx[ row_offset + jdx*2+1].real(), -mtx[ row_offset + jdx*2+1].imag());
                } else {
                    ::new (&B[swap_row_offset + jdx*2]) cplx_select_t<small_scalar_type>(mtx[ row_offset + jdx*2].real(), mtx[ row_offset + jdx*2].imag());
                    ::new (&B[swap_row_offset + jdx*2+1]) cplx_select_t<small_scalar_type>(mtx[ row_offset + jdx*2+1].real(), mtx[ row_offset + jdx*2+1].imag());
                }
                scale_factor_B += std::norm(B[swap_row_offset + jdx*2]);
                scale_factor_B += std::norm(B[swap_row_offset + jdx*2+1]);
            }
        }
#else
        // get the binary representation of permutation_idx
        // also get the number of 1's in the representation and their position as 2*i and 2*i+1 in consecutive slots of the vector bin_rep
        std::vector<unsigned char> bin_rep;
        std::vector<unsigned char> positions_of_ones;
        bin_rep.reserve(dim_over_2);
        positions_of_ones.reserve(dim);
        for (int i = 1 << (dim_over_2-1); i > 0; i = i / 2) {
            if (permutation_idx & i) {
                bin_rep.push_back(1);
                positions_of_ones.push_back((bin_rep.size()-1)*2);
                positions_of_ones.push_back((bin_rep.size()-1)*2+1);
            }
            else {
                bin_rep.push_back(0);
            }
        }
        size_t number_of_ones = positions_of_ones.size();


        // matrix B corresponds to A^(Z), i.e. to the square matrix constructed from
        // the elements of mtx=A indexed by the rows and colums, where the binary representation of
        // permutation_idx was 1
        // for details see the text below Eq.(3.20) of arXiv 1805.12498
        mtx_select_t<cplx_select_t<small_scalar_type>> B(number_of_ones, number_of_ones);
        small_scalar_type scale_factor_B = 0.0;
        for (size_t idx = 0; idx < number_of_ones; idx++) {
            size_t row_offset = (positions_of_ones[idx] ^ 1)*mtx.stride;
            for (size_t jdx = 0; jdx < number_of_ones; jdx++) {
                cplx_select_t<small_scalar_type> element(mtx[ row_offset + positions_of_ones[jdx]].real(), mtx[ row_offset + positions_of_ones[jdx]].imag());                
                ::new (&B[idx*number_of_ones + jdx]) cplx_select_t<small_scalar_type>(element);
                scale_factor_B = scale_factor_B + element.real()*element.real() + element.imag()*element.imag();
            }
        }
        // fact corresponds to the (-1)^{(n/2) - |Z|} prefactor from Eq (3.24) in arXiv 1805.12498
        bool fact = ((dim_over_2 - number_of_ones/2) % 2);
#endif


        // scale matrix B -- when matrix elements of B are scaled, larger part of the computations can be kept in double precision
        if ( scale_factor_B < 1e-8 ) {
            scale_factor_B = 1.0;
        }
        else {
            scale_factor_B = std::sqrt(scale_factor_B/2)/B.size();
            for (size_t idx=0; idx<B.size(); idx++) {
                B[idx] = B[idx]*scale_factor_B;
            }
        }

        // calculating Tr(B^j) for all j's that are 1<=j<=dim/2
        // this is needed to calculate f_G(Z) defined in Eq. (3.17b) of arXiv 1805.12498
        matrix_type traces(dim_over_2, 1);
#ifndef GLYNN
        if (number_of_ones != 0) {
#endif
            CalcPowerTraces<small_scalar_type, scalar_type>(B, dim_over_2, traces);
#ifndef GLYNN
        }
        else{
            // in case we have no 1's in the binary representation of permutation_idx we get zeros
            // this occurs once during the calculations
            //memset( traces.get_data(), 0.0, traces.rows*traces.cols*sizeof(complex_type));
            std::uninitialized_fill_n(traces.get_data(), traces.rows*traces.cols, complex_type(0.0, 0.0));
        }
#endif


        // auxiliary data arrays to evaluate the second part of Eqs (3.24) and (3.21) in arXiv 1805.12498
        matrix_type aux0(dim_over_2 + 1, 1);
        matrix_type aux1(dim_over_2 + 1, 1);
        //memset( aux0.get_data(), 0.0, (dim_over_2 + 1)*sizeof(complex_type));
        //memset( aux1.get_data(), 0.0, (dim_over_2 + 1)*sizeof(complex_type));
        std::uninitialized_fill_n(aux0.get_data(), dim_over_2 + 1, complex_type(0.0, 0.0));
        std::uninitialized_fill_n(aux1.get_data(), dim_over_2 + 1, complex_type(0.0, 0.0));        

        aux0[0] = 1.0;
        // pointers to the auxiliary data arrays
        complex_type *p_aux0=NULL, *p_aux1=NULL;

        small_scalar_type inverse_scale_factor = 1/scale_factor_B; // the (1/scale_factor_B)^idx power of the local scaling factor of matrix B to scale the power trace
        for (size_t idx = 1; idx <= dim_over_2; idx++) {


            complex_type factor = traces[idx - 1] * inverse_scale_factor / (2.0 * idx);

            // refresh the scaling factor
            inverse_scale_factor = inverse_scale_factor/scale_factor_B;

            complex_type powfactor(1.0,0.0);



            if (idx%2 == 1) {
                p_aux0 = aux0.get_data();
                p_aux1 = aux1.get_data();
            }
            else {
                p_aux0 = aux1.get_data();
                p_aux1 = aux0.get_data();
            }

            //memcpy(p_aux1, p_aux0, (dim_over_2+1)*sizeof(complex_type) );
            std::copy_n(p_aux0, dim_over_2+1, p_aux1);

            for (size_t jdx = 1; jdx <= (dim / (2 * idx)); jdx++) {
                powfactor = powfactor * factor / ((scalar_type)jdx);

                for (size_t kdx = idx * jdx + 1; kdx <= dim_over_2 + 1; kdx++) {
                    p_aux1[kdx-1] += p_aux0[kdx-idx*jdx - 1]*powfactor;
                }



            }



        }


        if (fact) {
            summand -= p_aux1[dim_over_2];
//std::cout << -p_aux1[dim_over_2] << std::endl;
        }
        else {
            summand += p_aux1[dim_over_2];
//std::cout << p_aux1[dim_over_2] << std::endl;
        }

        for (size_t n = aux0.size(); n > 0; --n) aux0[n-1].~complex_type();
        for (size_t n = aux1.size(); n > 0; --n) aux1[n-1].~complex_type();
        for (size_t n = traces.size(); n > 0; --n) traces[n-1].~complex_type();
        for (size_t n = B.size(); n > 0; --n) B[n-1].~cplx_select_t<small_scalar_type>();


    });

    // the resulting Hafnian of matrix mat
    complex_type res(0,0);
    summands.combine_each([&res](const complex_type& a) {
        res += a;
    });

    //Complex16 res = summand;


#if BLAS==0 // undefined BLAS
    omp_set_num_threads(NumThreads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(NumThreads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(NumThreads);
#endif

    // scale the result by the appropriate factor according to Eq (2.11) of in arXiv 1805.12498
    res *= pow(scale_factor, dim_over_2);
#ifdef GLYNN
    res /= (1ULL << (dim_over_2-1));
#endif
    return Complex16(res.real(), res.imag() );
}


/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in A symmetric matrix for which the hafnian is calculated. (For example a covariance matrix of the Gaussian state.)
*/
template <class small_scalar_type, class scalar_type>
void
PowerTraceHafnian<small_scalar_type, scalar_type>::Update_mtx( matrix &mtx_in) {

    mtx_orig = mtx_in;

    // scale the input matrix according to according to Eq (2.11) of in arXiv 1805.12498
    ScaleMatrix();


}


/**
@brief Call to scale the input matrix according to according to Eq (2.11) of in arXiv 1805.12498
*/
template <class small_scalar_type, class scalar_type>
void
PowerTraceHafnian<small_scalar_type, scalar_type>::ScaleMatrix() {

    // scale the matrix to have the mean magnitudes matrix elements equal to one.
    if ( mtx_orig.rows <= 10) {
        mtx = mtx_orig;
        scale_factor = 1.0;
    }
    else {

        // determine the scale factor
        scale_factor = 0.0;
        for (size_t idx=0; idx<mtx_orig.size(); idx++) {
            scale_factor = scale_factor + std::sqrt( mtx_orig[idx].real()*mtx_orig[idx].real() + mtx_orig[idx].imag()*mtx_orig[idx].imag() );
        }
        scale_factor = scale_factor/mtx_orig.size()/std::sqrt(2);
        //scale_factor = scale_factor*mtx_orig.rows;

        mtx = mtx_orig.copy();

        small_scalar_type inverse_scale_factor = 1/scale_factor;

        // scaling the matrix elements
        for (size_t idx=0; idx<mtx_orig.size(); idx++) {
            mtx[idx] = mtx[idx]*inverse_scale_factor;
        }

    }

}

template class PowerTraceHafnian<double, double>;
template class PowerTraceHafnian<double, long double>;
template class PowerTraceHafnian<long double, long double>;

template <>
Complex16
PowerTraceHafnian<RationalInf, RationalInf>::calculate(unsigned long long start_idx, unsigned long long step_idx, unsigned long long max_idx ) {
    using complex_type = cplx_select_t<RationalInf>;
    using matrix_type = mtx_select_t<complex_type>;
    if ( mtx.rows != mtx.cols) {
        std::cout << "The input matrix should be square shaped, but matrix with " << mtx.rows << " rows and with " << mtx.cols << " columns was given" << std::endl;
        std::cout << "Returning zero" << std::endl;
        return Complex16(0,0);
    }

    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return Complex16(1,0);
    }
    else if (mtx.rows % 2 != 0) {
        // the hafnian of odd shaped matrix is 0 by definition
        return Complex16(0.0, 0.0);
    }

#if BLAS==0 // undefined BLAS
    int NumThreads = omp_get_max_threads();
    omp_set_num_threads(1);
#elif BLAS==1 // MKL
    int NumThreads = mkl_get_max_threads();
    MKL_Set_Num_Threads(1);
#elif BLAS==2 //OpenBLAS
    int NumThreads = openblas_get_num_threads();
    openblas_set_num_threads(1);
#endif

    size_t dim = mtx.rows;
    size_t dim_over_2 = mtx.rows / 2;
    //unsigned long long permutation_idx_max = power_of_2( (unsigned long long) dim_over_2);


    // thread local storages for the partial hafnians
    tbb::combinable<complex_type> summands{[](){return complex_type(0.0,0.0);}};

    // for cycle over the permutations n/2 according to Eq (3.24) in arXiv 1805.12498
    tbb::parallel_for( start_idx, max_idx, step_idx, [&](unsigned long long permutation_idx) {


        complex_type &summand = summands.local();

/*
    complex_type summand(0.0,0.0);

    for (unsigned long long permutation_idx = 0; permutation_idx < permutation_idx_max; permutation_idx++) {
*/


#ifdef GLYNN
        matrix_type B(dim, dim);
        bool fact = false;
        //also possible to the pairs of rows and pairs of columns, places them on the opposite halves, and reverses these halves 
        for (size_t idx = 0; idx < dim; idx++) {
            size_t row_offset = (idx ^ 1)*mtx.stride;
            size_t swap_row_offset = idx*B.stride;
            for (size_t jdx = 0; jdx < dim_over_2; jdx++) {
                bool neg = (permutation_idx & (1ULL << jdx)) != 0;
                if (idx == jdx) fact ^= neg;
                if (neg) {
                    ::new (&B[swap_row_offset + jdx*2]) complex_type(-mtx[ row_offset + jdx*2].real(), -mtx[ row_offset + jdx*2].imag());
                    ::new (&B[swap_row_offset + jdx*2+1]) complex_type(-mtx[ row_offset + jdx*2+1].real(), -mtx[ row_offset + jdx*2+1].imag());                   
                } else {
                    ::new (&B[swap_row_offset + jdx*2]) complex_type(mtx[ row_offset + jdx*2].real(), mtx[ row_offset + jdx*2].imag());
                    ::new (&B[swap_row_offset + jdx*2+1]) complex_type(mtx[ row_offset + jdx*2+1].real(), mtx[ row_offset + jdx*2+1].imag());
                }
            }
        }
#else
        // get the binary representation of permutation_idx
        // also get the number of 1's in the representation and their position as 2*i and 2*i+1 in consecutive slots of the vector bin_rep
        std::vector<unsigned char> bin_rep;
        std::vector<unsigned char> positions_of_ones;
        bin_rep.reserve(dim_over_2);
        positions_of_ones.reserve(dim);
        for (int i = 1 << (dim_over_2-1); i > 0; i = i / 2) {
            if (permutation_idx & i) {
                bin_rep.push_back(1);
                positions_of_ones.push_back((bin_rep.size()-1)*2);
                positions_of_ones.push_back((bin_rep.size()-1)*2+1);
            }
            else {
                bin_rep.push_back(0);
            }
        }
        size_t number_of_ones = positions_of_ones.size();


        // matrix B corresponds to A^(Z), i.e. to the square matrix constructed from
        // the elements of mtx=A indexed by the rows and colums, where the binary representation of
        // permutation_idx was 1
        // for details see the text below Eq.(3.20) of arXiv 1805.12498
        matrix_type B(number_of_ones, number_of_ones);
        for (size_t idx = 0; idx < number_of_ones; idx++) {
            size_t row_offset = (positions_of_ones[idx] ^ 1)*mtx.stride;
            for (size_t jdx = 0; jdx < number_of_ones; jdx++) {
                ::new (&B[idx*number_of_ones + jdx]) complex_type(mtx[ row_offset + positions_of_ones[jdx]].real(), mtx[ row_offset + positions_of_ones[jdx]].imag());
            }
        }
        // fact corresponds to the (-1)^{(n/2) - |Z|} prefactor from Eq (3.24) in arXiv 1805.12498
        bool fact = ((dim_over_2 - number_of_ones/2) % 2);
#endif

        // calculating Tr(B^j) for all j's that are 1<=j<=dim/2
        // this is needed to calculate f_G(Z) defined in Eq. (3.17b) of arXiv 1805.12498
        matrix_type traces(dim_over_2, 1);
#ifndef GLYNN
        if (number_of_ones != 0) {
#endif
            CalcPowerTraces<RationalInf, RationalInf>(B, dim_over_2, traces);
#ifndef GLYNN
        }
        else {
            // in case we have no 1's in the binary representation of permutation_idx we get zeros
            // this occurs once during the calculations
            //memset( traces.get_data(), 0.0, traces.rows*traces.cols*sizeof(complex_type));
            std::uninitialized_fill_n(traces.get_data(), traces.rows*traces.cols, complex_type(0.0, 0.0));
        }
#endif

        // auxiliary data arrays to evaluate the second part of Eqs (3.24) and (3.21) in arXiv 1805.12498
        matrix_type aux0(dim_over_2 + 1, 1);
        matrix_type aux1(dim_over_2 + 1, 1);
        //memset( aux0.get_data(), 0.0, (dim_over_2 + 1)*sizeof(complex_type));        
        //memset( aux1.get_data(), 0.0, (dim_over_2 + 1)*sizeof(complex_type));
        std::uninitialized_fill_n(aux0.get_data(), dim_over_2 + 1, complex_type(0.0, 0.0));
        std::uninitialized_fill_n(aux1.get_data(), dim_over_2 + 1, complex_type(0.0, 0.0));
        aux0[0] = 1.0;
        // pointers to the auxiliary data arrays
        complex_type *p_aux0=NULL, *p_aux1=NULL;

        for (size_t idx = 1; idx <= dim_over_2; idx++) {


            complex_type factor = traces[idx - 1] / (2.0 * idx);

            complex_type powfactor(1.0,0.0);



            if (idx%2 == 1) {
                p_aux0 = aux0.get_data();
                p_aux1 = aux1.get_data();
            }
            else {
                p_aux0 = aux1.get_data();
                p_aux1 = aux0.get_data();
            }

            //memcpy(p_aux1, p_aux0, (dim_over_2+1)*sizeof(complex_type) );
            std::copy_n(p_aux0, dim_over_2+1, p_aux1);

            for (size_t jdx = 1; jdx <= (dim / (2 * idx)); jdx++) {
                powfactor *= factor / jdx;

                for (size_t kdx = idx * jdx + 1; kdx <= dim_over_2 + 1; kdx++) {
                    p_aux1[kdx-1] += p_aux0[kdx-idx*jdx - 1]*powfactor;
                }



            }



        }

        if (fact) {
            summand -= p_aux1[dim_over_2];
//std::cout << -p_aux1[dim_over_2] << std::endl;
        }
        else {
            summand += p_aux1[dim_over_2];
//std::cout << p_aux1[dim_over_2] << std::endl;
        }

        for (size_t n = aux0.size(); n > 0; --n) aux0[n-1].~complex_type();
        for (size_t n = aux1.size(); n > 0; --n) aux1[n-1].~complex_type();
        for (size_t n = traces.size(); n > 0; --n) traces[n-1].~complex_type();
        for (size_t n = B.size(); n > 0; --n) B[n-1].~complex_type();

    });

    // the resulting Hafnian of matrix mat
    complex_type res(0,0);
    summands.combine_each([&res](const complex_type& a) {
        res += a;
    });

    //Complex16 res = summand;


#if BLAS==0 // undefined BLAS
    omp_set_num_threads(NumThreads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(NumThreads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(NumThreads);
#endif

#ifdef GLYNN
    res /= (1ULL << (dim_over_2-1));
#endif
    return Complex16(res.real(), res.imag() );
}
template <>
void
PowerTraceHafnian<RationalInf, RationalInf>::ScaleMatrix() {
    mtx = mtx_orig;
    scale_factor = 1.0;
}
template class PowerTraceHafnian<RationalInf, RationalInf>;

} // PIC
