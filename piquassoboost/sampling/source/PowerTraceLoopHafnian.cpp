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
#include "PowerTraceLoopHafnian.h"
#include "PowerTraceHafnianUtilities.hpp"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>

#ifdef __MPI__
#include <mpi.h>
#endif // MPI

/*
static tbb::spin_mutex my_mutex;

static double time_nominator = 0.0;
static double time_nevezo = 0.0;
*/

namespace pic {

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
template <class complex_type>
PowerTraceLoopHafnian<complex_type>::PowerTraceLoopHafnian() : PowerTraceHafnian<complex_type>() {

}

/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix for which the hafnian is calculated. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state.)
@return Returns with the instance of the class.
*/
template <class complex_type>
PowerTraceLoopHafnian<complex_type>::PowerTraceLoopHafnian( matrix &mtx_in ) {
#ifdef DEBUG
    assert(isSymmetric(mtx_in));
#endif
   
    
    if (mtx_in.rows % 2 == 1){
        matrix extended_matrix(mtx_in.rows + 1, mtx_in.cols + 1);
        for (int row_idx = 0; row_idx < mtx_in.rows; row_idx++){
            std::memcpy(
                extended_matrix.get_data() + row_idx * extended_matrix.stride,
                mtx_in.get_data() + row_idx * mtx_in.stride,
                sizeof(Complex16) * mtx_in.cols
            );
            extended_matrix[row_idx * extended_matrix.stride + mtx_in.cols] = 
                Complex16(0.0, 0.0);
        }
        Complex16 *row_last = extended_matrix.get_data() + mtx_in.rows * extended_matrix.stride;
        std::fill(row_last, row_last + mtx_in.cols, Complex16(0.0, 0.0));
        row_last[mtx_in.cols] = Complex16(1.0, 0.0);

        Update_mtx( extended_matrix );
    }else{
        Update_mtx( mtx_in );
    }

}


/**
@brief Call to calculate the hafnian of a complex matrix stored in the instance of the class
@return Returns with the calculated hafnian
*/
template <class complex_type>
Complex16
PowerTraceLoopHafnian<complex_type>::calculate() {

    if (this->mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return Complex16(1,0);
    }
    else if (this->mtx.rows % 2 != 0) {
        // the hafnian of odd shaped matrix is 0 by definition
        return Complex16(0.0, 0.0);
    }

    size_t dim_over_2 = this->mtx.rows / 2;
    unsigned long long permutation_idx_max = power_of_2( (unsigned long long) dim_over_2);

#ifdef __MPI__

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    Complex16 hafnian = calculate(current_rank+1, world_size, permutation_idx_max);

    // send the calculated partial hafnian to rank 0
    Complex16* partial_hafnians = new Complex16[world_size];

    MPI_Allgather(&hafnian, 2, MPI_DOUBLE, partial_hafnians, 2, MPI_DOUBLE, MPI_COMM_WORLD);

    hafnian = Complex16(0.0,0.0);
    for (size_t idx=0; idx<world_size; idx++) {
        hafnian = hafnian + partial_hafnians[idx];
    }

    // release memory on the zero rank
    delete partial_hafnians;


    return hafnian;

#else

    unsigned long long current_rank = 0;
    unsigned long long world_size = 1;

    Complex16 hafnian = calculate(current_rank+1, world_size, permutation_idx_max);

    return hafnian;
#endif



}


/**
@brief Call to calculate the hafnian of a complex matrix stored in the instance of the class
@param start_idx The minimal index evaluated in the exponentially large sum (used to divide calculations between MPI processes)
@param step_idx The index step in the exponentially large sum (used to divide calculations between MPI processes)
@param max_idx The maximal indexe valuated in the exponentially large sum (used to divide calculations between MPI processes)
@return Returns with the calculated hafnian
*/
template <class complex_type>
Complex16
PowerTraceLoopHafnian<complex_type>::calculate(unsigned long long start_idx, unsigned long long step_idx, unsigned long long max_idx ) {


    if ( this->mtx.rows != this->mtx.cols) {
        std::cout << "The input matrix should be square shaped, bu matrix with " << this->mtx.rows << " rows and with " << this->mtx.cols << " columns was given" << std::endl;
        std::cout << "Returning zero" << std::endl;
        return Complex16(0,0);
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

    size_t dim = this->mtx.rows;


    if (this->mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return Complex16(1,0);
    }
    else if (this->mtx.rows % 2 != 0) {
        // the hafnian of odd shaped matrix is 0 by definition
        return Complex16(0.0, 0.0);
    }

    size_t dim_over_2 = this->mtx.rows / 2;

    // for cycle over the permutations n/2 according to Eq (3.24) in arXiv 1805.12498
    tbb::combinable<complex_type> summands{[](){return complex_type(0.0,0.0);}};

    tbb::parallel_for( start_idx, max_idx, step_idx, [&](unsigned long long permutation_idx) {


        complex_type &summand = summands.local();

/*
    complex_type summand(0.0,0.0);

    for (unsigned long long permutation_idx = 0; permutation_idx < permutation_idx_max; permutation_idx++) {
*/



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


        // matrix AZ corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix
        // the elements of mtx=A indexed by the rows and colums, where the binary representation of permutation_idx was 1
        // for details see the text below Eq.(3.20) of arXiv 1805.12498
        // diag_elements corresponds to the diagonal elements of the input  matrix used in the loop correction
        matrix AZ(number_of_ones, number_of_ones);
        matrix diag_elements(1, number_of_ones);
        for (size_t idx = 0; idx < number_of_ones; idx++) {
            size_t row_offset = (positions_of_ones[idx] ^ 1)*this->mtx.stride;
            for (size_t jdx = 0; jdx < number_of_ones; jdx++) {
                AZ[idx*AZ.stride + jdx] = this->mtx[row_offset + ((positions_of_ones[jdx]))];
            }
            diag_elements[idx] = this->mtx[(positions_of_ones[idx])*this->mtx.stride + positions_of_ones[idx]];

        }

        // select the X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
        matrix cx_diag_elements(number_of_ones, 1);
        for (size_t idx = 1; idx < number_of_ones; idx=idx+2) {
            cx_diag_elements[idx] = diag_elements[idx-1];
            cx_diag_elements[idx-1] = diag_elements[idx];
        }


        // calculating Tr(B^j) for all j's that are 1<=j<=dim/2 and loop corrections
        // this is needed to calculate f_G(Z) defined in Eq. (3.17b) of arXiv 1805.12498
        matrix32 traces;
        matrix32 loop_corrections;
        if (number_of_ones != 0) {
            CalcPowerTracesAndLoopCorrections(cx_diag_elements, diag_elements, AZ, dim_over_2, traces, loop_corrections);
        }
        else{
            // in case we have no 1's in the binary representation of permutation_idx we get zeros
            // this occurs once during the calculations
            traces = matrix32(dim_over_2, 1);
            loop_corrections = matrix32(dim_over_2, 1);
            memset( traces.get_data(), 0.0, traces.size()*sizeof(complex_type));
            memset( loop_corrections.get_data(), 0.0, loop_corrections.size()*sizeof(complex_type));
        }


        // fact corresponds to the (-1)^{(n/2) - |Z|} prefactor from Eq (3.24) in arXiv 1805.12498
        bool fact = ((dim_over_2 - number_of_ones/2) % 2);


        // auxiliary data arrays to evaluate the second part of Eqs (3.24) and (3.21) in arXiv 1805.12498
        matrix32 aux0(dim_over_2 + 1, 1);
        matrix32 aux1(dim_over_2 + 1, 1);
        memset( aux0.get_data(), 0.0, (dim_over_2 + 1)*sizeof(complex_type));
        memset( aux1.get_data(), 0.0, (dim_over_2 + 1)*sizeof(complex_type));
        aux0[0] = 1.0;
        // pointers to the auxiliary data arrays
        complex_type *p_aux0=NULL, *p_aux1=NULL;

        for (size_t idx = 1; idx <= dim_over_2; idx++) {


            complex_type factor = traces[idx - 1] / (2.0 * idx) + loop_corrections[idx-1]*0.5;
/*
{
      tbb::spin_mutex::scoped_lock my_lock{my_mutex};
      //traces.print_matrix();
      std::cout << factor << " " << loop_corrections[idx-1]*0.5 << std::endl;
  }
*/
            complex_type powfactor(1.0,0.0);

            if (idx%2 == 1) {
                p_aux0 = aux0.get_data();
                p_aux1 = aux1.get_data();
            }
            else {
                p_aux0 = aux1.get_data();
                p_aux1 = aux0.get_data();
            }

            memcpy(p_aux1, p_aux0, (dim_over_2+1)*sizeof(complex_type) );

            for (size_t jdx = 1; jdx <= (dim / (2 * idx)); jdx++) {
                powfactor = powfactor * factor / ((double)jdx);

                for (size_t kdx = idx * jdx + 1; kdx <= dim_over_2 + 1; kdx++) {
                    p_aux1[kdx-1] += p_aux0[kdx-idx*jdx - 1]*powfactor;
                }

            }



        }


        if (fact) {
            summand = summand - p_aux1[dim_over_2];
//std::cout << -p_aux1[dim_over_2] << std::endl;
        }
        else {
            summand = summand + p_aux1[dim_over_2];
//std::cout << p_aux1[dim_over_2] << std::endl;
        }



    });

    // the resulting Hafnian of matrix mat
    complex_type res(0,0);
    summands.combine_each([&res](complex_type a) {
        res = res + a;
    });

    //Complex16 res = summand;


#if BLAS==0 // undefined BLAS
    omp_set_num_threads(NumThreads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(NumThreads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(NumThreads);
#endif

    // scale the result by the appropriate facto according to Eq (2.11) of in arXiv 1805.12498
    res = res * pow(this->scale_factor, dim_over_2);

    return Complex16(res.real(), res.imag() );
}





/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in A symmetric matrix for which the hafnian is calculated. (For example a covariance matrix of the Gaussian state.)
*/
template <class complex_type>
void
PowerTraceLoopHafnian<complex_type>::Update_mtx( matrix &mtx_in) {

    this->mtx_orig = mtx_in;

    // scale the input matrix according to according to Eq (2.14) of in arXiv 1805.12498
    ScaleMatrix();


}



/**
@brief Call to scale the input matrix according to according to Eq (2.14) of in arXiv 1805.12498
*/
template <class complex_type>
void
PowerTraceLoopHafnian<complex_type>::ScaleMatrix() {

    // scale the matrix to have the mean magnitudes matrix elements equal to one.
    if ( this->mtx_orig.rows <= 10) {
        this->mtx = this->mtx_orig;
        this->scale_factor = 1.0;
    }
    else {

        // determine the scale factor
        this->scale_factor = 0.0;
        for (size_t idx=0; idx<this->mtx_orig.size(); idx++) {
            this->scale_factor = this->scale_factor + std::sqrt( this->mtx_orig[idx].real()*this->mtx_orig[idx].real() + this->mtx_orig[idx].imag()*this->mtx_orig[idx].imag() );
        }
        this->scale_factor = this->scale_factor/this->mtx_orig.size()/std::sqrt(2);

        this->mtx = this->mtx_orig.copy();

        double inverse_scale_factor = 1/this->scale_factor;

        // scaling the matrix elements
        for (size_t row_idx=0; row_idx<this->mtx_orig.rows; row_idx++) {

            size_t row_offset = row_idx*this->mtx.stride;

            for (size_t col_idx=0; col_idx<this->mtx_orig.cols; col_idx++) {
                if (col_idx == row_idx ) {
                    this->mtx[row_offset+col_idx] = this->mtx[row_offset+col_idx]*sqrt(inverse_scale_factor);
                }
                else {
                    this->mtx[row_offset+col_idx] = this->mtx[row_offset+col_idx]*inverse_scale_factor;
                }

            }
        }

    }

}

template class PowerTraceLoopHafnian<Complex32>;

} // PIC
