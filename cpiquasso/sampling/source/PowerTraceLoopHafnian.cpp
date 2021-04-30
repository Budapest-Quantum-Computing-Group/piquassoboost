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
PowerTraceLoopHafnian::PowerTraceLoopHafnian() : PowerTraceHafnian() {

}

/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix for which the hafnian is calculated. (For example a covariance matrix of the Gaussian state.)
@return Returns with the instance of the class.
*/
PowerTraceLoopHafnian::PowerTraceLoopHafnian( matrix &mtx_in ) {
    assert(isSymmetric(mtx_in));

    Update_mtx( mtx_in );

}


/**
@brief Call to calculate the hafnian of a complex matrix
@param mtx The matrix
@return Returns with the calculated hafnian
*/
Complex16
PowerTraceLoopHafnian::calculate() {

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
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
Complex16
PowerTraceLoopHafnian::calculate(unsigned long long start_idx, unsigned long long step_idx, unsigned long long max_idx ) {


    if ( mtx.rows != mtx.cols) {
        std::cout << "The input matrix should be square shaped, bu matrix with " << mtx.rows << " rows and with " << mtx.cols << " columns was given" << std::endl;
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

    size_t dim = mtx.rows;


    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return Complex16(1,0);
    }
    else if (mtx.rows % 2 != 0) {
        // the hafnian of odd shaped matrix is 0 by definition
        return Complex16(0.0, 0.0);
    }

    size_t dim_over_2 = mtx.rows / 2;

    // for cycle over the permutations n/2 according to Eq (3.24) in arXiv 1805.12498
    tbb::combinable<Complex32> summands{[](){return Complex32(0.0,0.0);}};

    tbb::parallel_for( start_idx, max_idx, step_idx, [&](unsigned long long permutation_idx) {


        Complex32 &summand = summands.local();

/*
    Complex32 summand(0.0,0.0);

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
            size_t row_offset = (positions_of_ones[idx] ^ 1)*mtx.stride;
            for (size_t jdx = 0; jdx < number_of_ones; jdx++) {
                AZ[idx*AZ.stride + jdx] = mtx[row_offset + ((positions_of_ones[jdx]))];
            }
            diag_elements[idx] = mtx[(positions_of_ones[idx])*mtx.stride + positions_of_ones[idx]];

        }

        // select the X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
        matrix cx_diag_elements(number_of_ones, 1);
        for (size_t idx = 1; idx < number_of_ones; idx=idx+2) {
            cx_diag_elements[idx] = diag_elements[idx-1];
            cx_diag_elements[idx-1] = diag_elements[idx];
        }


        matrix cx_diag_elements2 = cx_diag_elements.copy();
        matrix diag_elements2 = diag_elements.copy();
        matrix AZ2 = AZ.copy();
tbb::tick_count t0 = tbb::tick_count::now();
        matrix32 loop_corrections2 = CalculateLoopCorrectionWithHessenberg(cx_diag_elements2, diag_elements2, AZ2, mtx.rows/2);
tbb::tick_count t1 = tbb::tick_count::now();

tbb::tick_count t2 = tbb::tick_count::now();
        // calculate the loop correction elements for the loop hafnian
        matrix32 loop_corrections = CalculateLoopCorrection(cx_diag_elements, diag_elements, AZ);
tbb::tick_count t3 = tbb::tick_count::now();



{
      tbb::spin_mutex::scoped_lock my_lock{my_mutex};

time_szamlalo += (t1-t0).seconds();
time_nevezo += (t3-t2).seconds();

std::cout << time_szamlalo/time_nevezo << std::endl;


if (AZ.rows == 6) {
    loop_corrections.print_matrix();
    loop_corrections2.print_matrix();

}

}


if (AZ.rows == 6) {
    exit(-1);
}

        // calculating Tr(B^j) for all j's that are 1<=j<=dim/2
        // this is needed to calculate f_G(Z) defined in Eq. (3.17b) of arXiv 1805.12498
        matrix32 traces(dim_over_2, 1);
        if (number_of_ones != 0) {
            // here we need to make a copy since B will be transformed, but we need to use B in other calculations
            traces = calc_power_traces<matrix32, Complex32>(AZ, dim_over_2);
        }
        else{
            // in case we have no 1's in the binary representation of permutation_idx we get zeros
            // this occurs once during the calculations
            memset( traces.get_data(), 0.0, traces.rows*traces.cols*sizeof(Complex32));
        }



        // fact corresponds to the (-1)^{(n/2) - |Z|} prefactor from Eq (3.24) in arXiv 1805.12498
        bool fact = ((dim_over_2 - number_of_ones/2) % 2);


        // auxiliary data arrays to evaluate the second part of Eqs (3.24) and (3.21) in arXiv 1805.12498
        matrix32 aux0(dim_over_2 + 1, 1);
        matrix32 aux1(dim_over_2 + 1, 1);
        memset( aux0.get_data(), 0.0, (dim_over_2 + 1)*sizeof(Complex32));
        memset( aux1.get_data(), 0.0, (dim_over_2 + 1)*sizeof(Complex32));
        aux0[0] = 1.0;
        // pointers to the auxiliary data arrays
        Complex32 *p_aux0=NULL, *p_aux1=NULL;

        for (size_t idx = 1; idx <= dim_over_2; idx++) {


            Complex32 factor = traces[idx - 1] / (2.0 * idx) + loop_corrections[idx-1]*0.5;
/*
{
      tbb::spin_mutex::scoped_lock my_lock{my_mutex};
      //traces.print_matrix();
      std::cout << factor << " " << loop_corrections[idx-1]*0.5 << std::endl;
  }
*/
            Complex32 powfactor(1.0,0.0);

            if (idx%2 == 1) {
                p_aux0 = aux0.get_data();
                p_aux1 = aux1.get_data();
            }
            else {
                p_aux0 = aux1.get_data();
                p_aux1 = aux0.get_data();
            }

            memcpy(p_aux1, p_aux0, (dim_over_2+1)*sizeof(Complex32) );

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
    Complex32 res(0,0);
    summands.combine_each([&res](Complex32 a) {
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
    res = res * pow(scale_factor, dim_over_2);

    return Complex16(res.real(), res.imag() );
}


/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
matrix32
PowerTraceLoopHafnian::CalculateLoopCorrection( matrix &cx_diag_elements, matrix& diag_elements, matrix& AZ) {

    size_t dim_over_2 = mtx.rows/2;

    if (AZ.rows < 3000) {

        // for smaller matrices first calculate the corerction in 16 byte precision, than convert the result to 32 byte precision
        matrix &&loop_correction = calculate_loop_correction<matrix, Complex16>(cx_diag_elements, diag_elements, AZ, dim_over_2);

        matrix32 loop_correction32(dim_over_2, 1);
        for (size_t idx=0; idx<loop_correction.size(); idx++ ) {
            loop_correction32[idx].real( loop_correction[idx].real() );
            loop_correction32[idx].imag( loop_correction[idx].imag() );
        }

        return loop_correction32;

    }
    else{

        // for smaller matrices first convert the input matrices to 32 byte precision, than calculate the diag correction
        matrix32 diag_elements32( diag_elements.rows, diag_elements.cols);
        matrix32 cx_diag_elements32( cx_diag_elements.rows, cx_diag_elements.cols);
        for (size_t idx=0; idx<diag_elements32.size(); idx++) {
            diag_elements32[idx].real( diag_elements[idx].real() );
            diag_elements32[idx].imag( diag_elements[idx].imag() );

            cx_diag_elements32[idx].real( cx_diag_elements[idx].real() );
            cx_diag_elements32[idx].imag( cx_diag_elements[idx].imag() );
        }

        matrix32 AZ_32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ_32[idx].real( AZ[idx].real() );
            AZ_32[idx].imag( AZ[idx].imag() );
        }

        return calculate_loop_correction<matrix32, Complex32>(cx_diag_elements32, diag_elements32, AZ_32, dim_over_2);


    }


}











/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in A symmetric matrix for which the hafnian is calculated. (For example a covariance matrix of the Gaussian state.)
*/
void
PowerTraceLoopHafnian::Update_mtx( matrix &mtx_in) {

    mtx_orig = mtx_in;

    // scale the input matrix according to according to Eq (2.14) of in arXiv 1805.12498
    ScaleMatrix();



}



/**
@brief Call to scale the input matrix according to according to Eq (2.14) of in arXiv 1805.12498
@param mtx_in Input matrix defined by
*/
void
PowerTraceLoopHafnian::ScaleMatrix() {

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

        mtx = mtx_orig.copy();

        double inverse_scale_factor = 1/scale_factor;

        // scaling the matrix elements
        for (size_t row_idx=0; row_idx<mtx_orig.rows; row_idx++) {

            size_t row_offset = row_idx*mtx.stride;

            for (size_t col_idx=0; col_idx<mtx_orig.cols; col_idx++) {
                if (col_idx == row_idx ) {
                    mtx[row_offset+col_idx] = mtx[row_offset+col_idx]*sqrt(inverse_scale_factor);
                }
                else {
                    mtx[row_offset+col_idx] = mtx[row_offset+col_idx]*inverse_scale_factor;
                }

            }
        }

    }

}



} // PIC
