#include <iostream>
#include "PowerTraceLoopHafnian.h"
#include "PowerTraceHafnianUtilities.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include <math.h>


/*
tbb::spin_mutex my_mutex;

double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {


/**
@brief Calculates the n-th power of 2.
@param n An natural number
@return Returns with the n-th power of 2.
*/
static unsigned long long power_of_2(unsigned long long n) {
  if (n == 0) return 1;
  if (n == 1) return 2;

  return 2 * power_of_2(n-1);
}




/**
@brief Call to calculate the hafnian of a complex matrix
@param mtx The matrix
@return Returns with the calculated hafnian
*/
Complex16
PowerTraceLoopHafnian::calculate() {


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
    unsigned long long permutation_idx_max = power_of_2( (unsigned long long) dim_over_2);


    // for cycle over the permutations n/2 according to Eq (3.24) in arXiv 1805.12498
    tbb::combinable<Complex16> summands{[](){return Complex16(0.0,0.0);}};

    tbb::parallel_for( tbb::blocked_range<unsigned long long>(0, permutation_idx_max, 1), [&](tbb::blocked_range<unsigned long long> r ) {


        Complex16 &summand = summands.local();

        for ( unsigned long long permutation_idx=r.begin(); permutation_idx != r.end(); permutation_idx++) {

/*
    Complex16 summand(0.0,0.0);

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
            for (size_t jdx = 0; jdx < number_of_ones; jdx++) {
                AZ[idx*AZ.stride + jdx] = mtx[positions_of_ones[idx]*mtx.stride + ((positions_of_ones[jdx]) ^ 1)];

            }
            diag_elements[idx] = mtx[positions_of_ones[idx]*mtx.stride + positions_of_ones[idx]];

        }

        // select the X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
        matrix cx_diag_elements(1, number_of_ones);
        for (size_t idx = 1; idx < number_of_ones; idx=idx+2) {
            cx_diag_elements[idx] = diag_elements[idx-1];
            cx_diag_elements[idx-1] = diag_elements[idx];
        }

        // calculate the loop correction elements for the loop hafnian
        matrix loop_corrections = calculate_loop_correction(diag_elements, cx_diag_elements, AZ);

        // calculating Tr(B^j) for all j's that are 1<=j<=dim/2
        // this is needed to calculate f_G(Z) defined in Eq. (3.17b) of arXiv 1805.12498
        matrix traces(dim_over_2, 1);
        if (number_of_ones != 0) {
            // here we need to make a copy since B will be transformed, but we need to use B in other calculations
            traces = calc_power_traces(AZ, dim_over_2);
        }
        else{
            // in case we have no 1's in the binary representation of permutation_idx we get zeros
            // this occurs once during the calculations
            memset( traces.get_data(), 0.0, traces.rows*traces.cols*sizeof(Complex16));
        }



        // fact corresponds to the (-1)^{(n/2) - |Z|} prefactor from Eq (3.24) in arXiv 1805.12498
        bool fact = ((dim_over_2 - number_of_ones/2) % 2);


        // auxiliary data arrays to evaluate the second part of Eqs (3.24) and (3.21) in arXiv 1805.12498
        matrix aux0(dim_over_2 + 1, 1);
        matrix aux1(dim_over_2 + 1, 1);
        memset( aux0.get_data(), 0.0, (dim_over_2 + 1)*sizeof(Complex16));
        memset( aux1.get_data(), 0.0, (dim_over_2 + 1)*sizeof(Complex16));
        aux0[0] = 1.0;
        // pointers to the auxiliary data arrays
        Complex16 *p_aux0=NULL, *p_aux1=NULL;

        for (size_t idx = 1; idx <= dim_over_2; idx++) {


            Complex16 factor = traces[idx - 1] / (2.0 * idx) + loop_corrections[idx-1]*0.5;
            Complex16 powfactor(1.0,0.0);


            if (idx%2 == 1) {
                p_aux0 = aux0.get_data();
                p_aux1 = aux1.get_data();
            }
            else {
                p_aux0 = aux1.get_data();
                p_aux1 = aux0.get_data();
            }

            memcpy(p_aux1, p_aux0, (dim_over_2+1)*sizeof(Complex16) );

            for (size_t jdx = 1; jdx <= (dim / (2 * idx)); jdx++) {
                powfactor = powfactor * factor / ((double)jdx);

                for (size_t kdx = idx * jdx + 1; kdx <= dim_over_2 + 1; kdx++) {
                    p_aux1[kdx-1] += p_aux0[kdx-idx*jdx - 1]*powfactor;
                }

            }



        }


        if (fact) {
            summand = summand - p_aux1[dim_over_2];
        }
        else {
            summand = summand + p_aux1[dim_over_2];
        }

    }

    });

    // the resulting Hafnian of matrix mat
    Complex16 res(0,0);
    summands.combine_each([&res](Complex16 a) {
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


    return res;
}


/**
@brief Call to calculate the loop corrections in Eq (3.26) of arXiv1805.12498
@param diag_elements The diagonal elements of the input matrix to be used to calculate the loop correction
@param cx_diag_elements The X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
@param AZ Corresponds to A^(Z), i.e. to the square matrix constructed from the input matrix (see the text below Eq.(3.20) of arXiv 1805.12498)
@return Returns with the calculated loop correction
*/
matrix
PowerTraceLoopHafnian::calculate_loop_correction( matrix &diag_elements, matrix& cx_diag_elements, matrix& AZ) {

    size_t dim_over_2 = mtx.rows/2;

    matrix loop_correction(dim_over_2, 1);

    matrix tmp_vec(1, diag_elements.size());



    for (size_t idx=0; idx<dim_over_2; idx++) {

        Complex16 tmp(0.0,0.0);
        for (size_t jdx=0; jdx<diag_elements.size(); jdx++) {
            tmp = tmp + cx_diag_elements[jdx] * diag_elements[jdx];
        }

        loop_correction[idx] = tmp;


         memset(tmp_vec.get_data(), 0, tmp_vec.size()*sizeof(Complex16));

         for (size_t kdx=0; kdx<cx_diag_elements.size(); kdx++) {
             for (size_t jdx=0; jdx<cx_diag_elements.size(); jdx++) {
                  tmp_vec[jdx] = tmp_vec[jdx] + cx_diag_elements[kdx] * AZ[kdx * AZ.stride + jdx];
             }
         }

         memcpy(cx_diag_elements.get_data(), tmp_vec.get_data(), tmp_vec.size()*sizeof(Complex16));

    }


    return loop_correction;

}


} // PIC
