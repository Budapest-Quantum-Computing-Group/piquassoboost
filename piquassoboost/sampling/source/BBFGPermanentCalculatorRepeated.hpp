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

#ifndef CPYTHON
#include "tbb/tbb.h"
#endif

#include <thread>
#include "PicState.h"
#include "PicVector.hpp"
#include "common_functionalities.h"
#include "n_aryGrayCodeCounter.h"


namespace pic {



// relieve Python extension from TBB functionalities
#ifndef CPYTHON


/**
@brief Class to calculate the hafnian of a complex matrix by a recursive power trace method. This algorithm is designed to support gaussian boson sampling simulations, it is not a general
purpose hafnian calculator. This algorithm accounts for the repeated occupancy in the covariance matrix.
*/
template<class matrix_type, class scalar_type, class precision_type>
class BBFGPermanentCalculatorRepeated_Tasks  {


protected:
    /// The input matrix. Must be selfadjoint positive definite matrix with eigenvalues between 0 and 1.
    matrix_type mtx;
    ///
    matrix_type mtx2;
    ///
    PicState_int row_mult;
    ///
    PicState_int col_mult;



public:

/**
@brief Nullary constructor of the class.
@return Returns with the instance of the class.
*/
BBFGPermanentCalculatorRepeated_Tasks() {



}

/**
@brief Constructor of the class.
@param mtx_in A selfadjoint matrix for which the permanent is calculated. This matrix has to be positive definite matrix with eigenvalues between 0 and 1 (for example a covariance matrix of the Gaussian state.) matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state)
@return Returns with the instance of the class.
*/
BBFGPermanentCalculatorRepeated_Tasks( matrix_type &mtx_in, PicState_int& col_mult_in, PicState_int& row_mult_in ) {

    Update_mtx( mtx_in );
    
    int* minelem = std::min_element(&row_mult_in[0], &row_mult_in[0]+row_mult_in.size(), [](int a, int b) { return a==b ? false : (b==0 || (a!=0 && a < b)); });

    if ( row_mult_in.size() > 0 && *minelem != 0) {
        row_mult = PicState_int( row_mult_in.size() + 1);
        row_mult[0] = 1;
        memcpy( row_mult.get_data()+1, row_mult_in.get_data(), row_mult_in.size()*sizeof(int) );
        row_mult[1+(minelem - &row_mult_in[0])]--;
        mtx = matrix_type( mtx_in.rows+1, mtx_in.cols );
        memcpy( mtx.get_data(), mtx_in.get_data()+mtx_in.stride*(minelem - &row_mult_in[0]), mtx_in.cols*sizeof(*mtx.get_data()) );
        memcpy( mtx.get_data()+mtx.stride, mtx_in.get_data(), mtx_in.cols*mtx_in.rows*sizeof(*mtx.get_data()) );
    }
    else {
        row_mult = row_mult_in;
    }

    col_mult = col_mult_in;

}



/**
@brief Default destructor of the class.
*/
virtual ~BBFGPermanentCalculatorRepeated_Tasks() {

}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
Complex16 calculate() {


    int sum_row_mult = sum(row_mult);
    int sum_col_mult = sum(col_mult);
    if ( sum_row_mult != sum_col_mult) {
        std::string error("BBFGPermanentCalculatorRepeated_Tasks::calculate:  Number of input and output states should be equal");
        throw error;
    }

    if (mtx.rows == 0 || mtx.cols == 0 || sum_row_mult == 0 || sum_col_mult == 0)
        // the permanent of an empty matrix is 1 by definition
        return Complex16(1.0, 0.0);

    if (mtx.rows == 1) {

        scalar_type ret(1.0, 0.0);
        for (size_t idx=0; idx<col_mult.size(); idx++) {
            for (size_t jdx=0; jdx<col_mult[idx]; jdx++) {
                ret *= mtx[idx];
            }
        }

        return Complex16(ret.real(), ret.imag() );
    }


    mtx2 = matrix_type(mtx.rows, mtx.cols);
    for(size_t idx=0; idx<mtx.size(); idx++) {
        mtx2[idx] = mtx[idx]*2.0;
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

//row_mult.print_matrix();
    PicState_int n_ary_limits( row_mult.size()-1 );
    for (size_t idx=0; idx<n_ary_limits.size(); idx++) {
        n_ary_limits[idx] = row_mult[idx+1]+1;
    }


    uint64_t Idx_max = n_ary_limits[0]; 
    for (size_t idx=1; idx<n_ary_limits.size(); idx++) {
        Idx_max *= n_ary_limits[idx];
    }
   
    // thread local storage for partial hafnian
    tbb::combinable<scalar_type> priv_addend{[](){return scalar_type(0.0,0.0);}};

    // determine the concurrency of the calculation
    unsigned int nthreads = std::thread::hardware_concurrency();
    int64_t concurrency = (int64_t)nthreads * 4;
    concurrency = concurrency < Idx_max ? concurrency : (int64_t)Idx_max;
    
    tbb::parallel_for( (int64_t)0, concurrency, (int64_t)1, [&](int64_t job_idx) {
//    for( int64_t job_idx=0; job_idx<concurrency; job_idx++) {


        // initial offset and upper boundary of the gray code counter
        int64_t work_batch = Idx_max/concurrency;
        int64_t initial_offset = job_idx*work_batch;
        int64_t offset_max = (job_idx+1)*work_batch-1;
        if ( job_idx == concurrency-1) {
            offset_max = Idx_max-1;
        } 




        n_aryGrayCodeCounter gcode_counter(n_ary_limits, initial_offset);


        gcode_counter.set_offset_max( offset_max );
        PicState_int gcode = gcode_counter.get();

        // calculate the initial column sum and binomial coefficient
        int64_t binomial_coeff = 1;

        matrix_base<scalar_type> colsum( 1, col_mult.size());
        std::uninitialized_copy_n(mtx.get_data(), colsum.size(), colsum.get_data());  
        auto mtx_data = mtx.get_data() + mtx.stride;

        // variable to count all the -1 elements in the delta vector
        int minus_signs_all = 0;

        for( size_t idx=0; idx<gcode.size(); idx++ ) {

            // the value of the element of the gray code stand for the number of \delta_i=-1 elements in the subset of multiplicated rows
            const int& minus_signs = gcode[idx];
            int row_mult_current = row_mult[idx+1];

            for( size_t col_idx=0; col_idx<col_mult.size(); col_idx++) {
                colsum[col_idx] += (row_mult_current-2*minus_signs)*mtx_data[col_idx];
            }

            minus_signs_all += minus_signs;

            // update the binomial coefficient
            binomial_coeff *= binomialCoeffInt64(row_mult_current, minus_signs);

            mtx_data += mtx.stride;

        }


        // variable to refer to the parity of the delta vector (+1 if the number of -1 elements in delta vector is even, -1 otherwise)
        char parity = (minus_signs_all % 2 == 0) ? 1 : -1; 

        scalar_type colsum_prod((precision_type)parity, (precision_type)0.0);
        for( size_t idx=0; idx<col_mult.size(); idx++ ) {
            for (size_t jdx=0; jdx<col_mult[idx]; jdx++) {
                colsum_prod *= colsum[idx];
            }
        }



        // add the initial addend to the permanent
        scalar_type& addend_loc = priv_addend.local();
        addend_loc += colsum_prod*(precision_type)binomial_coeff;



        // iterate over gray codes to calculate permanent addends
        for (int64_t idx=initial_offset+1; idx<offset_max+1; idx++ ) { 

            int changed_index, value_prev, value;
            if ( gcode_counter.next(changed_index, value_prev, value) ) {
                break;
            }


            parity = -parity;


            // update column sum and calculate the product of the elements
            int row_offset = (changed_index+1)*mtx.stride;
            auto mtx_data = mtx2.get_data() + row_offset;
            scalar_type colsum_prod((precision_type)parity, (precision_type)0.0);
            for( size_t col_idx=0; col_idx<col_mult.size(); col_idx++) {
                if ( value_prev < value ) {
                    colsum[col_idx] -= mtx_data[col_idx];
                }
                else {
                    colsum[col_idx] += mtx_data[col_idx];
                }

                for (size_t jdx=0; jdx<col_mult[col_idx]; jdx++) {
                    colsum_prod *= colsum[col_idx];
                }

            }


            // update binomial factor
            int row_mult_current = row_mult[changed_index+1];
            binomial_coeff = value < value_prev ? binomial_coeff*value_prev/(row_mult_current-value) : binomial_coeff*(row_mult_current-value_prev)/value;
            //binomial_coeff /= binomialCoeffInt64(row_mult_current, value_prev);
            //binomial_coeff *= binomialCoeffInt64(row_mult_current, value);


            addend_loc += colsum_prod*(precision_type)binomial_coeff;

    
        }

        for (size_t n = colsum.size(); n > 0; --n)
            colsum[n-1].~scalar_type();



//    }
    });


    scalar_type permanent(0.0, 0.0);
    priv_addend.combine_each([&](scalar_type &a) {
        permanent += a;
    });

    permanent /= (precision_type)(1ULL << (sum_row_mult-1));

    return Complex16(permanent.real(), permanent.imag());
}



/**
@brief Call to update the memory address of the matrix mtx and reorder the matrix elements into a_1^*,a_1^*,a_2,a_2^*, ... a_N,a_N^* order.
@param mtx_in Input matrix defined by
*/
void Update_mtx( matrix_type &mtx_in) {

    mtx = mtx_in;
}



protected:





}; //BBFGPermanentCalculatorRepeated_Tasks


#endif // CPYTHON





} // PIC
