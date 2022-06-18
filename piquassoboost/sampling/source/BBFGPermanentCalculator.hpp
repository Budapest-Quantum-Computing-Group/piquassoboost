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


namespace pic {



// relieve Python extension from TBB functionalities
#ifndef CPYTHON


/**
@brief Class to calculate the hafnian of a complex matrix by a recursive power trace method. This algorithm is designed to support gaussian boson sampling simulations, it is not a general
purpose hafnian calculator. This algorithm accounts for the repeated occupancy in the covariance matrix.
*/
template<class matrix_type, class scalar_type, class precision_type>
class BBFGPermanentCalculator_Tasks  {


protected:
    /// The input matrix. Must be selfadjoint positive definite matrix with eigenvalues between 0 and 1.
    matrix_type mtx;
    ///
    matrix_type mtx2;



public:

/**
@brief Nullary constructor of the class.
@return Returns with the instance of the class.
*/
BBFGPermanentCalculator_Tasks() {



}

/**
@brief Constructor of the class.
@param mtx_in A selfadjoint matrix for which the permanent is calculated. This matrix has to be positive definite matrix with eigenvalues between 0 and 1 (for example a covariance matrix of the Gaussian state.) matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state)
@return Returns with the instance of the class.
*/
BBFGPermanentCalculator_Tasks( matrix_type &mtx_in ) {

    Update_mtx( mtx_in );

}



/**
@brief Default destructor of the class.
*/
virtual ~BBFGPermanentCalculator_Tasks() {

}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
Complex16 calculate() {


    if (mtx.rows == 0 || mtx.cols == 0) {
        // the permanent of an empty matrix is 1 by definition
        return 1.0;
    }
    if (mtx.rows >= mtx.cols + 2)
        return Complex16(0.0, 0.0);    

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


    mtx2 = matrix_type(mtx.rows, mtx.cols);
    for(size_t idx=0; idx<mtx.size(); idx++) {
        mtx2[idx] = mtx[idx]*2.0;
    }     

    // thread local storage for partial hafnian
    tbb::combinable<scalar_type> priv_addend{[](){return scalar_type(0.0,0.0);}};

    
    uint64_t Idx_max = 1ULL << mtx.rows-1; 
    

    // determine the concurrency of the calculation
    unsigned int nthreads = std::thread::hardware_concurrency();
    int64_t concurrency = (int64_t)nthreads * 4;
    concurrency = concurrency < Idx_max ? concurrency : (int64_t)Idx_max;
    
    tbb::parallel_for( (int64_t)0, concurrency, (int64_t)1, [&](int64_t job_idx) {
//    for( int64_t job_idx=0; job_idx<concurrency; job_idx++) {


        // initial offset and upper boundary of the gray code counter
        int64_t work_batch = Idx_max/concurrency;
        int64_t initial_offset = job_idx*work_batch;
        int64_t offset_max = (job_idx+1)*work_batch;
        if ( job_idx == concurrency-1) {
            offset_max = Idx_max;
        } 

        // array storing the actual delta vector
        PicState_int delta_vec( mtx.rows, 0 );

        // variable to refer to the parity of the delta vector (+1 if the number of changed elements in delta vector is even, 0 otherwise)
        char parity = 1;
    
        // determine initial columsn sum corresponding to the given intial delta vector. 
        matrix_base<scalar_type> colsum( 1, mtx.cols);
        std::uninitialized_copy_n(mtx.get_data(), colsum.size(), colsum.get_data());

        size_t initial_offset_gray_code = initial_offset ^ (initial_offset >> 1);

        auto mtx_data = mtx.get_data();// + mtx.stride; 

        for ( size_t row_idx=mtx.rows-1; row_idx>0; row_idx--) {
             
            if ( initial_offset_gray_code & 1 ) {
                for( size_t col_idx=0; col_idx<mtx.cols; col_idx++) {
                    colsum[col_idx] -= mtx_data[row_idx*mtx.stride+col_idx];
                }
                parity = -parity;
                delta_vec[row_idx] = 1;
            }
            else {
                for( size_t col_idx=0; col_idx<mtx.cols; col_idx++) {
                    colsum[col_idx] += mtx_data[row_idx*mtx.stride+col_idx];
                }
                delta_vec[row_idx] = 0;
            }
            //mtx_data += mtx.stride;
            initial_offset_gray_code = initial_offset_gray_code >> 1;
        }     

        scalar_type& addend_loc = priv_addend.local();

        if ( parity==1 ) {
            addend_loc += product_reduction( colsum );
        }
        else {
            addend_loc -= product_reduction( colsum );        
        }
   

        for( uint64_t idx=initial_offset+1; idx<offset_max; idx++) {

            // find the index of the changed bit:
            // If you number the bits starting with 0 for least significant, the position of the bit to change to increase
            // a binary-reflected Gray code is the position of the lowest bit set in an increasing binary number 
            // to get the numbering you presented, subtract from the number of bit positions.

            uint64_t idx_loc = idx;
            char change_idx = mtx.rows-1;
            while ( (idx_loc & 1ULL) == 0 ) {
                idx_loc = idx_loc >> 1;
                change_idx--;
            }

            // update gray code
            int& changed_delta_element = delta_vec[change_idx];
            changed_delta_element = changed_delta_element ? 0 : 1;
       
            // update the column sum
            size_t row_offset = change_idx*mtx2.stride;
            auto mtx2_data = mtx2.get_data() + row_offset;

            parity = -parity;
            scalar_type colsum_prod(parity, 0.0);


            if ( changed_delta_element ) {
                for( size_t col_idx=0; col_idx<mtx2.cols; col_idx++) {
                    colsum[col_idx] -= mtx2_data[col_idx];  
                    colsum_prod *= colsum[col_idx];
                }
            }
            else {
                for( size_t col_idx=0; col_idx<mtx2.cols; col_idx++) {
                    colsum[col_idx] += mtx2_data[col_idx];            
                    colsum_prod *= colsum[col_idx];
                }                
            }

            addend_loc += colsum_prod;


        }
        for (size_t n = colsum.size(); n > 0; --n)
            colsum[n-1].~scalar_type();
//    }
    });

    scalar_type permanent(0.0, 0.0);
    priv_addend.combine_each([&](scalar_type &a) {
        permanent += a;
    });
    
    permanent /= (precision_type)(1ULL << (mtx.rows-1));

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





}; //BBFGPermanentCalculator_Tasks


#endif // CPYTHON





} // PIC
