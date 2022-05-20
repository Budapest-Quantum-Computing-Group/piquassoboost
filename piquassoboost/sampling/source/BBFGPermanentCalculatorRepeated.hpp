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
    /// The input matrix. Must be selfadjoint positive definite matrix with eigenvalues between 0 and 1.
    matrix_type mtx_mult;
    ///
    matrix_type mtx2;
    ///
    matrix_type mtx2_mult;
    ///
    PicState_int row_mult;
    ///
    PicState_int col_mult;


    // thread local storage for partial hafnian
    tbb::combinable<scalar_type> priv_addend;


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
    row_mult = row_mult_in;
    col_mult = col_mult_in;

}


inline long double convertToDouble(Complex16& complex){
    return complex.real();
}
inline long double convertToDouble(Complex32& complex){
    return complex.real();
}
inline long double convertToDouble(double& complex){
    return complex;
}
inline long double convertToDouble(long double& complex){
    return complex;
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


    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return 1.0;
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
row_mult.print_matrix();
    PicState_int n_ary_limits( row_mult.size() );
    for (size_t idx=0; idx<row_mult.size(); idx++) {
        n_ary_limits[idx] = row_mult[idx]+1;
    }


    uint64_t Idx_max = n_ary_limits[0];
    for (size_t idx=1; idx<n_ary_limits.size(); idx++) {
        Idx_max *= n_ary_limits[idx];
    }
   
std::cout << Idx_max << std::endl;
std::cout << "      ";
    n_aryGrayCodeCounter gcode_counter(n_ary_limits, 0);
    for (int64_t idx=0; idx<Idx_max; idx++ ) { 

        PicState_int&& gcode = gcode_counter.get();

        for( size_t jdx=0; jdx<gcode.size(); jdx++ ) {
            std::cout << gcode[jdx] << ", ";
        }

        int changed_index;
        if ( gcode_counter.next(changed_index) ) {
            std::cout << std::endl;
            break;
        }

        std::cout << std::endl << changed_index << "     ";


    }


/////////////////////////////////

    mtx2 = matrix_type(mtx.rows, mtx.cols);
    for(size_t idx=0; idx<mtx.size(); idx++) {
        mtx2[idx] = mtx[idx]*2.0;
    } 


    // thread local storage for partial hafnian
    priv_addend = tbb::combinable<scalar_type>{[](){return scalar_type(0.0,0.0);}};

/////////////////////////////
    mtx_mult = mtx.copy();
    mtx2_mult = mtx2.copy();

    size_t start_index = 0;
    recursive_iteration( start_index );

///////////////////////////////    


    



    scalar_type permanent(0.0, 0.0);
    priv_addend.combine_each([&](scalar_type &a) {
        permanent = permanent + a;
    });

    permanent = permanent / (precision_type)(1ULL << (mtx.rows-1));

    return Complex16(permanent.real(), permanent.imag());
}

/**
@brief ???
@param ???
*/
void recursive_iteration( size_t& start_index) {


    // determine the concurrency of the calculation
    unsigned int nthreads = std::thread::hardware_concurrency();
    size_t fixed_delta_elements = 0;
    while ( nthreads ) {
        nthreads = nthreads >> 1;
        fixed_delta_elements++;
    }


    fixed_delta_elements = fixed_delta_elements < mtx.rows ? fixed_delta_elements : mtx.rows-1;


    size_t concurrent_delta_vectors = 1 << fixed_delta_elements;

//concurrent_delta_vectors = 16;
//fixed_delta_elements = 4;
//std::cout << mtx.rows << " " << fixed_delta_elements << " " << concurrent_delta_vectors << std::endl;
    


    calc_subpermanent(concurrent_delta_vectors, fixed_delta_elements);



}

/**
@brief ???
@param ???
*/
void calc_subpermanent(size_t& concurrent_delta_vectors, const size_t& fixed_delta_elements) {

    tbb::parallel_for( (size_t)0, concurrent_delta_vectors, (size_t)1, [&](size_t delta_vec_idx){
//    for( size_t delta_vec_idx=0; delta_vec_idx<concurrent_delta_vectors; delta_vec_idx++ ) {

        // variable to refer to the parity of the delta vector (+1 if the number of changed elements in delta vector is even, 0 otherwise)
        char parity = 1;
    
        // determine initial columsn sum corresponding to the given delta vector set. 
        // (the leading "concurrent_delta_vectors" elements of the delta vectors are kept fixed
        // during the iterations
        matrix_type colsum( 1, mtx_mult.cols);
        memcpy( colsum.get_data(), mtx_mult.get_data(), colsum.size()*sizeof( scalar_type ) );

        size_t delta_vec_idx_loc = delta_vec_idx;

        scalar_type* mtx_data = mtx_mult.get_data() + mtx_mult.stride; 

        for ( size_t row_idx=1; row_idx<mtx_mult.rows; row_idx++) {
             
            if ( delta_vec_idx_loc & 1 ) {
                for( size_t col_idx=0; col_idx<mtx_mult.cols; col_idx++) {
                    colsum[col_idx] -= mtx_data[col_idx];
                }
                parity = parity ? 0 : 1;
            }
            else {
                for( size_t col_idx=0; col_idx<mtx_mult.cols; col_idx++) {
                    colsum[col_idx] += mtx_data[col_idx];
                }
            }
            mtx_data += mtx_mult.stride;
            delta_vec_idx_loc = delta_vec_idx_loc >> 1;
        }
    
    

        scalar_type& addend_loc = priv_addend.local();

        if ( parity ) {
            addend_loc += product_reduction( colsum );
        }
        else {
            addend_loc -= product_reduction( colsum );        
        }




        // array storing the actual delta vector
        PicState_int delta_vec( mtx_mult.rows-fixed_delta_elements, 0 );
//delta_vec.print_matrix();


        // the number of variable delta vector elements
        size_t variable_delta_elements = mtx_mult.rows-1-fixed_delta_elements;      

        // launch the gray code counter
        uint64_t Idx_max = 1ULL << variable_delta_elements;


        for( uint64_t idx=1; idx<Idx_max; idx++) {

            // find the index of the changed bit:
            // If you number the bits starting with 0 for least significant, the position of the bit to change to increase
            // a binary-reflected Gray code is the position of the lowest bit set in an increasing binary number 
            // to get the numbering you presented, subtract from the number of bit positions.

            uint64_t idx_loc = idx;
            char change_idx = variable_delta_elements;
            while ( (idx_loc & 1ULL) == 0 ) {
                idx_loc = idx_loc >> 1;
                change_idx--;
            }

            // update gray code
            int& changed_delta_element = delta_vec[change_idx];
            changed_delta_element = changed_delta_element ? 0 : 1;
    

       
            // update the column sum
            size_t row_offset = (fixed_delta_elements+change_idx)*mtx2_mult.stride;
            scalar_type* mtx2_data = mtx2_mult.get_data() + row_offset;

            parity = parity ? 0 : 1;
            scalar_type colsum_prod(parity ? 1.0 : -1.0 ,0.0);


            if ( changed_delta_element ) {
                for( size_t col_idx=0; col_idx<mtx2_mult.cols; col_idx++) {
                    colsum[col_idx] -= mtx2_data[col_idx];  
                    colsum_prod *= colsum[col_idx];
                }
            }
            else {
                for( size_t col_idx=0; col_idx<mtx2_mult.cols; col_idx++) {
                    colsum[col_idx] += mtx2_data[col_idx];            
                    colsum_prod *= colsum[col_idx];
                }                
            }

            addend_loc += colsum_prod;


        }
  
//    }
    });

};

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
