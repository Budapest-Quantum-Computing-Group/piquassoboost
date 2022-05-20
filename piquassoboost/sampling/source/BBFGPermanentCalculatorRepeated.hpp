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

    //in the case when the first row multiplicity is zero, we need to find a nonzero multiplicity for the first row
    if (row_mult_in.size() > 0 && row_mult_in[0]==0) { 
        std::string error("BBFGPermanentCalculatorRepeated_Tasks:  The first element in row_mult dhould not be zero");
        throw error;   
    }

    // the first eleemnt is delta_vec is always 1, so we separate a roe with single multiplicity
    if ( row_mult_in.size() > 0 && row_mult_in[0]>1 ) {
        row_mult = PicState_int( row_mult_in.size() + 1);
        row_mult[0] = 1;
        row_mult[1] = row_mult_in[0]-1;
        memcpy( row_mult.get_data()+2, row_mult_in.get_data()+1, (row_mult_in.size()-1)*sizeof(int) );
        mtx = matrix_type( mtx_in.rows+1, mtx_in.cols );
        memcpy( mtx.get_data(), mtx_in.get_data(), mtx_in.cols*sizeof(scalar_type) );
        memcpy( mtx.get_data()+mtx.stride, mtx_in.get_data(), mtx_in.cols*mtx_in.rows*sizeof(scalar_type) );
//row_mult_in.print_matrix();
//row_mult.print_matrix();
    }
    else {
        row_mult = row_mult_in;
    }

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

    int sum_row_mult = sum(row_mult);
    int sum_col_mult = sum(col_mult);
    if ( sum_row_mult != sum_col_mult) {
        std::string error("BBFGPermanentCalculatorRepeated_Tasks::calculate:  Number of input and output states should be equal");
        throw error;
    }

    if (mtx.rows == 0 || sum_row_mult == 0 || sum_col_mult == 0)
        // the permanent of an empty matrix is 1 by definition
        return Complex16(1.0, 0.0);

    if (mtx.rows == 1) {

        Complex16 ret(1.0, 0.0);
        for (size_t idx=0; idx<col_mult.size(); idx++) {
            for (size_t jdx=0; jdx<col_mult[idx]; jdx++) {
                ret *= mtx[idx];
            }
        }

        return ret;
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
    for (size_t idx=0; idx<row_mult.size(); idx++) {
        n_ary_limits[idx] = row_mult[idx+1]+1;
    }


    uint64_t Idx_max = n_ary_limits[0]; 
    for (size_t idx=1; idx<n_ary_limits.size(); idx++) {
        Idx_max *= n_ary_limits[idx];
    }
   
    // thread local storage for partial hafnian
    priv_addend = tbb::combinable<scalar_type>{[](){return scalar_type(0.0,0.0);}};

//std::cout << Idx_max << std::endl;
//std::cout << "      ";
    n_aryGrayCodeCounter gcode_counter(n_ary_limits, 0);
    PicState_int&& gcode = gcode_counter.get();

    // calculate the initial column sum and binomial coefficient
    int64_t binomial_coeff = 1;
        
    matrix_type colsum( 1, mtx.cols);
    memcpy( colsum.get_data(), mtx.get_data(), colsum.size()*sizeof( scalar_type ) ); // the first eleemnt in detla_vec is always 1     
    scalar_type* mtx_data = mtx.get_data() + mtx.stride;

    // variable to refer to the parity of the delta vector (+1 if the number of -1 elements in delta vector is even, -1 otherwise)
    char parity = 1;

    for( size_t idx=0; idx<gcode.size(); idx++ ) {

        // the value of the element of the gray code stand for the number of \delta_i=-1 elements in the subset of multiplicated rows
        const int& minus_signs = gcode[idx];
        int row_mult_current = row_mult[idx+1];

        for( size_t col_idx=0; col_idx<mtx.cols; col_idx++) {
            colsum[col_idx] += (row_mult_current-minus_signs)*mtx_data[col_idx];
            if (  minus_signs & 1 ) {
                parity = -parity;
            }
        }

        // update the binomial coefficient
        binomial_coeff *= binomialCoeffInt64(row_mult_current, minus_signs);

        mtx_data += mtx.stride;

    }


    scalar_type colsum_prod = colsum[0];
    for( size_t idx=1; idx<col_mult[0]; idx++ ) {
        colsum_prod *= colsum[0];
    }

    for( size_t idx=1; idx<col_mult.size(); idx++ ) {
        for (size_t jdx=0; jdx<col_mult[idx]; jdx++) {
            colsum_prod *= colsum[idx];
        }
    }
//colsum.print_matrix();
//std::cout << colsum_prod << std::endl;


    // add the initial addend to the permanent
    scalar_type& addend_loc = priv_addend.local();
    if ( parity == 1 ) {
        addend_loc += colsum_prod*(precision_type)binomial_coeff;
    }
    else{
        addend_loc -= colsum_prod*(precision_type)binomial_coeff;
    }




    // iterate over gray codes to calculate permanent addends
    for (int64_t idx=1; idx<Idx_max; idx++ ) { 

        int changed_index, value_prev, value;
        if ( gcode_counter.next(changed_index, value_prev, value) ) {
            std::cout << std::endl;
            break;
        }


        parity = -parity;

//std::cout << value_prev << " " << value << std::endl;
        // update column sum and calculate the product of the elements
        int row_offset = (changed_index+1)*mtx.stride;
        scalar_type* mtx_data = mtx.get_data() + row_offset;
        scalar_type colsum_prod((precision_type)parity, 0.0);
        for( size_t col_idx=0; col_idx<col_mult.size(); col_idx++) {
            if ( value_prev < value ) {
                colsum[col_idx] -= mtx_data[col_idx]*2.0;
            }
            else {
                colsum[col_idx] += mtx_data[col_idx]*2.0;
            }

            for (size_t jdx=0; jdx<col_mult[col_idx]; jdx++) {
                colsum_prod *= colsum[col_idx];
            }

        }

        // update binomial factor
        int row_mult_current = row_mult[changed_index+1];
        binomial_coeff /= binomialCoeffInt64(row_mult_current, value_prev);
        binomial_coeff *= binomialCoeffInt64(row_mult_current, value);


        addend_loc += colsum_prod*(precision_type)binomial_coeff;


//std::cout << colsum_prod << " " << binomial_coeff << std::endl;
    

//////////////////
/*
        std::cout << changed_index << "     ";
        PicState_int&& gcode = gcode_counter.get();
        for( size_t jdx=0; jdx<gcode.size(); jdx++ ) {
            std::cout << gcode[jdx] << ", ";
        }
        std::cout << std::endl;
*/
///////////////////
    }



//std::cout << "opoooooooooooooooOO" << std::endl;    



    scalar_type permanent(0.0, 0.0);
    priv_addend.combine_each([&](scalar_type &a) {
        permanent = permanent + a;
    });

    permanent = permanent / (precision_type)(1ULL << (sum_row_mult-1));

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
