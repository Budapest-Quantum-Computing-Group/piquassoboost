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
#define LONG_DOUBLE_CUTOFF 50
#endif // LONG_DOUBLE_CUTOFF

#include <iostream>
#include "PowerTraceHafnianRecursive.h"
#include "PowerTraceHafnianUtilities.hpp"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>

#ifdef __MPI__
#include <mpi.h>
#endif // MPI




/*
static tbb::spin_mutex mymutex;

double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {


/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state.)
@param occupancy_in An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
template <class small_scalar_type, class scalar_type>
PowerTraceHafnianRecursive<small_scalar_type, scalar_type>::PowerTraceHafnianRecursive( matrix &mtx_in, PicState_int64& occupancy_in ) {
    assert(isSymmetric(mtx_in));

    this->mtx = mtx_in;
    occupancy = occupancy_in;

}


/**
@brief Destructor of the class.
*/
template <class small_scalar_type, class scalar_type>
PowerTraceHafnianRecursive<small_scalar_type, scalar_type>::~PowerTraceHafnianRecursive() {


}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
template <class small_scalar_type, class scalar_type>
Complex16
PowerTraceHafnianRecursive<small_scalar_type, scalar_type>::calculate() {

    if (this->mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return Complex16(1,0);
    }

    // number of modes spanning the gaussian state
    size_t num_of_modes = occupancy.size();

    unsigned long long permutation_idx_max = power_of_2( (unsigned long long) num_of_modes);

#ifdef __MPI__
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    PowerTraceHafnianRecursive_Tasks hafnian_calculator = PowerTraceHafnianRecursive_Tasks(mtx, occupancy);
    Complex16 hafnian = hafnian_calculator.calculate(current_rank+1, world_size, permutation_idx_max);

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

    PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type> hafnian_calculator = PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type>(this->mtx, occupancy);
    Complex16 hafnian = hafnian_calculator.calculate(current_rank+1, world_size, permutation_idx_max);

    return hafnian;
#endif


}

template class PowerTraceHafnianRecursive<double, long double>;
template class PowerTraceHafnianRecursive<double, double>;
template class PowerTraceHafnianRecursive<long double, long double>;
template class PowerTraceHafnianRecursive<RationalInf, RationalInf>;









/**
@brief Nullary constructor of the class.
@return Returns with the instance of the class.
*/
template <class small_scalar_type, class scalar_type>
PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type>::PowerTraceHafnianRecursive_Tasks() {

    // set the maximal number of spawned tasks living at the same time
    max_task_num = 300;
    // The current number of spawned tasks
    task_num = 0;
    // mutual exclusion to count the spawned tasks
    task_count_mutex = new tbb::spin_mutex();

}




/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state.)
@param occupancy_in An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
template <class small_scalar_type, class scalar_type>
PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type>::PowerTraceHafnianRecursive_Tasks( matrix &mtx_in, PicState_int64& occupancy_in ) {

    this->Update_mtx( mtx_in );

    occupancy = occupancy_in;


    if (this->mtx.rows != 2*occupancy.size()) {
        std::cout << "The length of array occupancy should be equal to the half of the dimension of the input matrix mtx. Exiting" << std::endl;
        exit(-1);
    }

    if (this->mtx.rows % 2 != 0) {
        // The dimensions of the matrix should be even
        std::cout << "In PowerTraceHafnianRecursive_Tasks algorithm the dimensions of the matrix should strictly be even. Exiting" << std::endl;
        exit(-1);
    }

    // set the maximal number of spawned tasks living at the same time
    max_task_num = 300;
    // The current number of spawned tasks
    task_num = 0;
    // mutual exclusion to count the spawned tasks
    task_count_mutex = new tbb::spin_mutex();


}


/**
@brief Destructor of the class.
*/
template <class small_scalar_type, class scalar_type>
PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type>::~PowerTraceHafnianRecursive_Tasks() {
    delete task_count_mutex;
}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
template <class small_scalar_type, class scalar_type>
Complex16
PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type>::calculate() {

    // number of modes spanning the gaussian state
    size_t num_of_modes = occupancy.size();

    unsigned long long permutation_idx_max = power_of_2( (unsigned long long) num_of_modes);

    return calculate(1, 1, permutation_idx_max );
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
PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type>::calculate(unsigned long long start_idx, unsigned long long step_idx, unsigned long long max_idx ) {
    using complex_type = cplx_select_t<scalar_type>;

    if (this->mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return Complex16(1,0);
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

    // number of modes spanning the gaussian state
    size_t num_of_modes = occupancy.size();

    // create task group to spawn tasks
    tbb::task_group tg;

    // thread local storage for partial hafnian
    tbb::combinable<cplxm_select_t<scalar_type>> priv_addend{[](){return cplxm_select_t<scalar_type>();}};

#ifdef GLYNN
    PicVector<char> selected_modes;
    selected_modes.reserve(num_of_modes);
    for (size_t idx = 0; idx < num_of_modes; idx++) {
        if (occupancy[idx] > 0) selected_modes.push_back(idx);
    }
    PicState_int64 current_occupancy(selected_modes.size());
    int minidx = *std::min_element(selected_modes.begin(), selected_modes.end(), [this](const char a, const char b) { return occupancy[a] < occupancy[b]; });
    for (size_t idx=0;idx<selected_modes.size(); idx++) {
        current_occupancy[idx] = 0;
    }
    PicState_int64 adjoccupancy = occupancy.copy();
    adjoccupancy[minidx]--;
    IterateOverSelectedModes( selected_modes, current_occupancy, 0, priv_addend, tg, adjoccupancy );
#else
    if (start_idx<1) {
        std::cout << "start_idx must be at least 1" << std::endl;
        exit(-1);
    }
    // for cycle over the combinations of occupancy
    tbb::parallel_for(start_idx, max_idx, step_idx, [&](unsigned long long permutation_idx) {
    //for (unsigned long long permutation_idx = 1; permutation_idx < permutation_idx_max; permutation_idx++) {


            // select modes corresponding to the binary representation of permutation_idx
            PicVector<char> selected_modes;
            selected_modes.reserve(num_of_modes);
            size_t idx = 0;
            for (int i = 1 << (num_of_modes-1); i > 0; i = i / 2) {
                if (permutation_idx & i) {
                    selected_modes.push_back(idx);
                }
                idx++;
            }

            // spawn iterations over the occupied numbers of the selected modes

            // initial filling of the occupancy
            bool skip_contribution = false;
            PicState_int64 current_occupancy(selected_modes.size());
            for (size_t idx=0;idx<selected_modes.size(); idx++) {
                if (occupancy[selected_modes[idx]] > 0 ) {
                     current_occupancy[idx] = 1;
                }
                else {
                    skip_contribution = true;
                    break;
                }
            }

            if (skip_contribution) {
                //if the maximal occupancy of one mode is zero, we skip this contribution
                return;
            }

            // start task over iterations on selected column-pairs
            IterateOverSelectedModes( selected_modes, current_occupancy, 0, priv_addend, tg, occupancy );



    //}

    });
#endif

    // wait until all spawned tasks are completed
    tg.wait();


    complex_type hafnian( 0.0, 0.0 );
    priv_addend.combine_each([&](cplxm_select_t<scalar_type> &a) {
        hafnian += a.get();
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
    hafnian *= pow(this->scale_factor, sum(occupancy));
#ifdef GLYNN
    hafnian /= (1ULL << (sum(occupancy)-1));
#endif

    return Complex16(hafnian.real(), hafnian.imag());
}

/**
@brief Call to run iterations over the selected modes to calculate partial hafnians
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param mode_to_iterate The mode for which the occupancy numbers are iterated
@param priv_addend Therad local storage for the partial hafnians
@param tg Reference to a tbb::task_group
*/
template <class small_scalar_type, class scalar_type>
void
PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type>::IterateOverSelectedModes( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, size_t mode_to_iterate, tbb::combinable<cplxm_select_t<scalar_type>>& priv_addend, tbb::task_group &tg, const PicState_int64& adjoccupancy ) {


    // spawn iteration over the next mode if available
    size_t new_mode_to_iterate = mode_to_iterate+1;
    while ( new_mode_to_iterate < selected_modes.size() ) {


        // prevent the exponential explosion of spawned tasks (and save the stack space)
        // and spawn new task only if the current number of tasks is smaller than a cutoff
        if (task_num < max_task_num) {


            if ( current_occupancy[new_mode_to_iterate] < adjoccupancy[selected_modes[new_mode_to_iterate]]) {

                {
                    tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                    task_num++;
                    //std::cout << "task num: " << task_num << std::endl;
                }

                tg.run( [this, new_mode_to_iterate, selected_modes, current_occupancy, &priv_addend, &tg, &adjoccupancy ]() {

                    PicState_int64 current_occupancy_new = current_occupancy.copy();
                    current_occupancy_new[new_mode_to_iterate]++;
                    IterateOverSelectedModes( selected_modes, current_occupancy_new, new_mode_to_iterate, priv_addend, tg, adjoccupancy );

                    {
                        tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                        task_num--;
                    }

                    return;

                });


            }

        }
        else {
           // if the current number of tasks is greater than the maximal number of tasks, than the task is sequentialy
           if ( current_occupancy[new_mode_to_iterate] < adjoccupancy[selected_modes[new_mode_to_iterate]]) {
                PicState_int64 current_occupancy_new = current_occupancy.copy();
                current_occupancy_new[new_mode_to_iterate]++;
                IterateOverSelectedModes( selected_modes, current_occupancy_new, new_mode_to_iterate, priv_addend, tg, adjoccupancy );
            }


        }

        new_mode_to_iterate++;


    }


    // spawn task on the next filling factor value of the mode labeled by mode_to_iterate
    if ( current_occupancy[mode_to_iterate] < adjoccupancy[selected_modes[mode_to_iterate]]) {

        // prevent the exponential explosion of spawned tasks (and save the stack space)
        // and spawn new task only if the current number of tasks is smaller than a cutoff
        if (task_num < max_task_num) {
            {
                tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                task_num++;
                //std::cout << "task num: " << task_num << std::endl;
            }

            tg.run( [this, mode_to_iterate, selected_modes, current_occupancy, &priv_addend, &tg, &adjoccupancy ](){

                PicState_int64 current_occupancy_new = current_occupancy.copy();
                current_occupancy_new[mode_to_iterate]++;
                IterateOverSelectedModes( selected_modes, current_occupancy_new, mode_to_iterate, priv_addend, tg, adjoccupancy );
                {
                    tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                    task_num--;
                }

                return;
            });

        }
        else {
            // if the current number of tasks is greater than the maximal number of tasks, than the task is sequentialy
            PicState_int64 current_occupancy_new = current_occupancy.copy();
            current_occupancy_new[mode_to_iterate]++;
            IterateOverSelectedModes( selected_modes, current_occupancy_new, mode_to_iterate, priv_addend, tg, adjoccupancy );
        }

    }

/*
std::cout << std::endl;
std::cout << "mode_to_iterate " << mode_to_iterate << " " << "number of selected modes " << selected_modes.size() << std::endl;

std::cout << "selected modes ";
for (size_t idx=0; idx<selected_modes.size(); idx++) {
std::cout << (short)selected_modes[idx];
}
std::cout << std::endl;
std::cout << "current_occupancy ";
for (size_t idx=0; idx<current_occupancy.size(); idx++) {
std::cout << current_occupancy[idx];
}
std::cout << std::endl;
*/


    // calculate the partial hafnian for the given filling factors of the selected occupancy
    cplx_select_t<scalar_type> partial_hafnian = CalculatePartialHafnian( selected_modes, current_occupancy);

    // add partial hafnian to the sum including the combinatorial factors
    unsigned long long combinatorial_fact = 1;
    for (size_t idx=0; idx < selected_modes.size(); idx++) {
        combinatorial_fact = combinatorial_fact * binomialCoeffInt64(adjoccupancy[selected_modes[idx]], // the maximal allowed occupancy
                                                                 current_occupancy[idx] // the current occupancy
                                                                 );
    }

    cplxm_select_t<scalar_type> &hafnian_priv = priv_addend.local();
//std::cout << "combinatorial_fact " << combinatorial_fact << std::endl;
//std::cout << "partial_hafnian " << partial_hafnian << std::endl;
    hafnian_priv += partial_hafnian * (scalar_type)combinatorial_fact;





}


/**
@brief Call to calculate the partial hafnian for given selected modes and their occupancies
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@return Returns with the calculated partial hafnian
*/
template <class small_scalar_type, class scalar_type>
cplx_select_t<scalar_type>
PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type>::CalculatePartialHafnian( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy ) {
    using complex_type = cplx_select_t<scalar_type>;
    using matrix_type = mtx_select_t<complex_type>;

    size_t total_num_of_modes = sum(occupancy);
    size_t dim = total_num_of_modes*2;
#ifdef GLYNN
    size_t num_of_modes = total_num_of_modes; 
    bool fact = sum(current_occupancy) % 2;
#else
    size_t num_of_modes = sum(current_occupancy);
    // fact corresponds to the (-1)^{(n/2) - |Z|} prefactor from Eq (3.24) in arXiv 1805.12498
    bool fact = ((total_num_of_modes - num_of_modes) % 2);
#endif


    // matrix B corresponds to A^(Z), i.e. to the square matrix constructed from
    small_scalar_type scale_factor_B = 0.0;
    mtx_select_t<cplx_select_t<small_scalar_type>>&& B = CreateAZ(selected_modes, current_occupancy, num_of_modes, scale_factor_B);

    // calculating Tr(B^j) for all j's that are 1<=j<=dim/2
    // this is needed to calculate f_G(Z) defined in Eq. (3.17b) of arXiv 1805.12498
    matrix_type traces(total_num_of_modes, 1);
#ifndef GLYNN
    if (num_of_modes != 0) {
#endif
        //traces = calc_power_traces<matrix32, complex_type>(B, total_num_of_modes);
        CalcPowerTraces<small_scalar_type, scalar_type>(B, total_num_of_modes, traces);
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
    matrix_type aux0(total_num_of_modes + 1, 1);
    matrix_type aux1(total_num_of_modes + 1, 1);
    //memset( aux0.get_data(), 0.0, (total_num_of_modes + 1)*sizeof(complex_type));
    //memset( aux1.get_data(), 0.0, (total_num_of_modes + 1)*sizeof(complex_type));
    std::uninitialized_fill_n(aux0.get_data(), total_num_of_modes + 1, complex_type(0.0, 0.0));
    std::uninitialized_fill_n(aux1.get_data(), total_num_of_modes + 1, complex_type(0.0, 0.0));        
    aux0[0] = 1.0;
    // pointers to the auxiliary data arrays
    complex_type *p_aux0=NULL, *p_aux1=NULL;
    double inverse_scale_factor = 1/scale_factor_B; // the (1/scale_factor_B)^idx power of the local scaling factor of matrix B to scale the power trace
    for (size_t idx = 1; idx <= total_num_of_modes; idx++) {


        complex_type factor = traces[idx - 1] * inverse_scale_factor / (2.0 * idx);
        complex_type powfactor(1.0,0.0);

        // refresh the scaling factor
        inverse_scale_factor = inverse_scale_factor/scale_factor_B;



        if (idx%2 == 1) {
            p_aux0 = aux0.get_data();
            p_aux1 = aux1.get_data();
        }
        else {
            p_aux0 = aux1.get_data();
            p_aux1 = aux0.get_data();
        }

        //memcpy(p_aux1, p_aux0, (total_num_of_modes+1)*sizeof(complex_type) );
        std::copy_n(p_aux0, total_num_of_modes+1, p_aux1);

        for (size_t jdx = 1; jdx <= (dim / (2 * idx)); jdx++) {
            powfactor *= factor / ((double)jdx);


            for (size_t kdx = idx * jdx + 1; kdx <= total_num_of_modes + 1; kdx++) {
                p_aux1[kdx-1] += p_aux0[kdx-idx*jdx - 1]*powfactor;
            }



        }


    }
    complex_type res = p_aux1[total_num_of_modes];
    for (size_t n = aux0.size(); n > 0; --n) aux0[n-1].~complex_type();
    for (size_t n = aux1.size(); n > 0; --n) aux1[n-1].~complex_type();
    for (size_t n = traces.size(); n > 0; --n) traces[n-1].~complex_type();
    for (size_t n = B.size(); n > 0; --n) B[n-1].~cplx_select_t<small_scalar_type>();


    if (fact) {
        return -res;
//std::cout << -p_aux1[total_num_of_modes] << std::endl;
    }
    else {
        return res;
//std::cout << p_aux1[total_num_of_modes] << std::endl;
    }




}


/**
@brief Call to construct matrix \f$ A^Z \f$ (see the text below Eq. (3.20) of arXiv 1805.12498) for the given modes and their occupancy
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param num_of_modes The number of modes (including degeneracies) that have been previously calculated. (it is the sum of values in current_occupancy)
@param scale_factor_AZ The scale factor that has been used to scale the matrix elements of AZ =returned by reference)
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
template <class small_scalar_type, class scalar_type>
mtx_select_t<cplx_select_t<small_scalar_type>>
PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type>::CreateAZ( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, const size_t& num_of_modes, small_scalar_type &scale_factor_AZ  ) {


//std::cout << "A" << std::endl;
#ifdef GLYNN
    matrix A(num_of_modes*2, num_of_modes*2);
    memset(A.get_data(), 0, A.size()*sizeof(Complex16));
    size_t row_idx = 0;
    for (size_t mode_idx = 0; mode_idx < selected_modes.size(); mode_idx++) {

        size_t row_offset_mtx_a = 2*selected_modes[mode_idx]*this->mtx.stride;
        size_t row_offset_mtx_aconj = (2*selected_modes[mode_idx]+1)*this->mtx.stride;

        for (size_t filling_factor_row=1; filling_factor_row<=occupancy[selected_modes[mode_idx]]; filling_factor_row++) {

            size_t row_offset_A_a = 2*row_idx*A.stride;
            size_t row_offset_A_aconj = (2*row_idx+1)*A.stride;


            size_t col_idx = 0;

            for (size_t mode_jdx = 0; mode_jdx < selected_modes.size(); mode_jdx++) {


                for (size_t filling_factor_col=1; filling_factor_col<=occupancy[selected_modes[mode_jdx]]; filling_factor_col++) {
                    bool neg = filling_factor_col <= current_occupancy[mode_jdx];
                    if (neg) {
                        A[row_offset_A_a + col_idx*2] = -this->mtx[row_offset_mtx_a + (selected_modes[mode_jdx]*2)];
                        A[row_offset_A_aconj + col_idx*2+1] = -this->mtx[row_offset_mtx_aconj + (selected_modes[mode_jdx]*2+1)];
                        A[row_offset_A_a + col_idx*2+1] = -this->mtx[row_offset_mtx_a + (selected_modes[mode_jdx]*2+1)];
                        A[row_offset_A_aconj + col_idx*2] = -this->mtx[row_offset_mtx_aconj + (selected_modes[mode_jdx]*2)];
                    } else {
                        A[row_offset_A_a + col_idx*2] = this->mtx[row_offset_mtx_a + (selected_modes[mode_jdx]*2)];
                        A[row_offset_A_aconj + col_idx*2+1] = this->mtx[row_offset_mtx_aconj + (selected_modes[mode_jdx]*2+1)];
                        A[row_offset_A_a + col_idx*2+1] = this->mtx[row_offset_mtx_a + (selected_modes[mode_jdx]*2+1)];
                        A[row_offset_A_aconj + col_idx*2] = this->mtx[row_offset_mtx_aconj + (selected_modes[mode_jdx]*2)];
                    }
                    col_idx++;
                }
            }


            row_idx++;
        }

    }    
#else
    matrix A(num_of_modes*2, num_of_modes*2);
    memset(A.get_data(), 0, A.size()*sizeof(Complex16));
    size_t row_idx = 0;
    for (size_t mode_idx = 0; mode_idx < selected_modes.size(); mode_idx++) {

        size_t row_offset_mtx_a = 2*selected_modes[mode_idx]*this->mtx.stride;
        size_t row_offset_mtx_aconj = (2*selected_modes[mode_idx]+1)*this->mtx.stride;

        for (size_t filling_factor_row=1; filling_factor_row<=current_occupancy[mode_idx]; filling_factor_row++) {

            size_t row_offset_A_a = 2*row_idx*A.stride;
            size_t row_offset_A_aconj = (2*row_idx+1)*A.stride;


            size_t col_idx = 0;

            for (size_t mode_jdx = 0; mode_jdx < selected_modes.size(); mode_jdx++) {


                for (size_t filling_factor_col=1; filling_factor_col<=current_occupancy[mode_jdx]; filling_factor_col++) {

                    A[row_offset_A_a + col_idx*2] = this->mtx[row_offset_mtx_a + (selected_modes[mode_jdx]*2)];
                    A[row_offset_A_aconj + col_idx*2+1] = this->mtx[row_offset_mtx_aconj + (selected_modes[mode_jdx]*2+1)];
                    A[row_offset_A_a + col_idx*2+1] = this->mtx[row_offset_mtx_a + (selected_modes[mode_jdx]*2+1)];
                    A[row_offset_A_aconj + col_idx*2] = this->mtx[row_offset_mtx_aconj + (selected_modes[mode_jdx]*2)];
                    col_idx++;
                }
            }


            row_idx++;
        }

    }
#endif
/*
{
    tbb::spin_mutex::scoped_lock my_lock{mymutex};
    A.print_matrix();
}
*/
    // A^(Z), i.e. to the square matrix constructed from the input matrix
    // for details see the text below Eq.(3.20) of arXiv 1805.12498
    mtx_select_t<cplx_select_t<small_scalar_type>> AZ(num_of_modes*2, num_of_modes*2);
    for (size_t idx = 0; idx < 2*num_of_modes; idx++) {
        size_t row_offset = (idx^1)*A.stride;
        for (size_t jdx = 0; jdx < 2*num_of_modes; jdx++) {
            cplx_select_t<small_scalar_type> element(A[row_offset + jdx].real(), A[row_offset + jdx].imag());
            ::new (&AZ[idx*AZ.stride + jdx]) cplx_select_t<small_scalar_type>(element);
        }
    }

    scaleMatrix(AZ, scale_factor_AZ);


/*
    // matrix B corresponds to A^(Z), i.e. to the square matrix constructed from
    // the elements of mtx=A indexed by the rows and colums, where the binary representation of
    // permutation_idx was 1
    // for details see the text below Eq.(3.20) of arXiv 1805.12498
    matrix B(total_num_of_modes*2, total_num_of_modes*2);
    for (size_t idx = 0; idx < total_num_of_modes; idx++) {

        size_t row_offset_B_a = 2*idx*B.stride;
        size_t row_offset_B_aconj = (2*idx+1)*B.stride;

        size_t row_offset_mtx_a = 2*selected_modes[idx]*mtx.stride;
        size_t row_offset_mtx_aconj = (2*selected_modes[idx]+1)*mtx.stride;

        for (size_t jdx = 0; jdx < total_num_of_modes; jdx++) {
            B[row_offset_B_a + jdx*2]   = mtx[row_offset_mtx_a + ((selected_modes[jdx]*2) ^ 1)];
            B[row_offset_B_a + jdx*2+1] = mtx[row_offset_mtx_a + ((selected_modes[jdx]*2+1) ^ 1)];
            B[row_offset_B_aconj + jdx*2]   = mtx[row_offset_mtx_aconj + ((selected_modes[jdx]*2) ^ 1)];
            B[row_offset_B_aconj + jdx*2+1] = mtx[row_offset_mtx_aconj + ((selected_modes[jdx]*2+1) ^ 1)];
        }
    }
*/

    return AZ;


}


/**
@brief Call to scale the input matrix according to according to Eq (2.11) of in arXiv 1805.12498
*/
template <class small_scalar_type, class scalar_type>
void
PowerTraceHafnianRecursive_Tasks<small_scalar_type, scalar_type>::ScaleMatrix() {
    PowerTraceHafnian<small_scalar_type, scalar_type>::ScaleMatrix();

}

template class PowerTraceHafnianRecursive_Tasks<double, double>;
template class PowerTraceHafnianRecursive_Tasks<double, long double>;
template class PowerTraceHafnianRecursive_Tasks<long double, long double>;

template class PowerTraceHafnianRecursive_Tasks<RationalInf, RationalInf>;

} // PIC
