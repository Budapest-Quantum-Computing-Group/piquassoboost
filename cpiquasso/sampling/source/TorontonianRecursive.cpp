#ifndef LONG_DOUBLE_CUTOFF
#define LONG_DOUBLE_CUTOFF 50
#endif // LONG_DOUBLE_CUTOFF

#include <iostream>
#include "TorontonianRecursive.h"
#include "TorontonianUtilities.hpp"
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
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_1^*,a_1, a_1^*, ... a_n, a_n^* \f$ ordered covariance matrix of the Gaussian state,
where \f$ n \f$ is the number of occupancy i n the Gaussian state).
@param occupancy An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
TorontonianRecursive::TorontonianRecursive( matrix &mtx_in ) {
    assert(isSymmetric(mtx_in));

    mtx = mtx_in;

}


/**
@brief Destructor of the class.
*/
TorontonianRecursive::~TorontonianRecursive() {


}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
double
TorontonianRecursive::calculate() {

    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return 0.0;
    }

    // number of modes spanning the gaussian state
    size_t num_of_modes = mtx.rows/2;


    TorontonianRecursive_Tasks torontonian_calculator = TorontonianRecursive_Tasks(mtx);
    double torontonian = torontonian_calculator.calculate();

    return torontonian;


}











/**
@brief Nullary constructor of the class.
@return Returns with the instance of the class.
*/
TorontonianRecursive_Tasks::TorontonianRecursive_Tasks() {

    // set the maximal number of spawned tasks living at the same time
    max_task_num = 100;
    // The current number of spawned tasks
    task_num = 0;
    // mutual exclusion to count the spawned tasks
    task_count_mutex = new tbb::spin_mutex();

}




/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_1^*,a_1, a_1^*, ... a_n, a_n^* \f$ ordered covariance matrix of the Gaussian state,
where \f$ n \f$ is the number of occupancy i n the Gaussian state).
@param occupancy An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
TorontonianRecursive_Tasks::TorontonianRecursive_Tasks( matrix &mtx_in ) {

    Update_mtx( mtx_in );

    // number of modes spanning the gaussian state
    num_of_modes = mtx.rows/2;


    if (mtx.rows % 2 != 0) {
        // The dimensions of the matrix should be even
        std::cout << "In PowerTraceHafnianRecursive_Tasks algorithm the dimensions of the matrix should strictly be even. Exiting" << std::endl;
        exit(-1);
    }

    // set the maximal number of spawned tasks living at the same time
    max_task_num = 100;
    // The current number of spawned tasks
    task_num = 0;
    // mutual exclusion to count the spawned tasks
    task_count_mutex = new tbb::spin_mutex();


}


/**
@brief Destructor of the class.
*/
TorontonianRecursive_Tasks::~TorontonianRecursive_Tasks() {
    delete task_count_mutex;
}


/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
double
TorontonianRecursive_Tasks::calculate() {


    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return 0.0D;
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



    // create task group to spawn tasks
    tbb::task_group tg;

    // thread local storage for partial hafnian
    tbb::combinable<long double> priv_addend{[](){return 0.0L;}};

    // construct the initial selection of the modes
    PicVector<char> selected_index_holes;


    // calculate the partial Torontonian for the selected index holes
    long double torontonian = CalculatePartialTorontonian( selected_index_holes );

    // add the first index hole in prior to the iterations
    selected_index_holes.push_back(num_of_modes-1);

    // start task iterations originating from the initial selected modes
    IterateOverSelectedModes( selected_index_holes, 0, priv_addend, tg );

    // wait until all spawned tasks are completed
    tg.wait();


    priv_addend.combine_each([&](long double &a) {
        torontonian = torontonian + a;
    });


    //Complex16 res = summand;



#if BLAS==0 // undefined BLAS
    omp_set_num_threads(NumThreads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(NumThreads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(NumThreads);
#endif


    return (double)torontonian;
}

/**
@brief Call to run iterations over the selected modes to calculate partial hafnians
@param selected_modes Selected modes over which the iterations are run
@param mode_to_iterate The mode for which the occupancy numbers are iterated
@param priv_addend Therad local storage for the partial hafnians
@param tg Reference to a tbb::task_group
*/
void
TorontonianRecursive_Tasks::IterateOverSelectedModes( const PicVector<char>& selected_index_holes, int hole_to_iterate, tbb::combinable<long double>& priv_addend, tbb::task_group &tg ) {
/*
for (size_t idx = 0; idx<selected_index_holes.size(); idx++) {
std::cout << (short)selected_index_holes[idx] << ", ";
}
std::cout << std::endl;
*/
    // add new index hole to th eiterations
    if ( selected_index_holes[hole_to_iterate] < num_of_modes-1) {

        int new_hole_to_iterate = hole_to_iterate+1;

        // prevent the exponential explosion of spawned tasks (and save the stack space)
        // and spawn new task only if the current number of tasks is smaller than a cutoff
        if (task_num < max_task_num) {

            {
                tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                task_num++;
                    //std::cout << "task num: " << task_num << std::endl;
            }

            tg.run( [this, new_hole_to_iterate, selected_index_holes, &priv_addend, &tg ]() {

                PicVector<char> new_selected_index_holes = selected_index_holes;
                new_selected_index_holes.push_back(this->num_of_modes-1);
                IterateOverSelectedModes( new_selected_index_holes, new_hole_to_iterate, priv_addend, tg );

                {
                    tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                    task_num--;
                }

                return;

            });


        }
        else {
           // if the current number of tasks is greater than the maximal number of tasks, than the task is sequentialy
            PicVector<char> new_selected_index_holes = selected_index_holes;
            new_selected_index_holes.push_back(num_of_modes-1);
            IterateOverSelectedModes( new_selected_index_holes, new_hole_to_iterate, priv_addend, tg );
        }

    }

    // iterations over the selected index hole
    if ( hole_to_iterate == 0 && selected_index_holes[hole_to_iterate] > 0) {

        if (task_num < max_task_num) {

            {
                tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                task_num++;
                    //std::cout << "task num: " << task_num << std::endl;
            }

            tg.run( [this, hole_to_iterate, selected_index_holes, &priv_addend, &tg ]() {

                PicVector<char> new_selected_index_holes = selected_index_holes;
                new_selected_index_holes[hole_to_iterate]--;
                IterateOverSelectedModes( new_selected_index_holes, hole_to_iterate, priv_addend, tg );

                {
                    tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                    task_num--;
                }

                return;

            });


        }
        else {
            PicVector<char> new_selected_index_holes = selected_index_holes;
            new_selected_index_holes[hole_to_iterate]--;
            IterateOverSelectedModes( new_selected_index_holes, hole_to_iterate, priv_addend, tg );
        }

    }
    else if (hole_to_iterate>0 && selected_index_holes[hole_to_iterate] > 1+selected_index_holes[hole_to_iterate-1]) {

        if (task_num < max_task_num) {

            {
                tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                task_num++;
                    //std::cout << "task num: " << task_num << std::endl;
            }

            tg.run( [this, hole_to_iterate, selected_index_holes, &priv_addend, &tg ]() {

                PicVector<char> new_selected_index_holes = selected_index_holes;
                new_selected_index_holes[hole_to_iterate]--;
                IterateOverSelectedModes( new_selected_index_holes, hole_to_iterate, priv_addend, tg );

                {
                    tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                    task_num--;
                }

                return;

            });


        }
        else {
            PicVector<char> new_selected_index_holes = selected_index_holes;
            new_selected_index_holes[hole_to_iterate]--;
            IterateOverSelectedModes( new_selected_index_holes, hole_to_iterate, priv_addend, tg );
        }

    }



    // calculate the partial Torontonian for the selected index holes
    long double partial_torontonian = CalculatePartialTorontonian( selected_index_holes );

    long double &torontonian_priv = priv_addend.local();

    torontonian_priv += partial_torontonian;




}


/**
@brief Call to calculate the partial hafnian for given selected modes and their occupancies
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@return Returns with the calculated hafnian
*/
long double
TorontonianRecursive_Tasks::CalculatePartialTorontonian( const PicVector<char>& selected_index_holes ) {


    size_t number_selected_modes = num_of_modes - selected_index_holes.size();


    size_t dimension_of_B = 2 * number_selected_modes;

    PicVector<char> positions_of_ones;
    positions_of_ones.reserve(number_selected_modes);
    if ( selected_index_holes.size() == 0 ) {

        for (size_t idx=0; idx<num_of_modes; idx++) {
            positions_of_ones.push_back(idx);
        }

    }
    else {

        size_t hole_idx = 0;
        for (size_t idx=0; idx<num_of_modes; idx++) {

            if ( idx == selected_index_holes[hole_idx] && hole_idx<selected_index_holes.size()) {
                hole_idx++;
                continue;
            }
            positions_of_ones.push_back(idx);
        }
    }


    // matrix mtx corresponds to id - A^(Z), i.e. to the square matrix constructed from
    // the elements of mtx = 1-A indexed by the rows and colums, where the binary representation of
    // permutation_idx was 1
    // details in Eq. (12) https://arxiv.org/pdf/1807.01639.pdf
    // B = (1 - A^(Z))
    // Calculating B^(Z)
    matrix B(dimension_of_B, dimension_of_B);
    for (size_t idx = 0; idx < number_selected_modes; idx++) {

        Complex16* mtx_data = mtx.get_data() + 2*(positions_of_ones[idx]*mtx.stride);
        Complex16* B_data = B.get_data() + 2*(idx*B.stride);

        for (size_t jdx = 0; jdx < number_selected_modes; jdx++) {
            memcpy( B_data + 2*jdx, mtx_data + 2*positions_of_ones[jdx], 2*sizeof(Complex16) );
        }

        B_data   = B_data + B.stride;
        mtx_data = mtx_data + mtx.stride;

        for (size_t jdx = 0; jdx < number_selected_modes; jdx++) {
            memcpy( B_data + 2*jdx, mtx_data + 2*positions_of_ones[jdx], 2*sizeof(Complex16) );
        }

    }


            // calculating -1^(number of ones)
            // !!! -1 ^ (number of ones - dim_over_2) ???
            double factor =
                (number_selected_modes + num_of_modes) % 2
                    ? -1.0D
                    : 1.0D;

            // calculating the determinant of B
            Complex16 determinant;
            if (number_selected_modes != 0) {
                // testing purpose (the matrix is not positive definite and selfadjoint)
                //determinant = determinant_byLU_decomposition(B);
                determinant = calc_determinant_cholesky_decomposition(B);
            }
            else{
                determinant = 1.0;
            }

            // calculating -1^(number of ones) / sqrt(det(1-A^(Z)))
            double sqrt_determinant = std::sqrt(determinant.real());


            return (long double) (factor / sqrt_determinant);


}



/**
@brief Call to update the memory address of the matrix mtx
@param mtx_in Input matrix defined by
*/
void
TorontonianRecursive_Tasks::Update_mtx( matrix &mtx_in ){
    mtx_orig = mtx_in;

    size_t dim = mtx_in.rows;

    // Calculating B := 1 - A
    mtx = matrix(dim, dim);
    for (size_t idx = 0; idx < dim; idx++) {
        //Complex16 *row_B_idx = B.get_data() + idx * B.stride;
        //Complex16 *row_mtx_pos_idx = mtx.get_data() + positions_of_ones[idx] * mtx.stride;
        for (size_t jdx = 0; jdx < dim; jdx++) {
            mtx[idx * dim + jdx] = -1.0 * mtx_in[idx * mtx_in.stride + jdx];
        }
        mtx[idx * dim + idx] += Complex16(1.0, 0.0);
    }


    // convert the input matrix from a1, a2, ... a_N, a_1^*, a_2^* ... a_N^* format to
    // a_1^*,a_1^*,a_2,a_2^*, ... a_N,a_N^* format

    size_t num_of_modes = dim/2;
    matrix mtx_reordered = matrix(dim, dim);
    for (size_t idx=0; idx<num_of_modes; idx++) {
        for (size_t jdx=0; jdx<num_of_modes; jdx++) {
            mtx_reordered[2*idx*mtx_reordered.stride + 2*jdx] = mtx[idx*mtx.stride + jdx];
            mtx_reordered[2*idx*mtx_reordered.stride + 2*jdx+1] = mtx[idx*mtx.stride + jdx + num_of_modes];
            mtx_reordered[(2*idx+1)*mtx_reordered.stride + 2*jdx] = mtx[(idx+num_of_modes)*mtx.stride + jdx];
            mtx_reordered[(2*idx+1)*mtx_reordered.stride + 2*jdx+1] = mtx[(idx+num_of_modes)*mtx.stride + jdx + num_of_modes];
        }
    }

    //mtx.print_matrix();
    //mtx_reordered.print_matrix();
    mtx = mtx_reordered;

    // Can scaling be used here since we have to calculate 1-A^Z?
    // It brings a multiplying for each determinant.
    // Should
    ScaleMatrix();
}


/**
@brief Call to construct matrix \f$ A^Z \f$ (see the text below Eq. (3.20) of arXiv 1805.12498) for the given modes and their occupancy
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param num_of_modes The number of modes (including degeneracies) that have been previously calculated. (it is the sum of values in current_occupancy)
@param scale_factor_AZ The scale factor that has been used to scale the matrix elements of AZ =returned by reference)
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
matrix
TorontonianRecursive_Tasks::CreateAZ( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, const size_t& num_of_modes, double &scale_factor_AZ  ) {


//std::cout << "A" << std::endl;
    matrix AZ(num_of_modes*2, num_of_modes*2);

/*
    memset(A.get_data(), 0, A.size()*sizeof(Complex16));
    size_t row_idx = 0;
    for (size_t mode_idx = 0; mode_idx < selected_modes.size(); mode_idx++) {

        size_t row_offset_mtx_a = 2*selected_modes[mode_idx]*mtx.stride;
        size_t row_offset_mtx_aconj = (2*selected_modes[mode_idx]+1)*mtx.stride;

        for (size_t filling_factor_row=1; filling_factor_row<=current_occupancy[mode_idx]; filling_factor_row++) {

            size_t row_offset_A_a = 2*row_idx*A.stride;
            size_t row_offset_A_aconj = (2*row_idx+1)*A.stride;


            size_t col_idx = 0;

            for (size_t mode_jdx = 0; mode_jdx < selected_modes.size(); mode_jdx++) {


                for (size_t filling_factor_col=1; filling_factor_col<=current_occupancy[mode_jdx]; filling_factor_col++) {

                    if ( (row_idx == col_idx) || (mode_idx != mode_jdx) ) {

                        A[row_offset_A_a + col_idx*2]   = mtx[row_offset_mtx_a + (selected_modes[mode_jdx]*2)];
                        A[row_offset_A_aconj + col_idx*2+1] = mtx[row_offset_mtx_aconj + (selected_modes[mode_jdx]*2+1)];

                    }

                    A[row_offset_A_a + col_idx*2+1] = mtx[row_offset_mtx_a + (selected_modes[mode_jdx]*2+1)];
                    A[row_offset_A_aconj + col_idx*2]   = mtx[row_offset_mtx_aconj + (selected_modes[mode_jdx]*2)];
                    col_idx++;
                }
            }


            row_idx++;
        }

    }

    // A^(Z), i.e. to the square matrix constructed from the input matrix
    // for details see the text below Eq.(3.20) of arXiv 1805.12498
    matrix AZ(num_of_modes*2, num_of_modes*2);
    scale_factor_AZ = 0.0;
    for (size_t idx = 0; idx < 2*num_of_modes; idx++) {
        size_t row_offset = (idx^1)*A.stride;
        for (size_t jdx = 0; jdx < 2*num_of_modes; jdx++) {
            Complex16 &element = A[row_offset + jdx];
            AZ[idx*AZ.stride + jdx] = element;
            scale_factor_AZ = scale_factor_AZ + element.real()*element.real() + element.imag()*element.imag();
        }
    }


    // scale matrix AZ -- when matrix elements of AZ are scaled, larger part of the computations can be kept in double precision
    if ( scale_factor_AZ < 1e-8 ) {
        scale_factor_AZ = 1.0;
    }
    else {
        scale_factor_AZ = std::sqrt(scale_factor_AZ/2)/AZ.size();
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ[idx] = AZ[idx]*scale_factor_AZ;
        }
    }

*/


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






} // PIC
