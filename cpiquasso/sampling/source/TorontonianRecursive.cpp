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

    // calculate the Cholesky decomposition of the initial matrix to be later reused
    matrix L = mtx.copy();
    calc_cholesky_decomposition(L,0);


    // calculate the partial Torontonian for the selected index holes
 //   long double torontonian_old = CalculatePartialTorontonian( selected_index_holes, 0 );

//        std::cout << "full cholesky: "       <<   torontonian_old <<  std::endl;



    long double torontonian = CalculatePartialTorontonian( selected_index_holes, L, num_of_modes);


//std::cout << "full cholesky reuse: " <<   torontonian <<  std::endl;
//exit(-1);

    // add the first index hole in prior to the iterations
    selected_index_holes.push_back(num_of_modes-1);

    // start task iterations originating from the initial selected modes
    IterateOverSelectedModes( selected_index_holes, 0, L, (num_of_modes-1)*1, priv_addend, tg );

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
#elif  BLAS==2 //OpenBLAS
    openblas_set_num_threads(NumThreads);
#endif

    // last correction comming from an empty submatrix contribution
    double factor =
                (num_of_modes) % 2
                    ? -1.0D
                    : 1.0D;

    torontonian = torontonian + factor;

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
TorontonianRecursive_Tasks::IterateOverSelectedModes( const PicVector<char>& selected_index_holes, int hole_to_iterate, matrix &L, const size_t reuse_index, tbb::combinable<long double>& priv_addend, tbb::task_group &tg ) {
/*
for (size_t idx = 0; idx<selected_index_holes.size(); idx++) {
std::cout << (short)selected_index_holes[idx] << ", ";
}
std::cout << std::endl;
*/
//std::cout << "reuse_index: " << reuse_index << std::endl;

    // calculate the partial Torontonian for the selected index holes
    size_t index_min;
    size_t index_max;
    if ( hole_to_iterate == 0 ) {
        index_min = 0;
        index_max = selected_index_holes[hole_to_iterate]+1;
    }
    else if (hole_to_iterate>0 ) {
        index_min = selected_index_holes[hole_to_iterate-1]+1;
        index_max = selected_index_holes[hole_to_iterate]+1;

    }


    // iterations over the selected index hole to calculate partial torontonians
    for (size_t idx=index_min; idx<index_max; idx++) {

        PicVector<char> new_selected_index_holes = selected_index_holes;
        new_selected_index_holes[hole_to_iterate] = idx;
        size_t reuse_index_new = idx-hole_to_iterate < reuse_index ? idx-hole_to_iterate : reuse_index;

//if ( idx == index_min )    reuse_index_new = new_selected_index_holes[0];//idx-1;
/*
std::cout << " bb "  << idx << " " << reuse_index_new << std::endl;
for (size_t kdx = 0; kdx<new_selected_index_holes.size(); kdx++) {
std::cout << (short)new_selected_index_holes[kdx] << ", ";
}
std::cout << std::endl;
*/
        long double partial_torontonian = CalculatePartialTorontonian( new_selected_index_holes, L, reuse_index_new );
        long double &torontonian_priv = priv_addend.local();
        torontonian_priv += partial_torontonian;
    }


    // add new index hole to the iterations
    if (selected_index_holes.size() == num_of_modes-1) return;

    int new_hole_to_iterate = hole_to_iterate+1;
    for (size_t idx=index_min; idx<index_max-1; idx++) {

        PicVector<char> new_selected_index_holes = selected_index_holes;
        new_selected_index_holes[hole_to_iterate] = idx;
        new_selected_index_holes.push_back(this->num_of_modes-1);

//std::cout << "creating L_new" << std::endl;
        size_t reuse_index_new = new_selected_index_holes[hole_to_iterate] +1 - selected_index_holes.size();//new_selected_index_holes[0];


        matrix &&L_new = CreateAZ(new_selected_index_holes, L, reuse_index_new);
        calc_cholesky_decomposition(L_new, 2*reuse_index_new);
        reuse_index_new = L_new.rows/2-1;//num_of_modes-new_selected_index_holes.size()-1;

//std::cout << "reuse_index_new: " << reuse_index_new << std::endl;

//L.print_matrix();
//L_new.print_matrix();
//std::cout << "reuse index new: " << reuse_index_new << std::endl;
/*
std::cout << "new selected holes: ";
for (size_t kdx = 0; kdx<new_selected_index_holes.size(); kdx++) {
std::cout << (short)new_selected_index_holes[kdx] << ", ";
}
std::cout << std::endl;
*/
        IterateOverSelectedModes( new_selected_index_holes, new_hole_to_iterate, L_new, reuse_index_new, priv_addend, tg );
    }






}


/**
@brief Call to calculate the partial hafnian for given selected modes and their occupancies
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@return Returns with the calculated hafnian
*/
long double
TorontonianRecursive_Tasks::CalculatePartialTorontonian( const PicVector<char>& selected_index_holes, matrix &L, const size_t reuse_index ) {


    size_t number_selected_modes = num_of_modes - selected_index_holes.size();


    matrix &&B = CreateAZ(selected_index_holes, L, reuse_index);

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
        determinant = calc_determinant_cholesky_decomposition(B, 2*reuse_index);
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
TorontonianRecursive_Tasks::CreateAZ( const PicVector<char>& selected_index_holes, matrix &L, const size_t reuse_index ) {

    size_t number_selected_modes = num_of_modes - selected_index_holes.size();
//std::cout << "reuse index in Create AZ: " << reuse_index << std::endl;
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
            //std::cout << idx << ", ";
        }
    }
    //std::cout << std::endl;


    matrix B(dimension_of_B, dimension_of_B);
    for (size_t idx = 0; idx < reuse_index; idx++) {

        //Complex16* mtx_data = mtx.get_data() + 2*(positions_of_ones[idx]*mtx.stride);
        Complex16* L_data = L.get_data() + 2*idx*L.stride;
        Complex16* B_data = B.get_data() + 2*idx*B.stride;

        for (size_t jdx = 0; jdx < reuse_index; jdx++) {
            memcpy( B_data + 2*jdx, L_data + 2*jdx, 2*sizeof(Complex16) );
        }

        B_data   = B_data + B.stride;
        L_data   = L_data + L.stride;

        for (size_t jdx = 0; jdx < reuse_index; jdx++) {
            memcpy( B_data + 2*jdx, L_data + 2*jdx, 2*sizeof(Complex16) );
        }

    }

    for (size_t idx = reuse_index; idx < number_selected_modes; idx++) {
//std::cout <<  (short)positions_of_ones[idx] << std::endl;
        Complex16* mtx_data = mtx.get_data() + 2*(positions_of_ones[idx]*mtx.stride);
        Complex16* B_data   = B.get_data() + 2*(idx*B.stride);

        for (size_t jdx = 0; jdx <= idx; jdx++) {
            memcpy( B_data + 2*jdx, mtx_data + 2*positions_of_ones[jdx], 2*sizeof(Complex16) );
        }

        B_data   = B_data + B.stride;
        mtx_data = mtx_data + mtx.stride;

        for (size_t jdx = 0; jdx <= idx; jdx++) {
            memcpy( B_data + 2*jdx, mtx_data + 2*positions_of_ones[jdx], 2*sizeof(Complex16) );
        }

    }



    return B;

}






} // PIC
