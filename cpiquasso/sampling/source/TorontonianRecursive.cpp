#ifndef LONG_DOUBLE_CUTOFF
#define LONG_DOUBLE_CUTOFF 50
#endif // LONG_DOUBLE_CUTOFF

#include <iostream>
#include "TorontonianRecursive.h"
#include "TorontonianUtilities.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>

#ifdef __MPI__
#include <mpi.h>
#endif // MPI





static tbb::spin_mutex my_mutex;
/*
double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {


/**
@brief Constructor of the class.
@param mtx_in A selfadjoint matrix for which the torontonian is calculated. This matrix has to be positive definite matrix with eigenvalues between 0 and 1 (for example a covariance matrix of the Gaussian state.) matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state)
@param occupancy An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
TorontonianRecursive::TorontonianRecursive( matrix &mtx_in ) {
    assert(isSymmetric(mtx_in));

    mtx = matrix32(mtx_in.rows, mtx_in.cols);
    for (size_t idx=0; idx<mtx_in.size(); idx++) {
        mtx[idx] = mtx_in[idx];
    }

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


    TorontonianRecursive_Tasks torontonian_calculator = TorontonianRecursive_Tasks(mtx);
    double torontonian = torontonian_calculator.calculate();

    return torontonian;


}











/**
@brief Nullary constructor of the class.
@return Returns with the instance of the class.
*/
TorontonianRecursive_Tasks::TorontonianRecursive_Tasks() {

}




/**
@brief Constructor of the class.
@param mtx_in A selfadjoint matrix for which the torontonian is calculated. This matrix has to be positive definite matrix with eigenvalues between 0 and 1 (for example a covariance matrix of the Gaussian state.) matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state)
@param occupancy An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
TorontonianRecursive_Tasks::TorontonianRecursive_Tasks( matrix32 &mtx_in ) {

    Update_mtx( mtx_in );

    // number of modes spanning the gaussian state
    num_of_modes = mtx.rows/2;


    if (mtx.rows % 2 != 0) {
        // The dimensions of the matrix should be even
        std::cout << "In PowerTraceHafnianRecursive_Tasks algorithm the dimensions of the matrix should strictly be even. Exiting" << std::endl;
        exit(-1);
    }


}


/**
@brief Destructor of the class.
*/
TorontonianRecursive_Tasks::~TorontonianRecursive_Tasks() {

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


    // thread local storage for partial hafnian
    tbb::combinable<RealM<long double>> priv_addend{[](){return RealM<long double>(0.0);}};

    // construct the initial selection of the modes
    PicVector<size_t> selected_index_holes;

    // calculate the Cholesky decomposition of the initial matrix to be later reused
    matrix32 L = mtx.copy();
    Complex32 determinant = calc_determinant_cholesky_decomposition(L);

    long double torontonian = CalculatePartialTorontonian( selected_index_holes, determinant);

    // add the first index hole in prior to the iterations
    selected_index_holes.push_back(num_of_modes-1);

    // start task iterations originating from the initial selected modes
    IterateOverSelectedModes( selected_index_holes, 0, L, num_of_modes-1, priv_addend );



    priv_addend.combine_each([&](RealM<long double> &a) {
        torontonian = torontonian + a.get();
    });



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
@brief Call to run iterations over the selected modes to calculate partial torontonians
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A^Z.
@param hole_to_iterate The index indicating which hole index should be iterated
@param L Matrix conatining partial Cholesky decomposition if the initial matrix to be reused
@param priv_addend Therad local storage for the partial torontonians
@param tg Reference to a tbb::task_group
*/
void
TorontonianRecursive_Tasks::IterateOverSelectedModes( const PicVector<size_t>& selected_index_holes, int hole_to_iterate, matrix32 &L, const size_t reuse_index, tbb::combinable<RealM<long double>>& priv_addend ) {

    // calculate the partial Torontonian for the selected index holes
    size_t index_min;
    size_t index_max;
    if ( hole_to_iterate == 0 ) {
        index_min = 0;
        index_max = selected_index_holes[hole_to_iterate]+1;
    }
    else {
        index_min = selected_index_holes[hole_to_iterate-1]+1;
        index_max = selected_index_holes[hole_to_iterate]+1;

    }


    // ***** iterations over the selected index hole to calculate partial torontonians *****

    // first do the first iteration without spawning iterations with new index hole

    PicVector<size_t> selected_index_holes_new = selected_index_holes;
    selected_index_holes_new[hole_to_iterate] = index_max-1;
    size_t reuse_index_new = index_max-1-hole_to_iterate < reuse_index ? index_max-1-hole_to_iterate : reuse_index;

    matrix32 &&L_new = CreateAZ(selected_index_holes_new, L, reuse_index_new);
    Complex32 determinant = calc_determinant_cholesky_decomposition(L_new, 2*reuse_index_new);

    long double partial_torontonian = CalculatePartialTorontonian( selected_index_holes_new, determinant );
    RealM<long double> &torontonian_priv = priv_addend.local();
    torontonian_priv += partial_torontonian;

    // logical variable to control whether spawning new iterations or not
    bool stop_spawning_iterations = (selected_index_holes.size() == num_of_modes-1);

    // add new index hole to the iterations
    int new_hole_to_iterate = hole_to_iterate+1;

    // now do the rest of the iterations
    tbb::parallel_for( index_min+1,  index_max, (size_t)1, [&](size_t idx){


        PicVector<size_t> selected_index_holes_new = selected_index_holes;
        selected_index_holes_new[hole_to_iterate] = idx-1;
        size_t reuse_index_new = idx-1-hole_to_iterate < reuse_index ? idx-1-hole_to_iterate : reuse_index;

        matrix32 &&L_new = CreateAZ(selected_index_holes_new, L, reuse_index_new);
        Complex32 determinant = calc_determinant_cholesky_decomposition(L_new, 2*reuse_index_new);

        long double partial_torontonian = CalculatePartialTorontonian( selected_index_holes_new, determinant );
        RealM<long double> &torontonian_priv = priv_addend.local();
        torontonian_priv += partial_torontonian;


        // return if new index hole would give no nontrivial result
        // (in this case the partial torontonian is unity and should be counted only once in function calculate)
        if (stop_spawning_iterations) return;

        PicVector<size_t> selected_index_holes_new2 = selected_index_holes_new;
        selected_index_holes_new2.push_back(this->num_of_modes-1);
        reuse_index_new = L_new.rows/2-1;

        IterateOverSelectedModes( selected_index_holes_new2, new_hole_to_iterate, L_new, reuse_index_new, priv_addend );

    });



}




/**
@brief Call to calculate the partial torontonian for given selected modes and their occupancies
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A_Z.
@param determinant The determinant of the submatrix A_Z
@return Returns with the calculated torontonian
*/
long double
TorontonianRecursive_Tasks::CalculatePartialTorontonian( const PicVector<size_t>& selected_index_holes, const Complex32 &determinant ) {


    size_t number_selected_modes = num_of_modes - selected_index_holes.size();


    // calculating -1^(N-|Z|)
    long double factor =
                (number_selected_modes + num_of_modes) % 2
                    ? -1.0
                    : 1.0;


    // calculating -1^(number of ones) / sqrt(det(1-A^(Z)))
    long double sqrt_determinant = std::sqrt(determinant.real()*scale_factors[number_selected_modes]);

    return (factor / sqrt_determinant);


}



/**
@brief Call to update the memory address of the matrix mtx and reorder the matrix elements into a_1^*,a_1^*,a_2,a_2^*, ... a_N,a_N^* order.
@param mtx_in Input matrix defined by
*/
void
TorontonianRecursive_Tasks::Update_mtx( matrix32 &mtx_in ){
    //mtx_orig = mtx_in;

    size_t dim = mtx_in.rows;

    // Calculating B := 1 - A
    mtx = matrix32(dim, dim);
    for (size_t idx = 0; idx < dim; idx++) {
        for (size_t jdx = 0; jdx < dim; jdx++) {
            mtx[idx * dim + jdx] = -1.0 * mtx_in[idx * mtx_in.stride + jdx];
        }
        mtx[idx * dim + idx] += Complex32(1.0, 0.0);
    }


    // convert the input matrix from a1, a2, ... a_N, a_1^*, a_2^* ... a_N^* format to
    // a_1^*,a_1^*,a_2,a_2^*, ... a_N,a_N^* format

    size_t num_of_modes = dim/2;
    matrix32 mtx_reordered = matrix32(dim, dim);
    for (size_t idx=0; idx<num_of_modes; idx++) {
        for (size_t jdx=0; jdx<num_of_modes; jdx++) {
            mtx_reordered[2*idx*mtx_reordered.stride + 2*jdx] = mtx[idx*mtx.stride + jdx];
            mtx_reordered[2*idx*mtx_reordered.stride + 2*jdx+1] = mtx[idx*mtx.stride + jdx + num_of_modes];
            mtx_reordered[(2*idx+1)*mtx_reordered.stride + 2*jdx] = mtx[(idx+num_of_modes)*mtx.stride + jdx];
            mtx_reordered[(2*idx+1)*mtx_reordered.stride + 2*jdx+1] = mtx[(idx+num_of_modes)*mtx.stride + jdx + num_of_modes];
        }
    }

    mtx = mtx_reordered;




    // Can scaling be used here since we have to calculate 1-A^Z?
    // It brings a multiplying for each determinant.
    // Should
    ScaleMatrix();
}


/**
@brief Call to construct matrix \f$ A^Z \f$ (see the text below Eq. (2) of arXiv 2009.01177) for the given modes
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A^Z.
@param L Matrix conatining partial Cholesky decomposition if the initial matrix to be reused
@param reuse_index Index labeling the highest mode for which previous Cholesky decomposition can be reused.
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
matrix32
TorontonianRecursive_Tasks::CreateAZ( const PicVector<size_t>& selected_index_holes, matrix32 &L, const size_t reuse_index ) {

    size_t number_selected_modes = num_of_modes - selected_index_holes.size();
//std::cout << "reuse index in Create AZ: " << reuse_index << std::endl;


    PicVector<size_t> positions_of_ones;
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

    // reuse the data in the L matrix (in place or copied to out of place
    size_t dimension_of_AZ = 2 * number_selected_modes;
    matrix32 AZ(dimension_of_AZ, dimension_of_AZ);
/*
    // The first 2*(reuse_index-1) rows of the matrix are not touched during the calculations they can be reused from Cholesky matrix L
    for (size_t idx = 0; idx < reuse_index; idx++) {

        Complex16* L_data = L.get_data() + 2*idx*L.stride;
        Complex16* AZ_data = AZ.get_data() + 2*idx*AZ.stride;

        memcpy(AZ_data, L_data, 2*(idx+1)*sizeof(Complex16));
        memcpy(AZ_data + AZ.stride, L_data + L.stride, 2*(idx+1)*sizeof(Complex16));

    }
*/

    // to calculate the determiannt only the diagonal elements of L are necessary
    for (size_t idx = 0; idx < reuse_index; idx++) {
        AZ[2*idx*AZ.stride + 2*idx] = L[2*idx*L.stride + 2*idx];
        AZ[(2*idx+1)*AZ.stride + 2*idx+1] = L[(2*idx+1)*L.stride + 2*idx + 1];
    }

    // copy data from the input matrix and the reusable partial Cholesky decomposition matrix L
    for (size_t idx = reuse_index; idx < number_selected_modes; idx++) {

        Complex32* mtx_data = mtx.get_data() + 2*(positions_of_ones[idx]*mtx.stride);
        Complex32* L_data = L.get_data() + 2*(idx+1)*L.stride;
        Complex32* AZ_data   = AZ.get_data() + 2*(idx*AZ.stride);


        memcpy(AZ_data, L_data, 2*(idx+1)*sizeof(Complex32));
        memcpy(AZ_data + AZ.stride, L_data + L.stride, 2*(idx+1)*sizeof(Complex32));

        for (size_t jdx = reuse_index; jdx <= idx; jdx++) {
            memcpy( AZ_data + 2*jdx, mtx_data + 2*positions_of_ones[jdx], 2*sizeof(Complex32) );
        }

        AZ_data   = AZ_data + AZ.stride;
        mtx_data = mtx_data + mtx.stride;

        for (size_t jdx = reuse_index; jdx <= idx; jdx++) {
            memcpy( AZ_data + 2*jdx, mtx_data + 2*positions_of_ones[jdx], 2*sizeof(Complex32) );
        }

    }

    return AZ;

}






} // PIC
