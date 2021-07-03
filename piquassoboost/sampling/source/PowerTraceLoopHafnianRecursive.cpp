#ifndef LONG_DOUBLE_CUTOFF
#define LONG_DOUBLE_CUTOFF 50
#endif // LONG_DOUBLE_CUTOFF

#include <iostream>
#include "PowerTraceLoopHafnianRecursive.h"
#include "PowerTraceHafnianUtilities.hpp"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
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
PowerTraceLoopHafnianRecursive::PowerTraceLoopHafnianRecursive( matrix &mtx_in, PicState_int64& occupancy_in ) {
    assert(isSymmetric(mtx_in));

    mtx = mtx_in;
    occupancy = occupancy_in;

    diag = matrix(mtx.rows,1);
    for (size_t idx=0; idx<mtx.rows; idx++) {
        diag[idx] = mtx[idx*mtx.stride + idx];
    }

}


/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state.)
@param diag_elements_in
@param occupancy_in An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
PowerTraceLoopHafnianRecursive::PowerTraceLoopHafnianRecursive( matrix &mtx_in, matrix &diag_in, PicState_int64& occupancy_in ) {
    assert(isSymmetric(mtx_in));


    mtx = mtx_in;
    diag = diag_in;
    occupancy = occupancy_in;
    


}


/**
@brief Destructor of the class.
*/
PowerTraceLoopHafnianRecursive::~PowerTraceLoopHafnianRecursive() {


}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
Complex16
PowerTraceLoopHafnianRecursive::calculate() {

    if (mtx.rows == 0) {
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

    PowerTraceLoopHafnianRecursive_Tasks hafnian_calculator = PowerTraceLoopHafnianRecursive_Tasks(mtx, diag, occupancy);
    Complex16 hafnian = hafnian_calculator.calculate(current_rank+1, world_size, permutation_idx_max);

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

    PowerTraceLoopHafnianRecursive_Tasks hafnian_calculator = PowerTraceLoopHafnianRecursive_Tasks(mtx, diag, occupancy);
    Complex16 hafnian = hafnian_calculator.calculate(current_rank+1, world_size, permutation_idx_max);

    return hafnian;
#endif


}
















/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state.)
@param occupancy_in An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
PowerTraceLoopHafnianRecursive_Tasks::PowerTraceLoopHafnianRecursive_Tasks( matrix &mtx_in, matrix &diag_in, PicState_int64& occupancy_in ) {

    assert(isSymmetric(mtx_in));

    Update_mtx( mtx_in );
    occupancy = occupancy_in;


    if (mtx.rows != 2*occupancy.size()) {
        std::cout << "The length of array occupancy should be equal to the half of the dimension of the input matrix mtx. Exiting" << std::endl;
        exit(-1);
    }

    if (mtx.rows % 2 != 0) {
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

    diag = diag_in;

}



/**
@brief Constructor of the class.
@param mtx_in A symmetric matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state.)
@param occupancy_in An \f$ n \f$ long array describing the number of rows an columns to be repeated during the hafnian calculation.
The \f$ 2*i \f$-th and  \f$ (2*i+1) \f$-th rows and columns are repeated occupancy[i] times.
(The matrix mtx itself does not contain any repeated rows and column.)
@return Returns with the instance of the class.
*/
PowerTraceLoopHafnianRecursive_Tasks::PowerTraceLoopHafnianRecursive_Tasks( matrix &mtx_in, PicState_int64& occupancy_in ) {
    assert(isSymmetric(mtx_in));

    Update_mtx( mtx_in );
    occupancy = occupancy_in;


    if (mtx.rows != 2*occupancy.size()) {
        std::cout << "The length of array occupancy should be equal to the half of the dimension of the input matrix mtx. Exiting" << std::endl;
        exit(-1);
    }

    if (mtx.rows % 2 != 0) {
        // The dimensions of the matrix should be even
        std::cout << "In PowerTraceHafnianRecursive_Tasks algorithm the dimensions of the matrix should strictly be even. Exiting" << std::endl;
        exit(-1);
    }

    diag = matrix(mtx.rows,1);
    for (size_t idx=0; idx<mtx.rows; idx++) {
        diag[idx] = mtx[idx*mtx.stride + idx];
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
PowerTraceLoopHafnianRecursive_Tasks::~PowerTraceLoopHafnianRecursive_Tasks() {
}



/**
@brief Call to calculate the partial hafnian for given selected modes and their occupancies
@param selected_modes Selected column pairs over which the iterations are run
@param current_occupancy Current occupancy of the selected column pairs for which the partial hafnian is calculated
@return Returns with the calculated hafnian
*/
Complex32
PowerTraceLoopHafnianRecursive_Tasks::CalculatePartialHafnian( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy ) {


//return Complex32(0.0,0.0);
    size_t num_of_modes = sum(current_occupancy);
    size_t total_num_of_modes = sum(occupancy);
    size_t dim = total_num_of_modes*2;


    // matrix B corresponds to A^(Z), i.e. to the square matrix constructed from
    double scale_factor_B = 0.0;

    matrix&& B = CreateAZ(selected_modes, current_occupancy, num_of_modes, scale_factor_B);

    // diag_elements corresponds to the diagonal elements of the input  matrix used in the loop correction
    matrix&& diag_elements = CreateDiagElements(selected_modes, current_occupancy, num_of_modes);



    // select the X transformed diagonal elements for the loop correction (operator X is the direct sum of sigma_x operators)
    matrix cx_diag_elements(num_of_modes*2, 1);
    for (size_t idx = 1; idx < diag_elements.size(); idx=idx+2) {
        cx_diag_elements[idx] = diag_elements[idx-1];
        cx_diag_elements[idx-1] = diag_elements[idx];
    }

    // calculating Tr(B^j) for all j's that are 1<=j<=dim/2 and loop corrections
    // this is needed to calculate f_G(Z) defined in Eq. (3.17b) of arXiv 1805.12498
    matrix32 traces;
    matrix32 loop_corrections;
    if (num_of_modes != 0) {
        CalcPowerTracesAndLoopCorrections(cx_diag_elements, diag_elements, B, total_num_of_modes, traces, loop_corrections);
    }
    else{
        // in case we have no 1's in the binary representation of permutation_idx we get zeros
        // this occurs once during the calculations
        traces = matrix32(total_num_of_modes, 1);
        loop_corrections = matrix32(total_num_of_modes, 1);
        memset( traces.get_data(), 0.0, traces.size()*sizeof(Complex32));
        memset( loop_corrections.get_data(), 0.0, loop_corrections.size()*sizeof(Complex32));
    }

    // fact corresponds to the (-1)^{(n/2) - |Z|} prefactor from Eq (3.24) in arXiv 1805.12498
    bool fact = ((total_num_of_modes - num_of_modes) % 2);

    // auxiliary data arrays to evaluate the second part of Eqs (3.24) and (3.21) in arXiv 1805.12498
    matrix32 aux0(total_num_of_modes + 1, 1);
    matrix32 aux1(total_num_of_modes + 1, 1);
    memset( aux0.get_data(), 0.0, (total_num_of_modes + 1)*sizeof(Complex32));
    memset( aux1.get_data(), 0.0, (total_num_of_modes + 1)*sizeof(Complex32));
    aux0[0] = 1.0;
    // pointers to the auxiliary data arrays
    Complex32 *p_aux0=NULL, *p_aux1=NULL;

    double inverse_scale_factor = 1/scale_factor_B; // the (1/scale_factor_B)^idx power of the local scaling factor of matrix B to scale the power trace
    double inverse_scale_factor_loop = 1; // the (1/scale_factor_B)^(idx-1) power of the local scaling factor of matrix B to scale the loop correction
    for (size_t idx = 1; idx <= total_num_of_modes; idx++) {

        Complex32 factor = traces[idx - 1] * inverse_scale_factor / (2.0 * idx) + loop_corrections[idx-1] * 0.5 * inverse_scale_factor_loop;
        Complex32 powfactor(1.0,0.0);

        // refresh the scaling factors
        inverse_scale_factor = inverse_scale_factor/scale_factor_B;
        inverse_scale_factor_loop = inverse_scale_factor_loop/scale_factor_B;

        if (idx%2 == 1) {
            p_aux0 = aux0.get_data();
            p_aux1 = aux1.get_data();
        }
        else {
            p_aux0 = aux1.get_data();
            p_aux1 = aux0.get_data();
        }

        memcpy(p_aux1, p_aux0, (total_num_of_modes+1)*sizeof(Complex32) );

        for (size_t jdx = 1; jdx <= (dim / (2 * idx)); jdx++) {
            powfactor = powfactor * factor / ((double)jdx);


            for (size_t kdx = idx * jdx + 1; kdx <= total_num_of_modes + 1; kdx++) {
                p_aux1[kdx-1] += p_aux0[kdx-idx*jdx - 1]*powfactor;
            }



        }


    }

    if (fact) {
//std::cout << -p_aux1[total_num_of_modes] << std::endl;
        return  -p_aux1[total_num_of_modes];

    }
    else {
//std::cout << p_aux1[total_num_of_modes] << std::endl;
        return p_aux1[total_num_of_modes];

    }



}

/**
@brief Call to scale the input matrix according to according to Eq (2.14) of in arXiv 1805.12498
@param mtx_in Input matrix defined by
*/
void
PowerTraceLoopHafnianRecursive_Tasks::ScaleMatrix() {

    // scale the matrix to have the mean magnitudes matrix elements equal to one.
    if ( mtx_orig.rows <= 100000) {
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



/**
@brief Call to construct matrix \f$ A^Z \f$ (see the text below Eq. (3.20) of arXiv 1805.12498) for the given modes and their occupancy
@param selected_modes Selected modes over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param num_of_modes The number of modes (including degeneracies) that have been previously calculated. (it is the sum of values in current_occupancy)
@param scale_factor_AZ The scale factor that has been used to scale the matrix elements of AZ =returned by reference)
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
matrix
PowerTraceLoopHafnianRecursive_Tasks::CreateAZ( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, const size_t& num_of_modes, double &scale_factor_AZ  ) {


    matrix A(num_of_modes*2, num_of_modes*2);
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

                    A[row_offset_A_a + col_idx*2]   = mtx[row_offset_mtx_a + (selected_modes[mode_jdx]*2)];
                    A[row_offset_A_aconj + col_idx*2+1] = mtx[row_offset_mtx_aconj + (selected_modes[mode_jdx]*2+1)];  
                    A[row_offset_A_a + col_idx*2+1] = mtx[row_offset_mtx_a + (selected_modes[mode_jdx]*2+1)];
                    A[row_offset_A_aconj + col_idx*2]   = mtx[row_offset_mtx_aconj + (selected_modes[mode_jdx]*2)];
                    col_idx++;
                }
            }

            A[row_offset_A_a + 2*row_idx]       = diag[selected_modes[mode_idx]*2];
            A[row_offset_A_aconj + 2*row_idx+1] = diag[selected_modes[mode_idx]*2+1];

            row_idx++;
        }

    }
/*
{
    tbb::spin_mutex::scoped_lock my_lock{mymutex};
    A.print_matrix();
}
*/
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
@brief Call to create diagonal elements corresponding to the diagonal elements of the input  matrix used in the loop correction
@param selected_modes Selected columns pairs over which the iterations are run
@param current_occupancy Current occupancy of the selected modes for which the partial hafnian is calculated
@param num_of_modes The number of modes (including degeneracies) that have been previously calculated. (it is the sum of values in current_occupancy)
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
matrix
PowerTraceLoopHafnianRecursive_Tasks::CreateDiagElements( const PicVector<char>& selected_modes, const PicState_int64& current_occupancy, const size_t& num_of_modes ) {


    matrix diag_elements(1, num_of_modes*2);

    size_t row_idx = 0;
    for (size_t mode_idx = 0; mode_idx < selected_modes.size(); mode_idx++) {

        size_t row_offset_mtx_a = 2*selected_modes[mode_idx]*mtx.stride;
        size_t row_offset_mtx_aconj = (2*selected_modes[mode_idx]+1)*mtx.stride;

        for (size_t filling_factor_row=1; filling_factor_row<=current_occupancy[mode_idx]; filling_factor_row++) {

            diag_elements[2*row_idx]   = diag[selected_modes[mode_idx]*2];
            diag_elements[2*row_idx+1] = diag[selected_modes[mode_idx]*2 + 1];

            row_idx++;
        }

    }


    return diag_elements;
}





} // PIC
