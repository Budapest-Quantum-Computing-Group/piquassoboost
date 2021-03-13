#include <iostream>
#include "PowerTraceHafnianRecursive.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include <math.h>

extern "C" {

#define LAPACK_ROW_MAJOR               101

/// Definition of the LAPACKE_zgehrd function from LAPACKE to calculate the upper triangle hessenberg transformation of a matrix
int LAPACKE_zgehrd( int matrix_layout, int n, int ilo, int ihi, pic::Complex16* a, int lda, pic::Complex16* tau );

}

/*
tbb::spin_mutex my_mutex;

double time_nominator = 0.0;
double time_nevezo = 0.0;
*/

namespace pic {


/**
@brief Call to calculate sum of integers stored in a container
@param vec a PicState_int64 instance
@return Returns with the sum of the elements of the container
*/
static inline int
sum( PicState_int64 vec) {

    int ret = 0;
    for (size_t idx=0; idx<vec.size(); idx++) {
        if ( vec[idx] == 0) {
            continue;
        }
        ret = ret + vec[idx];
    }
    return ret;
}


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
@brief Default constructor of the class.
@param mtx_in The covariance matrix of the Gaussian state.
@param modes An array describing the modes to be used to calculate the hafnian. The i-th mode is repeated modes[i] times.
@return Returns with the instance of the class.
*/
PowerTraceHafnianRecursive::PowerTraceHafnianRecursive( matrix &mtx_in, PicState_int64& modes_in ) : PowerTraceHafnian( mtx_in ) {

    modes = modes_in;

    //TODO check num_of modes with mtx size
}


/**
@brief Destructor of the class.
*/
PowerTraceHafnianRecursive::~PowerTraceHafnianRecursive() {

}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
Complex16
PowerTraceHafnianRecursive::calculate() {

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

    //mtx = getPermutedMatrix();
    //return calculate_tmp();

    matrix&& mtx_permuted = getPermutedMatrix();

    size_t num_of_modes = modes.size();


    if (mtx.rows == 0) {
        // the hafnian of an empty matrix is 1 by definition
        return Complex16(1,0);
    }
/*
    else if (mtx.rows % 2 != 0) {
        // the hafnian of odd shaped matrix is 0 by definition
        return Complex16(0.0, 0.0);
    }
*/

    unsigned long long permutation_idx_max = power_of_2( (unsigned long long) num_of_modes);

    // for cycle over the combinations of modes -- TODO make parallel (possible entrance point of MPI)
    Complex16 hafnian(0.0,0.0);

    for (unsigned long long permutation_idx = 1; permutation_idx < permutation_idx_max; permutation_idx++) {



        // select modes corresponding to the binary representation of permutation_idx
        std::vector<unsigned char> bin_rep;
        std::vector<unsigned char> selected_modes;
        bin_rep.reserve(num_of_modes);
        selected_modes.reserve(num_of_modes);
        for (int i = 1 << (num_of_modes-1); i > 0; i = i / 2) {
            if (permutation_idx & i) {
                bin_rep.push_back(1);
                selected_modes.push_back((bin_rep.size()-1));
            }
            else {
                bin_rep.push_back(0);
            }
        }
        size_t number_of_modes = selected_modes.size();

std::cout << "selected modes: ";
for (size_t idx=0; idx<selected_modes.size(); idx++) {
    std::cout << (short) selected_modes[idx];
}
std::cout << std::endl;


        // spawn iterations over the occupied numbers of the modes

        // initial filling of the modes
        PicState_int64 filling_factors(selected_modes.size());
        for (size_t idx=0;idx<selected_modes.size(); idx++) {
            filling_factors[idx] = 1;
        }
//filling_factors.print_matrix();

        IterateOverSelectedModes( selected_modes, filling_factors, 0, hafnian );



    }


    //Complex16 res = summand;



#if BLAS==0 // undefined BLAS
    omp_set_num_threads(NumThreads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(NumThreads);
#elif BLAS==2 //OpenBLAS
    openblas_set_num_threads(NumThreads);
#endif

    return hafnian;
}

/**
@brief ??????????????????
@return Returns with the calculated hafnian
*/
void
PowerTraceHafnianRecursive::IterateOverSelectedModes( std::vector<unsigned char>& selected_modes, PicState_int64& filling_factors, size_t mode_to_iterate, Complex16& hafnian ) {

    // spawn iteration over the next mode if available
    if ( mode_to_iterate+1 < selected_modes.size()) {

        size_t new_mode_to_iterate = mode_to_iterate+1;
        if ( filling_factors[new_mode_to_iterate] < modes[selected_modes[new_mode_to_iterate]]) {
            PicState_int64 filling_factors_new = filling_factors.copy();
            filling_factors_new[new_mode_to_iterate]++;
            IterateOverSelectedModes( selected_modes, filling_factors_new, new_mode_to_iterate, hafnian ); // TODO use task_group for this
        }

    }

    // spawn task on the next filling factor value of the mode labeled by mode_to_iterate
    if ( filling_factors[mode_to_iterate] < modes[selected_modes[mode_to_iterate]]) {
        PicState_int64 filling_factors_new = filling_factors.copy();
        filling_factors_new[mode_to_iterate]++;
        IterateOverSelectedModes( selected_modes, filling_factors_new, mode_to_iterate, hafnian ); // TODO use task_group for this

    }


filling_factors.print_matrix();

    // calculate the partial hafnian for the given filling factors of the selected modes
    hafnian = hafnian + CalculatePartialHafnian( selected_modes, filling_factors);





}



/**
@brief ??????????????????
@return Returns with the calculated hafnian
*/
Complex16
PowerTraceHafnianRecursive::CalculatePartialHafnian( std::vector<unsigned char>& selected_modes, PicState_int64& filling_factors ) {

    size_t dim = mtx.rows;
    size_t dim_over_2 = mtx.rows/2;

    //Complex16 &summand = summands.local();
    Complex16 summand(0.0,0.0);

    // matrix B corresponds to A^(Z), i.e. to the square matrix constructed from
    // the elements of mtx=A indexed by the rows and colums, where the binary representation of
    // permutation_idx was 1
    // for details see the text below Eq.(3.20) of arXiv 1805.12498
    size_t total_num_of_modes = sum(filling_factors);
    matrix B(total_num_of_modes*2, total_num_of_modes*2);
    for (size_t idx = 0; idx < total_num_of_modes; idx++) {
        for (size_t jdx = 0; jdx < total_num_of_modes; jdx++) {
            B[2*idx*B.stride + jdx*2]   = mtx[2*selected_modes[idx]*mtx.stride + ((selected_modes[jdx]*2) ^ 1)];
            B[2*idx*B.stride + jdx*2+1] = mtx[2*selected_modes[idx]*mtx.stride + ((selected_modes[jdx]*2+1) ^ 1)];
            B[(2*idx+1)*B.stride + jdx*2]   = mtx[(2*selected_modes[idx]+1)*mtx.stride + ((selected_modes[jdx]*2) ^ 1)];
            B[(2*idx+1)*B.stride + jdx*2+1] = mtx[(2*selected_modes[idx]+1)*mtx.stride + ((selected_modes[jdx]*2+1) ^ 1)];
        }
    }


    // calculating Tr(B^j) for all j's that are 1<=j<=dim/2
    // this is needed to calculate f_G(Z) defined in Eq. (3.17b) of arXiv 1805.12498
    matrix traces(dim_over_2, 1);
    if (total_num_of_modes != 0) {
        traces = calc_power_traces(B, dim_over_2);
    }
    else{
        // in case we have no 1's in the binary representation of permutation_idx we get zeros
        // this occurs once during the calculations
        memset( traces.get_data(), 0.0, traces.rows*traces.cols*sizeof(Complex16));
    }



    // fact corresponds to the (-1)^{(n/2) - |Z|} prefactor from Eq (3.24) in arXiv 1805.12498
    bool fact = ((dim_over_2 - total_num_of_modes) % 2);


    // auxiliary data arrays to evaluate the second part of Eqs (3.24) and (3.21) in arXiv 1805.12498
    matrix aux0(dim_over_2 + 1, 1);
    matrix aux1(dim_over_2 + 1, 1);
    memset( aux0.get_data(), 0.0, (dim_over_2 + 1)*sizeof(Complex16));
    memset( aux1.get_data(), 0.0, (dim_over_2 + 1)*sizeof(Complex16));
    aux0[0] = 1.0;
    // pointers to the auxiliary data arrays
    Complex16 *p_aux0=NULL, *p_aux1=NULL;

    for (size_t idx = 1; idx <= dim_over_2; idx++) {


        Complex16 factor = traces[idx - 1] / (2.0 * idx);
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


    return summand;


}

/**
@brief ??????????????????
@return Returns with the calculated hafnian
*/
matrix
PowerTraceHafnianRecursive::getPermutedMatrix() {


    matrix res(mtx.rows, mtx.cols);


    size_t num_of_modes = modes.size();

    for (size_t row_idx=0; row_idx<num_of_modes; row_idx++ ) {

        size_t row_offset_q_orig = row_idx*mtx.stride;
        size_t row_offset_p_orig = (row_idx+num_of_modes)*mtx.stride;

        size_t row_offset_q_permuted = 2*row_idx*res.stride;
        size_t row_offset_p_permuted = (2*row_idx+1)*res.stride;

        for (size_t col_idx=0; col_idx<num_of_modes; col_idx++ ) {

            res[row_offset_q_permuted + col_idx*2] = mtx[row_offset_q_orig + col_idx];
            res[row_offset_q_permuted + col_idx*2 + 1] = mtx[row_offset_q_orig + num_of_modes + col_idx];

            res[row_offset_p_permuted + col_idx*2] = mtx[row_offset_p_orig + col_idx];
            res[row_offset_p_permuted + col_idx*2 + 1] = mtx[row_offset_p_orig + num_of_modes + col_idx];

        }

    }

    //res.print_matrix();

    return res;

}

/**
@brief Call to calculate the hafnian of a complex matrix
@param mtx The matrix
@return Returns with the calculated hafnian
*/
Complex16
PowerTraceHafnianRecursive::calculate_tmp() {


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

    unsigned long long lower_bound = 0;


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

        // matrix B corresponds to A^(Z), i.e. to the square matrix constructed from
        // the elements of mtx=A indexed by the rows and colums, where the binary representation of
        // permutation_idx was 1
        // for details see the text below Eq.(3.20) of arXiv 1805.12498
        matrix B(number_of_ones, number_of_ones);
        for (size_t idx = 0; idx < number_of_ones; idx++) {
            for (size_t jdx = 0; jdx < number_of_ones; jdx++) {
                B[idx*number_of_ones + jdx] = mtx[positions_of_ones[idx]*dim + ((positions_of_ones[jdx]) ^ 1)];
            }
        }

        // calculating Tr(B^j) for all j's that are 1<=j<=dim/2
        // this is needed to calculate f_G(Z) defined in Eq. (3.17b) of arXiv 1805.12498
        matrix traces(dim_over_2, 1);
        if (number_of_ones != 0) {
            traces = calc_power_traces(B, dim_over_2);
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


            Complex16 factor = traces[idx - 1] / (2.0 * idx);
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




} // PIC
