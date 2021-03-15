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
@brief Call to calculate the Binomial Coefficient C(n, k)
@param n The integer n
@param k The integer k
@return Returns with the Binomial Coefficient C(n, k).
*/
static int BinomialCoeff(int n, int k) {
   int C[k+1];
   memset(C, 0, sizeof(C));
   C[0] = 1;
   for (int i = 1; i <= n; i++) {
      for (int j = std::min(i, k); j > 0; j--)
         C[j] = C[j] + C[j-1];
   }
   return C[k];

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
PowerTraceHafnianRecursive::PowerTraceHafnianRecursive( matrix &mtx_in, PicState_int64& occupancy_in ) : PowerTraceHafnian( mtx_in ) {

    occupancy = occupancy_in;

    if (mtx.rows != 2*occupancy.size()) {
        std::cout << "The length of array occupancy should be equal to the half of the dimension of the input matrix mtx. Exiting" << std::endl;
        exit(-1);
    }

    if (mtx.rows % 2 != 0) {
        // The dimensions of the matrix should be even
        std::cout << "In PowerTraceHafnianRecursive algorithm the dimensions of the matrix should strictly be even. Exiting" << std::endl;
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
PowerTraceHafnianRecursive::~PowerTraceHafnianRecursive() {

    delete task_count_mutex;

}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
Complex16
PowerTraceHafnianRecursive::calculate() {


    if (mtx.rows == 0) {
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

    unsigned long long permutation_idx_max = power_of_2( (unsigned long long) num_of_modes);


    // create task group to spawn tasks
    tbb::task_group tg;

    // thread local storage for partial hafnian
    tbb::combinable<Complex16> priv_addend{[](){return Complex16(0,0);}};

    // for cycle over the combinations of occupancy
    for (unsigned long long permutation_idx = 1; permutation_idx < permutation_idx_max; permutation_idx++) {


            // select occupancy corresponding to the binary representation of permutation_idx
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


            // spawn iterations over the occupied numbers of the occupancy

            // initial filling of the occupancy
            bool skip_contribution = false; //if the maximal occupancy of one mode is zero, we skip this contribution
            PicState_int64 filling_factors(selected_modes.size());
            for (size_t idx=0;idx<selected_modes.size(); idx++) {
                if (occupancy[selected_modes[idx]] > 0 ) {
                     filling_factors[idx] = 1;
                }
                else {
                    skip_contribution = true;
                    break;
                }
            }

            if (skip_contribution) {

                continue;
            }

            // prevent the exponential explosion of spawned tasks (and save the stack space)
            // and spawn new task only if the current number of tasks is smaller than a cutoff
            if (task_num < max_task_num) {


                {
                    tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                    task_num++;
                    //std::cout << "task num root: " << task_num << std::endl;
                }


                tg.run( [this, selected_modes, filling_factors, &priv_addend, &tg]() {

                    IterateOverSelectedoccupancy( selected_modes, filling_factors, 0, priv_addend, tg );

                    {
                        tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                        task_num--;
                    }
                    return;

                });

            }
            else {
                // if the current number of tasks is greater than the maximal number of tasks, than the task is sequentialy
                IterateOverSelectedoccupancy( selected_modes, filling_factors, 0, priv_addend, tg );
            }


    }

    // wait until all spawned tasks are completed
    tg.wait();


    Complex16 hafnian( 0.0, 0.0 );
    priv_addend.combine_each([&](Complex16 a) {
        hafnian = hafnian + a;
    });


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
PowerTraceHafnianRecursive::IterateOverSelectedoccupancy( const std::vector<unsigned char>& selected_modes, const PicState_int64& filling_factors, size_t mode_to_iterate, tbb::combinable<Complex16>& priv_addend, tbb::task_group &tg ) {



    // spawn iteration over the next mode if available
    size_t new_mode_to_iterate = mode_to_iterate+1;
    while ( new_mode_to_iterate < selected_modes.size() ) {


        // prevent the exponential explosion of spawned tasks (and save the stack space)
        // and spawn new task only if the current number of tasks is smaller than a cutoff
        if (task_num < max_task_num) {

            {
                tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                task_num++;
                //std::cout << "task num: " << task_num << std::endl;
            }

            tg.run( [this, new_mode_to_iterate, selected_modes, filling_factors, &priv_addend, &tg ](){


                if ( filling_factors[new_mode_to_iterate] < occupancy[selected_modes[new_mode_to_iterate]]) {
                    PicState_int64 filling_factors_new = filling_factors.copy();
                    filling_factors_new[new_mode_to_iterate]++;
                    IterateOverSelectedoccupancy( selected_modes, filling_factors_new, new_mode_to_iterate, priv_addend, tg );
                }

                {
                    tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                    task_num--;
                }

                return;

            });

        }
        else {
           // if the current number of tasks is greater than the maximal number of tasks, than the task is sequentialy
           if ( filling_factors[new_mode_to_iterate] < occupancy[selected_modes[new_mode_to_iterate]]) {
                PicState_int64 filling_factors_new = filling_factors.copy();
                filling_factors_new[new_mode_to_iterate]++;
                IterateOverSelectedoccupancy( selected_modes, filling_factors_new, new_mode_to_iterate, priv_addend, tg );
            }


        }

        new_mode_to_iterate++;


    }


    // spawn task on the next filling factor value of the mode labeled by mode_to_iterate
    if ( filling_factors[mode_to_iterate] < occupancy[selected_modes[mode_to_iterate]]) {

        // prevent the exponential explosion of spawned tasks (and save the stack space)
        // and spawn new task only if the current number of tasks is smaller than a cutoff
        if (task_num < max_task_num) {
            {
                tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                task_num++;
                //std::cout << "task num: " << task_num << std::endl;
            }

            tg.run( [this, mode_to_iterate, selected_modes, filling_factors, &priv_addend, &tg ](){

                PicState_int64 filling_factors_new = filling_factors.copy();
                filling_factors_new[mode_to_iterate]++;
                IterateOverSelectedoccupancy( selected_modes, filling_factors_new, mode_to_iterate, priv_addend, tg );
                {
                    tbb::spin_mutex::scoped_lock my_lock{*task_count_mutex};
                    task_num--;
                }

                return;
            });

        }
        else {
            // if the current number of tasks is greater than the maximal number of tasks, than the task is sequentialy
            PicState_int64 filling_factors_new = filling_factors.copy();
            filling_factors_new[mode_to_iterate]++;
            IterateOverSelectedoccupancy( selected_modes, filling_factors_new, mode_to_iterate, priv_addend, tg );
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
std::cout << "filling_factors ";
for (size_t idx=0; idx<filling_factors.size(); idx++) {
std::cout << filling_factors[idx];
}
std::cout << std::endl;
*/


    // calculate the partial hafnian for the given filling factors of the selected occupancy
    Complex16 partial_hafnian = CalculatePartialHafnian( selected_modes, filling_factors);

    // add partial hafnian to the sum including the combinatorial factors
    double combinatorial_fact(1.0);
    for (size_t idx=0; idx < selected_modes.size(); idx++) {
        combinatorial_fact = combinatorial_fact * BinomialCoeff(occupancy[selected_modes[idx]], // the maximal allowed occupancy
                                                                 filling_factors[idx] // the current occupancy
                                                                 );
    }

    Complex16 &hafnian_priv = priv_addend.local();
//std::cout << "combinatorial_fact " << combinatorial_fact << std::endl;
//std::cout << "partial_hafnian " << partial_hafnian << std::endl;
    hafnian_priv = hafnian_priv + partial_hafnian * combinatorial_fact;





}



/**
@brief ??????????????????
@return Returns with the calculated hafnian
*/
Complex16
PowerTraceHafnianRecursive::CalculatePartialHafnian( const std::vector<unsigned char>& selected_modes, const PicState_int64& filling_factors ) {


    //size_t dim_over_2 = mtx.rows/2;

    //Complex16 &summand = summands.local();
    Complex16 summand(0.0,0.0);

    size_t num_of_modes = sum(filling_factors);
    size_t total_num_of_modes = sum(occupancy);
    size_t dim = total_num_of_modes*2;



    // matrix B corresponds to A^(Z), i.e. to the square matrix constructed from
    matrix&& B = CreateAZ(selected_modes, filling_factors, num_of_modes);

    // calculating Tr(B^j) for all j's that are 1<=j<=dim/2
    // this is needed to calculate f_G(Z) defined in Eq. (3.17b) of arXiv 1805.12498
    matrix traces(total_num_of_modes, 1);
    if (num_of_modes != 0) {
        traces = calc_power_traces(B, total_num_of_modes);
    }
    else{
        // in case we have no 1's in the binary representation of permutation_idx we get zeros
        // this occurs once during the calculations
        memset( traces.get_data(), 0.0, traces.rows*traces.cols*sizeof(Complex16));
    }


    // fact corresponds to the (-1)^{(n/2) - |Z|} prefactor from Eq (3.24) in arXiv 1805.12498
    bool fact = ((total_num_of_modes - num_of_modes) % 2);


    // auxiliary data arrays to evaluate the second part of Eqs (3.24) and (3.21) in arXiv 1805.12498
    matrix aux0(total_num_of_modes + 1, 1);
    matrix aux1(total_num_of_modes + 1, 1);
    memset( aux0.get_data(), 0.0, (total_num_of_modes + 1)*sizeof(Complex16));
    memset( aux1.get_data(), 0.0, (total_num_of_modes + 1)*sizeof(Complex16));
    aux0[0] = 1.0;
    // pointers to the auxiliary data arrays
    Complex16 *p_aux0=NULL, *p_aux1=NULL;

    for (size_t idx = 1; idx <= total_num_of_modes; idx++) {


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

        memcpy(p_aux1, p_aux0, (total_num_of_modes+1)*sizeof(Complex16) );

        for (size_t jdx = 1; jdx <= (dim / (2 * idx)); jdx++) {
            powfactor = powfactor * factor / ((double)jdx);


            for (size_t kdx = idx * jdx + 1; kdx <= total_num_of_modes + 1; kdx++) {
                p_aux1[kdx-1] += p_aux0[kdx-idx*jdx - 1]*powfactor;
            }



        }


    }


    if (fact) {
        summand = summand - p_aux1[total_num_of_modes];
//std::cout << -p_aux1[total_num_of_modes] << std::endl;
    }
    else {
        summand = summand + p_aux1[total_num_of_modes];
//std::cout << p_aux1[total_num_of_modes] << std::endl;
    }


    return summand;


}



/**
@brief ??????????????????
@return Returns with the calculated hafnian
*/
matrix
PowerTraceHafnianRecursive::CreateAZ( const std::vector<unsigned char>& selected_modes, const PicState_int64& filling_factors, const size_t& num_of_modes ) {


    // matrix B corresponds to A^(Z), i.e. to the square matrix constructed from
    // the elements of mtx=A indexed by the rows and colums, where the binary representation of
    // permutation_idx was 1
    // for details see the text below Eq.(3.20) of arXiv 1805.12498
    matrix B(num_of_modes*2, num_of_modes*2);

    size_t row_idx = 0;
    for (size_t mode_idx = 0; mode_idx < selected_modes.size(); mode_idx++) {

        size_t row_offset_mtx_a = 2*selected_modes[mode_idx]*mtx.stride;
        size_t row_offset_mtx_aconj = (2*selected_modes[mode_idx]+1)*mtx.stride;

        for (size_t filling_factor_row=1; filling_factor_row<=filling_factors[mode_idx]; filling_factor_row++) {

            size_t row_offset_B_a = 2*row_idx*B.stride;
            size_t row_offset_B_aconj = (2*row_idx+1)*B.stride;


            size_t col_idx = 0;

            for (size_t mode_jdx = 0; mode_jdx < selected_modes.size(); mode_jdx++) {


                for (size_t filling_factor_col=1; filling_factor_col<=filling_factors[mode_jdx]; filling_factor_col++) {

                    B[row_offset_B_a + col_idx*2]   = mtx[row_offset_mtx_a + ((selected_modes[mode_jdx]*2) ^ 1)];
                    B[row_offset_B_a + col_idx*2+1] = mtx[row_offset_mtx_a + ((selected_modes[mode_jdx]*2+1) ^ 1)];
                    B[row_offset_B_aconj + col_idx*2]   = mtx[row_offset_mtx_aconj + ((selected_modes[mode_jdx]*2) ^ 1)];
                    B[row_offset_B_aconj + col_idx*2+1] = mtx[row_offset_mtx_aconj + ((selected_modes[mode_jdx]*2+1) ^ 1)];

                    col_idx++;
                }
            }


            row_idx++;
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

    return B;


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
