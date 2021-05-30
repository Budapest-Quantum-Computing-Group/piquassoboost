
#include "TorontonianUtilities.hpp"
#include "PicState.h"
#include "PicVector.hpp"


#ifndef CPYTHON
#include "tbb/tbb.h"
#endif

#ifdef __MPI__
#include <mpi.h>

#define ACTIVITY_TAG 0
#define FINISHED_TAG 1
#define WORK_TAG 100
#define SELECTED_INDEX_HOLES_SIZE_TAG 2
#define SELECTED_INDEX_HOLES_TAG 3
#define HOLE_TO_ITERATE_TAG 4
#define L_SIZE_TAG 5
#define L_TAG 6
#define REUSE_INDEX_TAG 7

#endif // MPI

namespace pic {



// relieve Python extension from TBB functionalities
#ifndef CPYTHON

#ifdef __MPI__

static tbb::spin_mutex my_mutex;

/**
@brief Class to monitor the activity of MPI processes
*/
class MPIActivityComunicator {

protected:
    /// The number of MPI processes
    int world_size;
    /// The rank of the current MPI process
    int current_rank;
    /// The rank of the parent MPI process (from which work can be received)
    int parent_process;
    /// The rank of the child MPI process (for which work can be transmitted)
    int child_process;
    /// 1 if the child process is occupied, 0 if the child process is idle
    int child_status;
    /// 1 if the current process is occupied, 0 if the current process is idle
    int current_activity;

public:

/**
@brief Nullary constructor of the class.
@return Returns with the instance of the class.
*/
MPIActivityComunicator() {

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    parent_process     = current_rank -1 ;
    child_process      = current_rank +1 ;
    child_status       = -1;
    current_activity   = 0;

}



/**
@brief Set the current process to active and sen a message about this to the parent MPI process
*/
void ActivateCurrentProcess() {

    current_activity = 1;
    SendActivityStatus();
    return;

}


/**
@brief Set the current process to idle and sen a message about this to the parent MPI process
*/
void DisableCurrentProcess() {

    current_activity = 0;
    SendActivityStatus();
    return;

}


/**
@brief Call to send information about the activity status of the current MPI process to the parent MPI process.
*/
void SendActivityStatus() {

    if (parent_process >= 0) {
        MPI_Send(&current_activity, 1, MPI_INT, parent_process, ACTIVITY_TAG, MPI_COMM_WORLD);
    }

    return;

}



/**
@brief Call to receive information about the activity status of the child MPI process.
*/
void ListenToActivityStatus( ) {

    if (child_process < world_size) {
        MPI_Recv( &child_status, 1, MPI_INT, child_process, ACTIVITY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //std::cout << current_rank << ": activity of child process: " << child_process << " is " << child_status << std::endl;
    }

}


/**
@brief Call to check whether the child MPI process has sent a message about it's activity status, or not. If message has been sent it is received.
*/
void CheckForActivityStatusMessage( ) {

    if (child_process < world_size) {
        int flag = 0;
        MPI_Iprobe(child_process, ACTIVITY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE );
//std::cout << "FLAG " << flag << std::endl;

        if (flag==1) {
            ListenToActivityStatus( );
        }
    }

}





/**
@brief Call to check the cached activity status of the child process.
@return Return with 1 if the child process is active, and with 0 otherwise.
*/
int CheckChildProcessActivity() {

    return child_status;

}

/**
@brief Call to retrieve the rank of the parent MPI process
@return Returns with the rank of the parent MPI process
*/
int getParentProcess() {

    return parent_process;

}



/**
@brief Call to retrieve the rank of the child MPI process
@return Returns with the rank of the child MPI process
*/
int getChildProcess() {

    return child_process;

}

};

#endif // MPI

/**
@brief Class to calculate the hafnian of a complex matrix by a recursive power trace method. This algorithm is designed to support gaussian boson sampling simulations, it is not a general
purpose hafnian calculator. This algorithm accounts for the repeated occupancy in the covariance matrix.
*/
template<class matrix_type, class complex_type>
class TorontonianRecursive_Tasks  {


protected:
    /// The input matrix. Must be selfadjoint positive definite matrix with eigenvalues between 0 and 1.
    matrix mtx_orig;
    /** The scaled input matrix for which the calculations are performed.
    If the mean magnitude of the matrix elements is one, the treshold of quad precision can be set to higher values.
    */
    matrix_type mtx;
    /// number of modes spanning the gaussian state
    int num_of_modes;
    /// logical variable to indicate whether a work is transmitting to a child process at the moment
    bool sending_work;

    tbb::combinable<RealM<long double>> priv_addend;

#ifdef __MPI__
    /// The number of MPI processes
    int world_size;
    /// The rank of the current MPI process
    int current_rank;
    /// MPI activity listener
    MPIActivityComunicator listener;
#endif // MPI

public:

/**
@brief Nullary constructor of the class.
@return Returns with the instance of the class.
*/
TorontonianRecursive_Tasks() {


#ifdef __MPI__
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    // initialize MPI activity listener
    listener = MPIActivityComunicator();
#endif // MPI

}

/**
@brief Constructor of the class.
@param mtx_in A selfadjoint matrix for which the torontonian is calculated. This matrix has to be positive definite matrix with eigenvalues between 0 and 1 (for example a covariance matrix of the Gaussian state.) matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state)
@return Returns with the instance of the class.
*/
TorontonianRecursive_Tasks( matrix &mtx_in ) {

    Update_mtx( mtx_in );

    // number of modes spanning the gaussian state
    num_of_modes = mtx.rows/2;


    if (mtx.rows % 2 != 0) {
        // The dimensions of the matrix should be even
        std::cout << "In PowerTraceHafnianRecursive_Tasks algorithm the dimensions of the matrix should strictly be even. Exiting" << std::endl;
        exit(-1);
    }

#ifdef __MPI__
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    sending_work = false;
#endif // MPI

}


/**
@brief Default destructor of the class.
*/
virtual ~TorontonianRecursive_Tasks() {

}

/**
@brief Call to calculate the hafnian of a complex matrix
@return Returns with the calculated hafnian
*/
double calculate() {


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


    // thread local storage for partial hafnian
    priv_addend = tbb::combinable<RealM<long double>>{[](){return RealM<long double>(0.0);}};

    // construct the initial selection of the modes
    PicVector<int> selected_index_holes;

    // calculate the Cholesky decomposition of the initial matrix to be later reused
    matrix_type L = mtx.copy();
    Complex32 determinant = calc_determinant_cholesky_decomposition<matrix_type, complex_type>(L);

#ifdef __MPI__
    long double torontonian;
    if (current_rank == 0) {
        torontonian = CalculatePartialTorontonian( selected_index_holes, determinant);
    }
    else{
        torontonian = 0.0;
    }
#else
    long double torontonian = CalculatePartialTorontonian( selected_index_holes, determinant);
#endif



    // add the first index hole in prior to the iterations
    selected_index_holes.push_back(num_of_modes-1);

    // start task iterations originating from the initial selected modes
#ifdef __MPI__
    StartMPIIterations( L );
    //IterateOverSelectedModes( selected_index_holes, 0, L, num_of_modes-1, 0 );
#else
    IterateOverSelectedModes( selected_index_holes, 0, L, num_of_modes-1 );
#endif



    priv_addend.combine_each([&](RealM<long double> &a) {
        torontonian = torontonian + a.get();
    });

#ifdef __MPI__
    // send terminating signal to the child process (until now child might wait for new work assigned from the current process)
    SendTerminatingSignalToChildProcess();
#endif


#if BLAS==0 // undefined BLAS
    omp_set_num_threads(NumThreads);
#elif BLAS==1 //MKL
    MKL_Set_Num_Threads(NumThreads);
#elif  BLAS==2 //OpenBLAS
    openblas_set_num_threads(NumThreads);
#endif


#ifdef __MPI__
    if (current_rank == 0) {
        // last correction comming from an empty submatrix contribution
        double factor =  (num_of_modes) % 2  ? -1.0 : 1.0;
        torontonian = torontonian + factor;
    }


    // send the calculated partial hafnian to rank 0
    long double* partial_torontonians = new long double[world_size];
//std::cout << current_rank << ": gathering results" << std::endl;
    MPI_Allgather(&torontonian, 1, MPI_LONG_DOUBLE, partial_torontonians, 1, MPI_LONG_DOUBLE, MPI_COMM_WORLD);

    torontonian = 0.0;
    for (int idx=0; idx<world_size; idx++) {
        torontonian = torontonian + partial_torontonians[idx];
    }

    // release memory on the zero rank
    delete partial_torontonians;

#else

    // last correction comming from an empty submatrix contribution
    double factor =  (num_of_modes) % 2  ? -1.0 : 1.0;
    torontonian = torontonian + factor;
#endif


    return (double)torontonian;
}



/**
@brief Call to update the memory address of the matrix mtx and reorder the matrix elements into a_1^*,a_1^*,a_2,a_2^*, ... a_N,a_N^* order.
@param mtx_in Input matrix defined by
*/
void Update_mtx( matrix &mtx_in) {

#ifdef __MPI__
    // ensure that each MPI process gets the same input matrix from rank 0
    void* syncronized_data = (void*)mtx_in.get_data();
    MPI_Bcast(syncronized_data, mtx_in.size()*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    size_t dim = mtx_in.rows;

    mtx = matrix_type(dim, dim);
    for (size_t idx = 0; idx < mtx_in.size(); idx++) {
        mtx[idx] = mtx_in[idx];
    }


    // convert the input matrix from a1, a2, ... a_N, a_1^*, a_2^* ... a_N^* format to
    // a_1^*,a_1^*,a_2,a_2^*, ... a_N,a_N^* format

    size_t num_of_modes = dim/2;
    matrix_type mtx_reordered(dim, dim);
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
    //ScaleMatrix();
}



protected:

#ifdef __MPI__

/**
@brief Call to run iterations over the selected modes to calculate partial torontonians
@param L Matrix conatining partial Cholesky decomposition if the initial matrix to be reused
@param priv_addend Therad local storage for the partial torontonians
@param tg Reference to a tbb::task_group
*/

void StartMPIIterations( matrix_type &L ) {

    // start activity on the current MPI process
    listener.ActivateCurrentProcess();
    listener.ListenToActivityStatus();


    int index_min = 0;
    int index_max = num_of_modes;

    // ***** iterations over the selected index hole to calculate partial torontonians *****

    // first do the first iteration without spawning iterations with new index hole
    // construct the initial selection of the modes
    PicVector<int> selected_index_holes;
    selected_index_holes.push_back(index_max-1);



    // now do the rest of the iterations
    for( int idx = index_min+current_rank;  idx < index_max; idx=idx+world_size){

        PicVector<int> selected_index_holes_new = selected_index_holes;
        selected_index_holes_new[0] = idx;

        int reuse_index_new = idx;

        matrix_type &&L_new = CreateAZ(selected_index_holes_new, L, reuse_index_new);
        Complex32 determinant = calc_determinant_cholesky_decomposition<matrix_type, complex_type>(L_new, 2*reuse_index_new);

        long double partial_torontonian = CalculatePartialTorontonian( selected_index_holes_new, determinant );
        RealM<long double> &torontonian_priv = priv_addend.local();
        torontonian_priv += partial_torontonian;

        selected_index_holes_new.push_back(this->num_of_modes-1);
        reuse_index_new = L_new.rows/2-1;

        if (idx==index_max-1) continue;

        IterateOverSelectedModes( selected_index_holes_new, 1, L_new, reuse_index_new );

    }


    // indicate that current process has finished an activity
    listener.DisableCurrentProcess();

//std::cout << current_rank << ": PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP" << std::endl;
    ListenToNewWork();

}
//#endif


/**
@brief Call to run iterations over the selected modes to calculate partial torontonians
@param L Matrix conatining partial Cholesky decomposition if the initial matrix to be reused
@param priv_addend Therad local storage for the partial torontonians
@param tg Reference to a tbb::task_group
*/
void StartProcessActivity( const PicVector<int>& selected_index_holes, int hole_to_iterate, matrix_type &L, const int reuse_index ) {

    // start activity on the current MPI process
    listener.ActivateCurrentProcess();

//std::cout << "starting received work on :" << current_rank << std::endl;
    IterateOverSelectedModes( selected_index_holes, hole_to_iterate, L, reuse_index );
//std::cout << "ending received work on :" << current_rank << std::endl;

    // indicate that current process has finished an activity
    listener.DisableCurrentProcess();

//std::cout << "?????????????????????????????????????? " << current_rank << std::endl;
    ListenToNewWork();


}

/**
@brief
*/
void ListenToNewWork() {

    int parent_process = listener.getParentProcess();
    if (parent_process < 0) return;

//std::cout << current_rank << ": listening to work" << std::endl;

    int reuse_index;
    MPI_Recv( &reuse_index, 1, MPI_INT, parent_process, REUSE_INDEX_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // return if terminating signal was sent from the parent
    if (reuse_index == -1) return;

    int selected_index_holes_size;
    MPI_Recv( &selected_index_holes_size, 1, MPI_INT, parent_process, SELECTED_INDEX_HOLES_SIZE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    PicVector<int> selected_index_holes(selected_index_holes_size);
    if (selected_index_holes_size>0) {
        int* selected_index_holes_data = selected_index_holes.data();
        MPI_Recv( selected_index_holes_data, selected_index_holes_size, MPI_INT, parent_process, SELECTED_INDEX_HOLES_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int hole_to_iterate;
    MPI_Recv( &hole_to_iterate, 1, MPI_INT, parent_process, HOLE_TO_ITERATE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int L_size[3];
    MPI_Recv( &L_size, 3, MPI_INT, parent_process, L_SIZE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    matrix_type L(L_size[0], L_size[1], L_size[2]);

    if (L.size() >0 ) {
        complex_type* L_data = L.get_data();
        if (sizeof(complex_type) == sizeof(Complex16)) {
            MPI_Recv(L_data, 2*L.size(), MPI_DOUBLE, parent_process, L_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else {
            MPI_Recv(L_data, 2*L.size(), MPI_LONG_DOUBLE, parent_process, L_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // start working on the received task
    StartProcessActivity(selected_index_holes, hole_to_iterate, L, reuse_index );

}

/**
@brief Call to run iterations over the selected modes to calculate partial torontonians
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A^Z.
@param L Matrix conatining partial Cholesky decomposition if the initial matrix to be reused
@param hole_to_iterate The index indicating which hole index should be iterated
*/
void SendWorkToChildProcess( const PicVector<int> selected_index_holes, int hole_to_iterate, matrix_type L, const int reuse_index ) {

    int child_process = listener.getChildProcess();
    if (child_process >= world_size) return;
/*
if (L.size() > 0) {
    std::cout << "process " << current_rank << ": sending work to child process " << L.size() << "Lsize" << std::endl;
}
else {
    std::cout << "process " << current_rank << ": sending finalizing signal" << std::endl;
}
*/

    int selected_index_holes_size = selected_index_holes.size();
    MPI_Send(&selected_index_holes_size, 1, MPI_INT, child_process, SELECTED_INDEX_HOLES_SIZE_TAG, MPI_COMM_WORLD);

    if (selected_index_holes_size>0) {
        int* selected_index_holes_data = selected_index_holes.data();
        MPI_Send(selected_index_holes_data, selected_index_holes_size, MPI_INT, child_process, SELECTED_INDEX_HOLES_TAG, MPI_COMM_WORLD);
    }


    MPI_Send(&hole_to_iterate, 1, MPI_INT, child_process, HOLE_TO_ITERATE_TAG, MPI_COMM_WORLD);

    int L_size[3];
    L_size[0] = L.rows;
    L_size[1] = L.cols;
    L_size[2] = L.stride;
    MPI_Send(&L_size, 3, MPI_INT, child_process, L_SIZE_TAG, MPI_COMM_WORLD);

    if (L.size() >0 ) {
        complex_type* L_data = L.get_data();
        if (sizeof(complex_type) == sizeof(Complex16)) {
            MPI_Send(L_data, 2*L.size(), MPI_DOUBLE, child_process, L_TAG, MPI_COMM_WORLD);
        }
        else {
            MPI_Send(L_data, 2*L.size(), MPI_LONG_DOUBLE, child_process, L_TAG, MPI_COMM_WORLD);
        }
    }

    MPI_Send(&reuse_index, 1, MPI_INT, child_process, REUSE_INDEX_TAG, MPI_COMM_WORLD);

    // listen that child process activates himself
    listener.ListenToActivityStatus();


}



/**
@brief Call to send terminating signal to the child process.
*/
void SendTerminatingSignalToChildProcess() {

    int child_process = listener.getChildProcess();
    if (child_process >= world_size) return;

    // the terminating signal is indicated by reuse_index = -1
    int reuse_index = -1;

    MPI_Send(&reuse_index, 1, MPI_INT, child_process, REUSE_INDEX_TAG, MPI_COMM_WORLD);

    return;


}
#endif


/**
@brief Call to run iterations over the selected modes to calculate partial torontonians
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A^Z.
@param L Matrix conatining partial Cholesky decomposition if the initial matrix to be reused
@param hole_to_iterate The index indicating which hole index should be iterated
@param priv_addend Therad local storage for the partial torontonians
@param tg Reference to a tbb::task_group
*/
void IterateOverSelectedModes( const PicVector<int>& selected_index_holes, int hole_to_iterate, matrix_type &L, const int reuse_index ) {


    // calculate the partial Torontonian for the selected index holes
    int index_min;
    int index_max;
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
    PicVector<int> selected_index_holes_new = selected_index_holes;
    selected_index_holes_new[hole_to_iterate] = index_max-1;

    int reuse_index_new = index_max-1-hole_to_iterate < reuse_index ? index_max-1-hole_to_iterate : reuse_index;
    matrix_type &&L_new = CreateAZ(selected_index_holes_new, L, reuse_index_new);
    Complex32 determinant = calc_determinant_cholesky_decomposition<matrix_type, complex_type>(L_new, 2*reuse_index_new);

    long double partial_torontonian = CalculatePartialTorontonian( selected_index_holes_new, determinant );
    RealM<long double> &torontonian_priv = priv_addend.local();
    torontonian_priv += partial_torontonian;


    // logical variable to control whether spawning new iterations or not
    bool stop_spawning_iterations = (selected_index_holes.size() == num_of_modes-1);
    // add new index hole to the iterations
    int new_hole_to_iterate = hole_to_iterate+1;

    // now do the rest of the iterations
    tbb::parallel_for( index_min+1,  index_max, (int)1, [&](int idx){
    //for( int idx = index_min+1;  idx < index_max; idx++){

        PicVector<int> selected_index_holes_new = selected_index_holes;
        selected_index_holes_new[hole_to_iterate] = idx-1;

        int reuse_index_new = idx-1-hole_to_iterate < reuse_index ? idx-1-hole_to_iterate : reuse_index;

        matrix_type &&L_new = CreateAZ(selected_index_holes_new, L, reuse_index_new);
        Complex32 determinant = calc_determinant_cholesky_decomposition<matrix_type, complex_type>(L_new, 2*reuse_index_new);

        long double partial_torontonian = CalculatePartialTorontonian( selected_index_holes_new, determinant );
        RealM<long double> &torontonian_priv = priv_addend.local();
        torontonian_priv += partial_torontonian;


        // return if new index hole would give no nontrivial result
        // (in this case the partial torontonian is unity and should be counted only once in function calculate)
        if (stop_spawning_iterations) return;

        selected_index_holes_new.push_back(this->num_of_modes-1);
        reuse_index_new = L_new.rows/2-1;
#ifdef __MPI__
        // check for incoming activity message from the child process
        {
            tbb::spin_mutex::scoped_lock my_lock{my_mutex};
            listener.CheckForActivityStatusMessage( );
        }

        if (!listener.CheckChildProcessActivity()) {

            {
                tbb::spin_mutex::scoped_lock my_lock{my_mutex};

                if (!sending_work) {
                    sending_work = true;
                    SendWorkToChildProcess(selected_index_holes_new, new_hole_to_iterate, L_new, reuse_index_new);
                    sending_work = false;
                    //continue;
                    return;
                }
            }

        }
#endif

        IterateOverSelectedModes( selected_index_holes_new, new_hole_to_iterate, L_new, reuse_index_new );

    });
    //}





}


/**
@brief Call to calculate the partial torontonian for given selected modes and their occupancies
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A_Z.
@param determinant The determinant of the submatrix A_Z
@return Returns with the calculated torontonian
*/
virtual long double CalculatePartialTorontonian( const PicVector<int>& selected_index_holes, const Complex32 &determinant  ) {


    size_t number_selected_modes = num_of_modes - selected_index_holes.size();


    // calculating -1^(N-|Z|)
    long double factor =
                (number_selected_modes + num_of_modes) % 2
                    ? -1.0
                    : 1.0;
/*
                    {
      tbb::spin_mutex::scoped_lock my_lock{my_mutex};
if (number_selected_modes == 11) {
std::cout << factor*determinant.real()  << std::endl;
}
                    }
*/
    // calculating -1^(number of ones) / sqrt(det(1-A^(Z)))
    long double sqrt_determinant = std::sqrt(determinant.real());

    return (factor / sqrt_determinant);


}

/**
@brief Call to construct matrix \f$ A^Z \f$ (see the text below Eq. (2) of arXiv 2009.01177) for the given modes
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A^Z.
@param L Matrix conatining partial Cholesky decomposition if the initial matrix to be reused
@param reuse_index Index labeling the highest mode for which previous Cholesky decomposition can be reused.
@return Returns with the constructed matrix \f$ A^Z \f$.
*/
matrix_type
CreateAZ( const PicVector<int>& selected_index_holes, matrix_type &L, const int reuse_index ) {

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
    int dimension_of_AZ = 2 * number_selected_modes;
    matrix_type AZ(dimension_of_AZ, dimension_of_AZ);
    memset(AZ.get_data(), 0, AZ.size()*sizeof(complex_type));
/*
    // The first 2*(reuse_index-1) rows of the matrix are not touched during the calculations they can be reused from Cholesky matrix L
    for (int idx = 0; idx < reuse_index; idx++) {

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
    for (size_t idx = (size_t)reuse_index; idx < number_selected_modes; idx++) {

        complex_type* mtx_data = mtx.get_data() + 2*(positions_of_ones[idx]*mtx.stride);
        complex_type* L_data = L.get_data() + 2*(idx+1)*L.stride;
        complex_type* AZ_data   = AZ.get_data() + 2*(idx*AZ.stride);


        memcpy(AZ_data, L_data, 2*(idx+1)*sizeof(complex_type));
        memcpy(AZ_data + AZ.stride, L_data + L.stride, 2*(idx+1)*sizeof(complex_type));

        for (int jdx = reuse_index; jdx <= idx; jdx++) {
            memcpy( AZ_data + 2*jdx, mtx_data + 2*positions_of_ones[jdx], 2*sizeof(complex_type) );
        }

        AZ_data   = AZ_data + AZ.stride;
        mtx_data = mtx_data + mtx.stride;

        for (int jdx = reuse_index; jdx <= idx; jdx++) {
            memcpy( AZ_data + 2*jdx, mtx_data + 2*positions_of_ones[jdx], 2*sizeof(complex_type) );
        }

    }

    return AZ;

}





}; //TorontonianRecursive_Tasks


#endif // CPYTHON





} // PIC

