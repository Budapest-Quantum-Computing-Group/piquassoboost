
#ifndef CPYTHON
#include "tbb/tbb.h"
#endif

#include <thread>
#include "TorontonianUtilities.hpp"
#include "MPIActivityCommunicator.hpp"
#include "PicState.h"
#include "PicVector.hpp"
#include "common_functionalities.h"


namespace pic {



// relieve Python extension from TBB functionalities
#ifndef CPYTHON


/**
@brief Class to calculate the hafnian of a complex matrix by a recursive power trace method. This algorithm is designed to support gaussian boson sampling simulations, it is not a general
purpose hafnian calculator. This algorithm accounts for the repeated occupancy in the covariance matrix.
*/
template<class input_matrix_type, class matrix_type, class scalar_type, class scalar_type_long>
class TorontonianRecursive_Tasks  {


protected:
    /// The input matrix. Must be selfadjoint positive definite matrix with eigenvalues between 0 and 1.
    input_matrix_type mtx_orig;
    /** The scaled input matrix for which the calculations are performed.
    If the mean magnitude of the matrix elements is one, the treshold of quad precision can be set to higher values.
    */
    matrix_type mtx;
    /// number of modes spanning the gaussian state
    size_t num_of_modes;
    /// logical variable to indicate whether a work is transmitting to a child process at the moment
    bool sending_work;

    tbb::combinable<RealM<long double>> priv_addend;

#ifdef __MPI__
    /// The number of modes when MPI work recived
    size_t num_of_modes_MPI;
    /// The number of MPI processes
    int world_size;
    /// The rank of the current MPI process
    int current_rank;
    /// MPI activity listener
    MPIActivityComunicator listener;
    /// initial amount of work
    unsigned long long initial_work;
    /// received amount of work
    unsigned long long received_work;
    /// sent amount of work
    unsigned long long sent_work;
    /// amount of addends that is expected to be done by the worker
    unsigned long long expected_work;
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

    // initial amount of work
    initial_work = 0;
    // received amount of work
    received_work = 0;
    // sent amount of work
    sent_work = 0;
    // amount of addends that is expected to be done by the worker
    expected_work = power_of_2( (unsigned long long) num_of_modes )/world_size;

#endif // MPI

}

/**
@brief Constructor of the class.
@param mtx_in A selfadjoint matrix for which the torontonian is calculated. This matrix has to be positive definite matrix with eigenvalues between 0 and 1 (for example a covariance matrix of the Gaussian state.) matrix. ( In GBS calculations the \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ... a_n^* \f$ ordered covariance matrix of the Gaussian state)
@return Returns with the instance of the class.
*/
TorontonianRecursive_Tasks( input_matrix_type &mtx_in ) {

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

    // MPI work is currently not sending
    sending_work = false;

    // initial amount of work
    initial_work = 0;
    // received amount of work
    received_work = 0;
    // sent amount of work
    sent_work = 0;
    // amount of addends that is expected to be done by the worker
    expected_work = power_of_2( (unsigned long long) num_of_modes )/world_size;

#endif // MPI

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
    else if (mtx.rows == 2) {
        scalar_type determinant = mtx[0]*mtx[3] - mtx[1]*mtx[2];
        long double partial_torontonian = 1.0/std::sqrt(convertToDouble(determinant));
        return (double)partial_torontonian - 1.0;
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
    scalar_type_long determinant;
    calc_determinant_cholesky_decomposition<matrix_type, scalar_type, scalar_type_long>(L, determinant);

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
    StartMPIIterations<scalar_type_long>( L );
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
void Update_mtx( input_matrix_type &mtx_in) {

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
}



protected:

#ifdef __MPI__

/**
@brief Call to start MPI distributed iterations
@param L Matrix containing partial Cholesky decomposition if the initial matrix to be reused
*/
template<class scalar_type_long>
void StartMPIIterations( matrix_type &L ) {

    // start activity on the current MPI process

    int index_min = 0;
    int index_max = num_of_modes;

    // ***** iterations over the selected index hole to calculate partial torontonians *****

    // first do the first iteration without spawning iterations with new index hole
    // construct the initial selection of the modes
    PicVector<int> selected_index_holes;
    selected_index_holes.push_back(index_max-1);


    int current_rank_index_max = index_min+current_rank+1;
    for( int idx = index_min+current_rank;  idx < index_max; idx=idx+world_size){
        current_rank_index_max += world_size;
    }
    current_rank_index_max -= world_size;

    // estimate initial amount of work
    for( int idx = current_rank_index_max-1;  idx >= 0; idx=idx-world_size){
        initial_work += power_of_2( (unsigned long long) (num_of_modes - idx - 1) );
    }



    // now do the rest of the iterations
    for( int idx = current_rank_index_max-1;  idx >= 0; idx=idx-world_size){
    //for( int idx = index_min+current_rank;  idx < index_max; idx=idx+world_size){

        PicVector<int> selected_index_holes_new = selected_index_holes;
        selected_index_holes_new[0] = idx;

        int reuse_index_new = idx;

        matrix_type &&L_new = CreateAZ(selected_index_holes_new, L, reuse_index_new);
        scalar_type_long determinant;
        calc_determinant_cholesky_decomposition<matrix_type, scalar_type, scalar_type_long>(L_new, 2*reuse_index_new, determinant);

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




    ListenToNewWork();

}


/**
@brief Call to start received work on the current MPI process. After work is completed, the current process starts to listen for new assigned work.
@param selected_index_holes Selected modes which should be omitted from the input matrix to construct A^Z.
@param L Matrix containing partial Cholesky decomposition if the initial matrix to be reused
@param hole_to_iterate The index indicating which hole index should be iterated
@param reuse index The index  labeling the number of rows of the matrix L that can be reused in the Cholesky decomposition
*/
void StartProcessActivity( PicVector<int>& selected_index_holes, int hole_to_iterate, matrix_type &L, const int reuse_index ) {


    IterateOverSelectedModes( selected_index_holes, hole_to_iterate, L, reuse_index );

    // indicate that current process has finished an activity
    listener.DisableCurrentProcess();

    ListenToNewWork();


}

/**
@brief Call to listen for work when the current MPI process is idle.
*/
void ListenToNewWork() {

    std::vector<int> &&parent_processes = listener.getParentProcesses();

    if (parent_processes.size() == 0) return;


    // listening for incomming message
    int parent_idx = 0;
    int flag = 0;
    while (true) {
        MPI_Iprobe(parent_processes[parent_idx], COMPRESSED_MESSAGE_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE );

        if (flag==1) break;

        parent_idx = (parent_idx+1) % parent_processes.size();
    }

    // receive incomming message
    int parent_process = parent_processes[parent_idx];

    unsigned long long compressed_message;
    MPI_Recv( &compressed_message, 1, MPI_UNSIGNED_LONG_LONG, parent_process, COMPRESSED_MESSAGE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    // return if terminating signal was sent from the parent
    if (compressed_message == 0) {

        // set the parent flag to finished
        listener.setParentFinished( parent_idx );

        // stop listening for work if parents are terminated
        if ( listener.checkParentFinished() ) {

            return;
        }

        ListenToNewWork();
        return;
    }


    MPI_Status status;
    // Probe for an incoming message from process zero
    MPI_Probe(parent_process, SELECTED_INDEX_HOLES_TAG, MPI_COMM_WORLD, &status);
    int selected_index_holes_size;
    MPI_Get_count(&status, MPI_INT, &selected_index_holes_size);

    PicVector<int> selected_index_holes(selected_index_holes_size);
    if (selected_index_holes_size>0) {
        int* selected_index_holes_data = selected_index_holes.data();
        MPI_Recv( selected_index_holes_data, selected_index_holes_size, MPI_INT, parent_process, SELECTED_INDEX_HOLES_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    // Indicate to other processes the change in the activity status
    listener.ActivateCurrentProcess();


    int hole_to_iterate = selected_index_holes.size() - 1;


    PicVector<int> selected_index_holes_old = selected_index_holes;
    selected_index_holes_old.pop_back();
    unsigned long L_size = 2*(num_of_modes-selected_index_holes.size());
    matrix_type tmp(L_size, L_size);
    matrix_type &&L = CreateAZ(selected_index_holes_old, tmp, 0);
    scalar_type_long determinant;
    calc_determinant_cholesky_decomposition<matrix_type, scalar_type, scalar_type_long>(L, 0, determinant);

    int reuse_index = L.rows-1;

    received_work += power_of_2( (unsigned long long) (num_of_modes - selected_index_holes[hole_to_iterate-1]) - 1 );

    // start working on the received task
    StartProcessActivity(selected_index_holes, hole_to_iterate, L, reuse_index );


}

/**
@brief Call to send work to child MPI process
@param selected_index_holes Selected modes which should be omitted from the input matrix to construct A^Z.
@param L Matrix containing partial Cholesky decomposition if the initial matrix to be reused
@param hole_to_iterate The index indicating which hole index should be iterated
@param reuse index The index  labeling the number of rows of the matrix L that can be reused in the Cholesky decomposition
*/
void SendWorkToChildProcess( const PicVector<int> selected_index_holes, int hole_to_iterate, matrix_type L, const int reuse_index ) {

    // get the index of the first idle child process
    int child_index = listener.getIdleProcessIndex();

    // get the rank of the child process
    int child_process = listener.getChildRank( child_index );

    if (child_process < 0) return;

    // indicate that child becomes active by receiving work
    listener.setChildActive(child_index);

    // compress data in selected_index_holes
    unsigned long long compressed_message = 0;
    for (size_t idx=0; idx<selected_index_holes.size(); idx++ ) {
        //compressed_message += power_of_2( (unsigned long long) (selected_index_holes[idx]));
        compressed_message += power_of_2( (unsigned long long) (num_of_modes - selected_index_holes[idx] - 1));
    }
    MPI_Send(&compressed_message, 1, MPI_UNSIGNED_LONG_LONG, child_process, COMPRESSED_MESSAGE_TAG, MPI_COMM_WORLD);

    int selected_index_holes_size = selected_index_holes.size();
    if (selected_index_holes_size>0) {
        const int* selected_index_holes_data = selected_index_holes.data();
        MPI_Send(selected_index_holes_data, selected_index_holes_size, MPI_INT, child_process, SELECTED_INDEX_HOLES_TAG, MPI_COMM_WORLD);
    }


    sent_work += power_of_2( (unsigned long long) (num_of_modes - selected_index_holes[hole_to_iterate-1]) - 1);

}



/**
@brief Call to send terminating signal to the child process.
*/
void SendTerminatingSignalToChildProcess() {

    std::vector<int> && child_processes = listener.getChildProcesses();
    if (child_processes.size() == 0) return;


    // the terminating signal is indicated by nullary compressed message
    unsigned long long compressed_message = 0;

    for ( size_t idx=0; idx<child_processes.size(); idx++) {
        MPI_Send(&compressed_message, 1, MPI_UNSIGNED_LONG_LONG, child_processes[idx], COMPRESSED_MESSAGE_TAG, MPI_COMM_WORLD);
    }

    return;


}
#endif




/**
@brief Call to run iterations over the selected modes to calculate partial torontonians
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A^Z.
@param L Matrix conatining partial Cholesky decomposition if the initial matrix to be reused
@param hole_to_iterate The index indicating which hole index should be iterated
@param reuse index The index  labeling the number of rows of the matrix L that can be reused in the Cholesky decomposition
*/
void IterateOverSelectedModes( PicVector<int>& selected_index_holes, int hole_to_iterate, matrix_type &L, const int reuse_index ) {


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




    // logical variable to control whether spawning new iterations or not
    bool stop_spawning_iterations = (selected_index_holes.size() == num_of_modes-1);
    // add new index hole to the iterations
    int new_hole_to_iterate = hole_to_iterate+1;

    // now do the rest of the iterations
    tbb::parallel_for( tbb::blocked_range<int>(index_min+1, index_max, 1), [&](tbb::blocked_range<int> r ) {
        for ( int idx=r.begin(); idx != r.end(); idx++) {

            PicVector<int> selected_index_holes_new = selected_index_holes;
            selected_index_holes_new[hole_to_iterate] = idx-1;

            int reuse_index_new = idx-1-hole_to_iterate < reuse_index ? idx-1-hole_to_iterate : reuse_index;

            matrix_type &&L_new = CreateAZ(selected_index_holes_new, L, reuse_index_new);
            scalar_type_long determinant;
            calc_determinant_cholesky_decomposition<matrix_type, scalar_type, scalar_type_long>(L_new, 2*reuse_index_new, determinant);

            long double partial_torontonian = CalculatePartialTorontonian( selected_index_holes_new, determinant );
            RealM<long double> &torontonian_priv = priv_addend.local();
            torontonian_priv += partial_torontonian;



            // return if new index hole would give no nontrivial result
            // (in this case the partial torontonian is unity and should be counted only once in function calculate)
            if (stop_spawning_iterations) continue;

            selected_index_holes_new.push_back(this->num_of_modes-1);
            reuse_index_new = L_new.rows/2-1;
#ifdef __MPI__

            if ( expected_work < initial_work + received_work - sent_work && selected_index_holes_new[hole_to_iterate] < num_of_modes/3 && selected_index_holes_new[hole_to_iterate] > num_of_modes/8 ) {
                // check for incoming activity message from the child process
                if (listener.CheckChildrenProcessActivity() && !sending_work) {
                    {
                        tbb::spin_mutex::scoped_lock my_lock{my_mutex};
                        listener.CheckForActivityStatusMessage( );
                    }
                }

                if (!listener.CheckChildrenProcessActivity() && !sending_work) {

                    {
                        tbb::spin_mutex::scoped_lock my_lock{my_mutex};
                        listener.CheckForActivityStatusMessage( );

                        if (!listener.CheckChildrenProcessActivity() && !sending_work) {
                            sending_work = true;
                            SendWorkToChildProcess(selected_index_holes_new, new_hole_to_iterate, L_new, reuse_index_new);
                            listener.CheckForActivityStatusMessage( );
                            sending_work = false;
                            continue;
                        }
                    }

                }
            }

            if ( listener.getCurrentActivity() != -1 && expected_work*1.4 < initial_work + received_work - sent_work ) {
                {
                    tbb::spin_mutex::scoped_lock my_lock{my_mutex};
                    listener.DeclineReceivingWork();
                }
            }
            else if( listener.getCurrentActivity() == -1 && expected_work*1.4 >= initial_work + received_work - sent_work) {
                {
                    tbb::spin_mutex::scoped_lock my_lock{my_mutex};
                    listener.ActivateCurrentProcess();
                }
            }

#endif

            IterateOverSelectedModes( selected_index_holes_new, new_hole_to_iterate, L_new, reuse_index_new );
        }
    });


    // first do the first iteration without spawning iterations with new index hole
    selected_index_holes[hole_to_iterate] = index_max-1;

    int reuse_index_new = index_max-1-hole_to_iterate < reuse_index ? index_max-1-hole_to_iterate : reuse_index;
    matrix_type &&L_new = CreateAZ(selected_index_holes, L, reuse_index_new);
    scalar_type_long determinant;
    calc_determinant_cholesky_decomposition<matrix_type, scalar_type, scalar_type_long>(L_new, 2*reuse_index_new, determinant);

    long double partial_torontonian = CalculatePartialTorontonian( selected_index_holes, determinant );
    RealM<long double> &torontonian_priv = priv_addend.local();
    torontonian_priv += partial_torontonian;




}


/**
@brief Call to calculate the partial torontonian for given selected modes and their occupancies
@param selected_index_holes Selected modes which should be omitted from thh input matrix to construct A_Z.
@param determinant The determinant of the submatrix A_Z
@return Returns with the calculated torontonian
*/
long double CalculatePartialTorontonian( const PicVector<int>& selected_index_holes, scalar_type_long &determinant  ) {


    size_t number_selected_modes = num_of_modes - selected_index_holes.size();


    // calculating -1^(N-|Z|)
    long double factor =
                (number_selected_modes + num_of_modes) % 2
                    ? -1.0
                    : 1.0;

    // calculating -1^(number of ones) / sqrt(det(1-A^(Z)))
    long double sqrt_determinant = std::sqrt(convertToDouble(determinant));

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

    int number_selected_modes = num_of_modes - selected_index_holes.size();
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

            if ( idx == (size_t)selected_index_holes[hole_idx] && hole_idx<selected_index_holes.size()) {
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
    memset(AZ.get_data(), 0, AZ.size()*sizeof(scalar_type));
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
    for (int idx = 0; idx < reuse_index; idx++) {
        AZ[2*idx*AZ.stride + 2*idx] = L[2*idx*L.stride + 2*idx];
        AZ[(2*idx+1)*AZ.stride + 2*idx+1] = L[(2*idx+1)*L.stride + 2*idx + 1];
    }

    // copy data from the input matrix and the reusable partial Cholesky decomposition matrix L
    for (int idx = reuse_index; idx < number_selected_modes; idx++) {

        scalar_type* mtx_data = mtx.get_data() + 2*(positions_of_ones[idx]*mtx.stride);
        scalar_type* L_data = L.get_data() + 2*(idx+1)*L.stride;
        scalar_type* AZ_data   = AZ.get_data() + 2*(idx*AZ.stride);


        memcpy(AZ_data, L_data, 2*(idx+1)*sizeof(scalar_type));
        memcpy(AZ_data + AZ.stride, L_data + L.stride, 2*(idx+1)*sizeof(scalar_type));

        for (int jdx = reuse_index; jdx <= idx; jdx++) {
            memcpy( AZ_data + 2*jdx, mtx_data + 2*positions_of_ones[jdx], 2*sizeof(scalar_type) );
        }

        AZ_data   = AZ_data + AZ.stride;
        mtx_data = mtx_data + mtx.stride;

        for (int jdx = reuse_index; jdx <= idx; jdx++) {
            memcpy( AZ_data + 2*jdx, mtx_data + 2*positions_of_ones[jdx], 2*sizeof(scalar_type) );
        }

    }

    return AZ;

}





}; //TorontonianRecursive_Tasks


#endif // CPYTHON





} // PIC
