
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




} // PIC

