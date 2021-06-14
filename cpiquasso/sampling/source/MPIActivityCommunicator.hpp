
#ifdef __MPI__
#include <mpi.h>

#define ACTIVITY_TAG 0
#define FINISHED_TAG 1
#define WORK_TAG 100
#define COMPRESSED_MESSAGE_TAG 2
#define SELECTED_INDEX_HOLES_TAG 3


#define NUM_OF_CONNECTIONS 20

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
    /// The list of ranks of the parent MPI processes (from which work can be received)
    std::vector<int> parent_processes;
    /// The list of parent status flags: 1 if the parent process is finished, 0 if the parent process is still working
    std::vector<int> parent_status_flags;
    /// The list of ranks of the child MPI processes (for which work can be transmitted)
    std::vector<int> child_processes;
    /// The list of child status flags: 1 if the child process is occupied, 0 if the child process is idle, -1 if child is not able to recive more work
    std::vector<int> child_status_flags;
    /// 1 if the current process is occupied, 0 if the current process is idle, -1 if child is not able to recive more work
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

    current_activity   = 0;

    size_t num_of_child_processes = NUM_OF_CONNECTIONS < world_size-1-current_rank ? NUM_OF_CONNECTIONS : world_size-1-current_rank;

    // initialize child process stats
    child_processes.reserve(num_of_child_processes);
    child_status_flags.reserve(num_of_child_processes);
    for (size_t idx=current_rank+1; idx<current_rank+1+num_of_child_processes; idx++) {
        child_processes.push_back(idx);
        child_status_flags.push_back(1);
    }


    size_t num_of_parent_processes = NUM_OF_CONNECTIONS < current_rank ? NUM_OF_CONNECTIONS : current_rank;

    // initialize parent process stats
    parent_processes.reserve(num_of_parent_processes);
    parent_status_flags.reserve(num_of_parent_processes);
    for (size_t idx=current_rank-num_of_parent_processes; idx<current_rank; idx++) {
        parent_processes.push_back(idx);
        parent_status_flags.push_back(0);
    }

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
@brief Set the current process to decline to receive any additional work
*/
void DeclineReceivingWork() {

    current_activity = -1;
    SendActivityStatus();
    return;

}


/**
@brief Call to send information about the activity status of the current MPI process to the parent MPI process.
*/
void SendActivityStatus() {

    for (int idx = 0; idx<parent_processes.size(); idx++) {
        MPI_Send(&current_activity, 1, MPI_INT, parent_processes[idx], ACTIVITY_TAG, MPI_COMM_WORLD);
    }

    return;

}



/**
@brief Call to receive information about the activity status of the child MPI process.
@param child_idx The index of the child MPI process from which status report is about to be received
*/
void ListenToActivityStatus( int child_idx) {

    MPI_Recv( &child_status_flags[child_idx], 1, MPI_INT, child_processes[child_idx], ACTIVITY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}





/**
@brief Call to check whether the child MPI process has sent a message about it's activity status, or not. If message has been sent it is received.
*/
void CheckForActivityStatusMessage() {

    for (int idx = 0; idx<child_processes.size(); idx++) {
        int flag = 0;
        MPI_Iprobe(child_processes[idx], ACTIVITY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE );

        if (flag==1) {
            ListenToActivityStatus( idx );
        }
    }

}




/**
@brief Call to check the cached activity status of the children processes and retrive the index of the first idle child process.
@return Return with 1 if all the children processes are active or with 0 if there is at least one idle child process
*/
int CheckChildrenProcessActivity() {

    for (size_t idx = 0; idx<child_status_flags.size(); idx++) {
        if (child_status_flags[idx] == 0) return 0;
    }

    return 1;

}



/**
@brief Call to get the index of the first idle child process
@return Return with the index (not rank) of the first idle child process in the list, or with -1 if all children are active (or there are no children processes)
*/
int getIdleProcessIndex() {

    for (int idx = child_status_flags.size()-1; idx>=0; idx--) {
        if (child_status_flags[idx] == 0) return idx;
    }

    return -1;

}

/**
@brief Call to get the rank of a child process from its index
@return Return with the rank of the child process, or with -1 if there is no appropriate child in the list.
*/
int getChildRank( int idx ) {

    if (idx < child_processes.size() && idx >= 0) return child_processes[idx];


    return -1;


}


/**
@brief Call to retrieve the rank of the parent MPI process
@return Returns with the rank of the parent MPI process
*/
std::vector<int> getParentProcesses() {

    return parent_processes;

}



/**
@brief Call to retrieve the rank of the child MPI process
@return Returns with the rank of the child MPI process
*/
std::vector<int> getChildProcesses() {

    return child_processes;

}



/**
@brief Call to set the status of a parent MPI process to be finished
@param parent_idx The index (not rank) of the parent MPI process
*/
void setParentFinished( int parent_idx ) {

    if ( parent_idx < 0 || parent_idx >= parent_status_flags.size() ) return;

    parent_status_flags[parent_idx] = 1;


}



/**
@brief Call to set the status of a parent MPI process to be finished
@param parent_idx The index (not rank) of the parent MPI process
*/
void setChildActive( int child_idx ) {

    if ( child_idx < 0 || child_idx >= child_status_flags.size() ) return;

    child_status_flags[child_idx] = 1;


}



/**
@brief Call check whether all parents have finshed the work
@return Returns with 1 if all the parents have finished the work.
*/
int checkParentFinished() {


    for (size_t idx=0; idx<parent_status_flags.size(); idx++) {
        if ( parent_status_flags[idx] == 0 ) return 0;
    }


    return 1;

}



/**
@brief Call to get the activity  flag of the current process
@return 1 if the current process is occupied, 0 if the current process is idle, -1 if child is not able to recive more work
*/
int getCurrentActivity() {


    return current_activity;

}



};

#endif // MPI




} // PIC

