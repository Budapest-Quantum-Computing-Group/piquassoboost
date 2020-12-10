#include <iostream>
#include "dependency_graph.h"

namespace pic {

/**
@brief Default constructor of the class.
*/
StartingNode::StartingNode() {
}

/**
@brief Operator to extract a row labeled by i-th element of modes.
@param msg A TBB message fireing the node
*/
const tbb::flow::continue_msg &
StartingNode::operator()(const tbb::flow::continue_msg &msg) {

#ifdef DEBUG
    std::cout<<"starting node fired"<<std::endl;
#endif
    return msg;

}

} // PIC
