#ifndef DEPENDENCY_GRAPH_H
#define DEPENDENCY_GRAPH_H

#include "tbb/tbb.h"

using Node = tbb::flow::continue_node<tbb::flow::continue_msg>;
using NodePtr = std::shared_ptr<Node>;

namespace pic {

/**
@brief Class representing the starting node used in the dependency graphs
*/
class StartingNode {


public:

/**
@brief Default constructor of the class.
*/
StartingNode();

/**
@brief Operator to extract a row labeled by i-th element of modes.
@param msg A TBB message fireing the node
*/
const tbb::flow::continue_msg & operator()(const tbb::flow::continue_msg &msg);




}; //StartingNode


} // PIC

#endif
