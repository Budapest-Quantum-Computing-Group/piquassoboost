/**
 * Copyright 2021 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
