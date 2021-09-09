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

#ifndef PICSTATEHASH_H
#define PICSTATEHASH_H

#include "PicState.h"

namespace pic {


/**
@brief Class to hash function operator for PicState keys in unordered maps
*/
class PicStateHash {

protected:

public:

/**
@brief Constructor of the class.
@return Returns with the instance of the class.
*/
PicStateHash();

/**
@brief Operator to generate hash key for class instance PicState_int64
@param key An instance of class PicState
@return Returns with the calculated hash value.
*/
size_t operator()(const PicState_int64 &key) const;


}; //PicStateHash







} // PIC

#endif
