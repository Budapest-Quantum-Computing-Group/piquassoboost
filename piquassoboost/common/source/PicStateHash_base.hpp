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

#ifndef PICSTATEHASH_BASE_HPP
#define PICSTATEHASH_BASE_HPP

#include "PicState_base.hpp"

namespace pic {


/**
@brief Class to hash function operator for PicState keys in unordered maps
*/
template <typename intType>
class PicStateHash_base {

protected:

public:

/**
@brief Constructor of the class.
@return Returns with the instance of the class.
*/
PicStateHash_base() {

}

/**
@brief Operator to generate hash key for class instance PicState_base<intType>
@param key An instance of class PicState
@return Returns with the calculated hash value.
*/
size_t
operator()(const PicState_base<intType> &key) const {

    PicState_base<intType> &key_loc = const_cast<PicState_base<intType> &>(key);
    int64_t *data = key_loc.get_data();
    size_t hash_val = 0;
    size_t pow2 = 1;

    for (size_t idx=0; idx<key.cols; idx++) {
        hash_val = hash_val + data[idx]*pow2;
        pow2 = pow2*2;
    }

    return hash_val;
}


}; //PicStateHash_base


} // PIC



#endif // PICSTATEHASH_BASE_HPP
