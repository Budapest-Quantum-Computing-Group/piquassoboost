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

#ifndef PICSTATE_H
#define PICSTATE_H

#include <functional>
#include <matrix_base.hpp>
#include <PicState_base.hpp>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/concurrent_vector.h>
#include <vector>



namespace pic {

/// alias for Piquassoboost state with values of type int64_t
/// Compatible with the Piquasso numpy interface.
using PicState_int64 = PicState_base<int64_t>;

/// container of aligned states aligned to cache line border
using PicStates = std::vector< PicState_int64, tbb::cache_aligned_allocator<PicState_int64> >;

/// concurrent container of aligned states aligned to cache line border
using concurrent_PicStates = tbb::concurrent_vector<PicState_int64, tbb::cache_aligned_allocator<PicState_int64> >;


/// alias for Piquassoboost state with values of type int64_t
using PicState_int = PicState_base<int>;


/** @brief Call to convert a piquassoboost state from PicState_int64 to PicState_int type
 *  @param state_int64 The state to convert
 *  @returns with the converted state of type PicState_int
 */
PicState_int convert_PicState_int64_to_PicState_int(PicState_int64& state_int64);

} // PIC

#endif
