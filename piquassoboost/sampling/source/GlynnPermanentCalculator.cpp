/**
 * Copyright 2022 Budapest Quantum Computing Group
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

#include "GlynnPermanentCalculator.hpp"


namespace pic {


/** Explicit specialization of template classes GlynnPermanentCalculator
 */

/// Specialization based on pic::matrix32, pic::Complex_base<long double> and long double
template class GlynnPermanentCalculator<pic::matrix32, long double>;

/// Specialization based on pic::matrix, pic::Complex_base<double> and double
template class GlynnPermanentCalculator<pic::matrix, double>;

/// Specialization based on pic::matrix8, pic::Complex_base<float> and float
template class GlynnPermanentCalculator<pic::matrix8, float>;


} // PIC
