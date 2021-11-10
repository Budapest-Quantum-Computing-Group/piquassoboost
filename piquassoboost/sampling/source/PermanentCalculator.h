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

#ifndef PERMANENT_CALCULATOR_H
#define PERMANENT_CALCULATOR_H

#include "matrix.h"
#include "matrix32.h"
#include "PicState.h"

#ifndef CPYTHON
#include <tbb/tbb.h>
#endif

namespace pic {

class PermanentCalculator{
public:
    matrix& mtx;
    //PicState_int rowMultiplicities;
    PicState_int colMultiplicities;
    PicState_int initialRowMultiplicities;

    // row number of final matrix
    int finalRowNumber;
    int finalColNumber;
    PicState_int rowSummation;
    /// storage for partial permanents
    ComplexM<long double> sumOfPartialPermanents;

    PermanentCalculator(matrix& mtx);

    Complex16 calculatePermanent(
        PicState_int64& inputState,
        PicState_int64& outputState
    );

    void PermanentCalculator::calculatePermanentWithStartIndex(
        PicState_int& rowMultiplicities,
        int startIndex,
        int coefficient
    );

    Complex16 calculate(matrix mtx);

    Complex16 calculatePermanentFromExplicitMatrix(
        PicState_int& rowMultiplicities,
        int coefficient
    );

};

} // PIC

#endif // PERMANENT_CALCULATOR_H
