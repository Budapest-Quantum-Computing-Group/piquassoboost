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


namespace pic {

class PermanentCalculator{
private:
    /// pic matrix which stores the specific matrix we want to work on
    matrix& mtx;

    /// the 2^n factor which is decreased because of the algorithm
    Complex16 normalizationFactor;
    /// column multiplicities determined by the input_state
    PicState_int colMultiplicities;
    /// row number of final matrix
    int finalRowNumber;
    /// column number of final matrix
    int finalColNumber;
    /// array determines whether the specific row has to be added to the first with some multiplicity
    PicState_int rowSummation;
    /// storage for partial permanents
    ComplexM<long double> sumOfPartialPermanents;

public:
    /** @brief Constructs the calculator class.
     *  @param mtx The matrix which we are calculating the permanent of
     */
    PermanentCalculator(matrix& mtx);

    /** @brief Call to calculate the permanent via Glynn formula scaling with n*2^n.
     *         This algorithm improves the calculation in case if there are greater
     *         numbers in the output state
     *  @param mtx The effective scattering matrix of a boson sampling instance
     *  @param input_state The input state
     *  @param output_state The output state
     *  @return Returns with the calculated permanent
     */
    Complex16 calculatePermanent(
        PicState_int64& inputState,
        PicState_int64& outputState
    );

private:
    /** @brief calculates the rowMultiplicities from @param startIndex to
     *         the number of rows in the matrix recursively.
     *  If the multiplicity is greater than 1 it calculates the corresponding
     *  coefficient and calls itself with the specific multiplicity and
     *  coefficient. If the startIndex reaches the number of rows then permanent
     *  is calculated based on Glyyn's formula and storing it in the
     *  member field.
     *  @param rowMultiplicities array storing the multiplicties of the rows
     *  @param startIndex index from where the algorithm shoud calculate
     *  @param coefficient coefficient based on the previous function calls
     */
    void PermanentCalculator::calculatePermanentWithStartIndex(
        PicState_int& rowMultiplicities,
        int startIndex,
        int coefficient
    );

    /** @brief calculates the partial permanent based on Glynn's formula
     *         and stores it in member variable.
     *  
     *  @param rowMultiplicities array storing the multiplicties of the rows
     *  @param coefficient coefficient of the current partial permanent
     */
    void calculatePermanentFromExplicitMatrix(
        PicState_int& rowMultiplicities,
        int coefficient
    );

};

} // PIC

#endif // PERMANENT_CALCULATOR_H
