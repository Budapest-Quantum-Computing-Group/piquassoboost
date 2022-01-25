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

#include "GlynnPermanentCalculatorDFE.h"

#include "common_functionalities.h" // binomialCoeff, power_of_2


#ifndef CPYTHON
#include <tbb/tbb.h>
#endif


namespace pic {

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculator_DFEDualCard(matrix& matrix_mtx, Complex16& perm)
{
    

    // calulate the maximal sum of the columns to normalize the matrix
    matrix colSumMax( matrix_mtx.cols, 1);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex16) );
    for (size_t idx=0; idx<matrix_mtx.rows; idx++) {
        for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
            Complex16 value1 = colSumMax[jdx] + matrix_mtx[ idx*matrix_mtx.stride + jdx];
            Complex16 value2 = colSumMax[jdx] - matrix_mtx[ idx*matrix_mtx.stride + jdx];
            if ( std::abs( value1 ) < std::abs( value2 ) ) {
                colSumMax[jdx] = value2;
            }
            else {
                colSumMax[jdx] = value1;
            }

        }

    }




    // calculate the renormalization coefficients
    matrix_base<double> renormalize_data(matrix_mtx.cols, 1);
    for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(colSumMax[jdx]);
    }

    // renormalize the input matrix
    for (size_t idx=0; idx<matrix_mtx.rows; idx++) {
        for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
            matrix_mtx[ idx*matrix_mtx.stride + jdx] = matrix_mtx[ idx*matrix_mtx.stride + jdx]/renormalize_data[jdx];
        }

    }    


    // SLR and DFE split input matrices
    matrix mtx_split[8];
    Complex16* mtx_data_split[8];


    size_t max_fpga_rows =  MAX_FPGA_DIM;
    size_t max_fpga_cols =  MAX_FPGA_DIM/8;

    // SLR splitted data for the first DFE card
    size_t cols_half1_tot = matrix_mtx.cols/2;
    size_t cols_half2_tot = matrix_mtx.cols - cols_half1_tot;

    size_t rows = matrix_mtx.rows;
    size_t cols_half1[4];
    cols_half1[0] = max_fpga_cols < cols_half1_tot ? max_fpga_cols : cols_half1_tot;
    cols_half1[1] = max_fpga_cols < (cols_half1_tot -cols_half1[0]) ? max_fpga_cols : (cols_half1_tot-cols_half1[0]);
    cols_half1[2] = max_fpga_cols < (cols_half1_tot - cols_half1[0] - cols_half1[1]) ? max_fpga_cols : (cols_half1_tot - cols_half1[0] - cols_half1[1]);
    cols_half1[3] = max_fpga_cols < (cols_half1_tot - cols_half1[0] - cols_half1[1] - cols_half1[2]) ? max_fpga_cols : (cols_half1_tot - cols_half1[0] - cols_half1[1] - cols_half1[2]);


    size_t col_offset = 0;
    for (int kdx=0; kdx<4; kdx++) {

        mtx_split[kdx] = matrix(max_fpga_rows, max_fpga_cols);
        mtx_data_split[kdx] = mtx_split[kdx].get_data();

        Complex16 padding_element(1.0,0.0);
        for (size_t idx=0; idx<rows; idx++) {
            size_t offset = idx*matrix_mtx.stride+col_offset;
            size_t offset_small = idx*mtx_split[kdx].stride;
            for (size_t jdx=0; jdx<cols_half1[kdx]; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = matrix_mtx[offset+jdx];
            }

            for (size_t jdx=cols_half1[kdx]; jdx<max_fpga_cols; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = padding_element;
            }
            padding_element.real(0.0);
        }
        col_offset = col_offset + cols_half1[kdx];

        memset( mtx_data_split[kdx] + rows*mtx_split[kdx].stride, 0.0, (max_fpga_rows-rows)*max_fpga_cols*sizeof(Complex16));
    }


    // SLR splitted data for the second DFE card
    size_t cols_half2[4];
    cols_half2[0] = max_fpga_cols < cols_half2_tot ? max_fpga_cols : cols_half2_tot;
    cols_half2[1] = max_fpga_cols < (cols_half2_tot - cols_half2[0]) ? max_fpga_cols : (cols_half2_tot - cols_half2[0]);
    cols_half2[2] = max_fpga_cols < (cols_half2_tot - cols_half2[0] - cols_half2[1]) ? max_fpga_cols : (cols_half2_tot - cols_half2[0] - cols_half2[1]);
    cols_half2[3] = max_fpga_cols < (cols_half2_tot - cols_half2[0] - cols_half2[1] - cols_half2[2]) ? max_fpga_cols : (cols_half2_tot - cols_half2[0] - cols_half2[1] - cols_half2[2]);

    for (int kdx=0; kdx<4; kdx++) {

        mtx_split[kdx+4] = matrix(max_fpga_rows, max_fpga_cols);
        mtx_data_split[kdx+4] = mtx_split[kdx+4].get_data();

        Complex16 padding_element(1.0,0.0);
        for (size_t idx=0; idx<rows; idx++) {
            size_t offset = idx*matrix_mtx.stride+col_offset;
            size_t offset_small = idx*mtx_split[kdx+4].stride;
            for (size_t jdx=0; jdx<cols_half2[kdx]; jdx++) {
                mtx_data_split[kdx+4][offset_small+jdx] = matrix_mtx[offset+jdx];
            }

            for (size_t jdx=cols_half2[kdx]; jdx<max_fpga_cols; jdx++) {
                mtx_data_split[kdx+4][offset_small+jdx] = padding_element;
            }
            padding_element.real(0.0);

        }
        col_offset = col_offset + cols_half2[kdx];
        memset( mtx_data_split[kdx+4] + rows*mtx_split[kdx+4].stride, 0.0, (max_fpga_rows-rows)*max_fpga_cols*sizeof(Complex16));
    }


/*
matrix_mtx.print_matrix();
for (int idx=0; idx<8; idx++) {
   mtx_split[idx].print_matrix();
}
*/
    
    calcPermanentGlynn_DualDFE( mtx_data_split, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, &perm);


    return;
}



/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculator_DFESingleCard(matrix& matrix_mtx, Complex16& perm) {

    

    // calulate the maximal sum of the columns to normalize the matrix
    matrix colSumMax( matrix_mtx.cols, 1);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex16) );
    for (size_t idx=0; idx<matrix_mtx.rows; idx++) {
        for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
            Complex16 value1 = colSumMax[jdx] + matrix_mtx[ idx*matrix_mtx.stride + jdx];
            Complex16 value2 = colSumMax[jdx] - matrix_mtx[ idx*matrix_mtx.stride + jdx];
            if ( std::abs( value1 ) < std::abs( value2 ) ) {
                colSumMax[jdx] = value2;
            }
            else {
                colSumMax[jdx] = value1;
            }

        }

    }




    // calculate the renormalization coefficients
    matrix_base<double> renormalize_data(matrix_mtx.cols, 1);
    for (size_t jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(colSumMax[jdx]);
    }

    // renormalize the input matrix
    for (size_t idx=0; idx<matrix_mtx.rows; idx++) {
        for( size_t jdx=0; jdx<matrix_mtx.cols; jdx++) {
            matrix_mtx[ idx*matrix_mtx.stride + jdx] = matrix_mtx[ idx*matrix_mtx.stride + jdx]/renormalize_data[jdx];
        }

    }    

    // SLR and DFE split input matrices
    matrix mtx_split[4];
    Complex16* mtx_data_split[4];


    size_t max_fpga_rows =  MAX_SINGLE_FPGA_DIM;
    size_t max_fpga_cols =  MAX_SINGLE_FPGA_DIM/4;

    // SLR splitted data for the DFE card

    size_t rows = matrix_mtx.rows;
    size_t cols_split[4];
    cols_split[0] = max_fpga_cols < matrix_mtx.cols ? max_fpga_cols : matrix_mtx.cols;
    cols_split[1] = max_fpga_cols < (matrix_mtx.cols-cols_split[0]) ? max_fpga_cols : (matrix_mtx.cols-cols_split[0]);
    cols_split[2] = max_fpga_cols < (matrix_mtx.cols - cols_split[0] - cols_split[1]) ? max_fpga_cols : (matrix_mtx.cols - cols_split[0] - cols_split[1]);
    cols_split[3] = max_fpga_cols < (matrix_mtx.cols - cols_split[0] - cols_split[1] - cols_split[2]) ? max_fpga_cols : (matrix_mtx.cols - cols_split[0] - cols_split[1] - cols_split[2]);


    size_t col_offset = 0;
    for (int kdx=0; kdx<4; kdx++) {

        mtx_split[kdx] = matrix(max_fpga_rows, max_fpga_cols);
        mtx_data_split[kdx] = mtx_split[kdx].get_data();

        Complex16 padding_element(1.0,0.0);
        for (size_t idx=0; idx<rows; idx++) {
            size_t offset = idx*matrix_mtx.stride+col_offset;
            size_t offset_small = idx*mtx_split[kdx].stride;
            for (size_t jdx=0; jdx<cols_split[kdx]; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = matrix_mtx[offset+jdx];
            }

            for (size_t jdx=cols_split[kdx]; jdx<max_fpga_cols; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = padding_element;
            }
            padding_element.real(0.0);
        }
        col_offset = col_offset + cols_split[kdx];

        memset( mtx_data_split[kdx] + rows*mtx_split[kdx].stride, 0.0, (max_fpga_rows-rows)*max_fpga_cols*sizeof(Complex16));
    }
/*
matrix_mtx.print_matrix();
for (int idx=0; idx<4; idx++) {
   mtx_split[idx].print_matrix();
}
*/

    calcPermanentGlynn_SingleDFE( mtx_data_split, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, &perm);


    return;

}



GlynnPermanentCalculatorDFE::GlynnPermanentCalculatorDFE(matrix &mtx)
    : mtx(mtx)
{

}


Complex16 GlynnPermanentCalculatorDFE::calculatePermanent(
    PicState_int64 &inputState,
    PicState_int64 &outputState
){

    sumOfPartialPermanents = ComplexM<double>();

    // row multiplicities are determined by the output state
    PicState_int rowMultiplicities =
        convert_PicState_int64_to_PicState_int(outputState);

    // column multiplicities are determined by the input state
    colMultiplicities =
        convert_PicState_int64_to_PicState_int(inputState);

    // create vector of values which determine whether the specific row has to be added or not
    // 0 means not
    // 1 means it has to be added to the first row with multiplicity rowMultiplicity[i]


    // first row always has to be there! (it is not calculated explicitly just here)
    // this number is updated based on the parity of the rowMultiplicities
    finalRowNumber = 1;
    
    // we are reducing the size of the matrix, the normalization factor of the BB/FG algorithm has to be updated manually
    // we always get the same size of matrix, hence, the factor has to be updated once
    normalizationFactor = 1;

    // rowSummation shows whether the current row has to added to the first row or not with specific multiplicities
    rowSummation = PicState_int(outputState.size());
    // first row is always there, it is always updated with the current multiplicity.
    rowSummation[0] = 0;

    // checking the paritiy of each multiplicity
    for (size_t i = 1; i < rowSummation.size(); i++){
        if ( 0 == rowMultiplicities[i] % 2 ){
            rowSummation[i] = 1;
            normalizationFactor *= 1.0 / power_of_2(rowMultiplicities[i]);
        }else{
            rowSummation[i] = 0;
            finalRowNumber++;
            if (rowMultiplicities[i] > 1){
                normalizationFactor *= 1.0 / power_of_2(rowMultiplicities[i]-1);
            }
        }
    }

    // final number of columns. This has to be calculated once as well.
    finalColNumber = 0;
    for (size_t i = 0; i < colMultiplicities.size(); i++){
        finalColNumber += colMultiplicities[i];
    }

    // first row is calculated differently from the others since all the deltas can not be -1's
    if (rowMultiplicities[0] > 0){
        int currentMultiplicity = rowMultiplicities[0];

        normalizationFactor *= 1.0 / power_of_2(currentMultiplicity-1);

        int sign = 1;
        int numberOfMinuses = 0;
        // the difference is in the binomial coefficient and the limit of the loop
        for (int multiplicity = currentMultiplicity; multiplicity > -currentMultiplicity; multiplicity -= 2){
            int coefficient = sign * binomialCoeff(currentMultiplicity-1, numberOfMinuses);
            PicState_int newRowMultiplicities = rowMultiplicities.copy();
            newRowMultiplicities[0] = multiplicity;
            
            calculatePermanentWithStartIndex(newRowMultiplicities, 1, coefficient);
            
            sign *= -1;
            numberOfMinuses++;
        }
    }else{
        calculatePermanentWithStartIndex(rowMultiplicities, 1, 1);
    }


    Complex16 sumOfPermanents = sumOfPartialPermanents.get();
    Complex16 finalPermanent = sumOfPermanents * normalizationFactor;

    return finalPermanent;
}


void GlynnPermanentCalculatorDFE::calculatePermanentWithStartIndex(
    PicState_int& rowMultiplicities,
    int startIndex,
    int coefficient
){
    const int rows = rowMultiplicities.size();

    while (startIndex < rows && rowMultiplicities[startIndex] <= 1){
        startIndex++;
    }

    if (startIndex == rows){
        // create matrix with the given values
        // calculate permanent
        calculatePermanentFromExplicitMatrix(
            rowMultiplicities,
            coefficient
        );
    }else{
        // here the multiplicity is higher than 1 !
        const int rowMultiplicity = rowMultiplicities[startIndex];
        // even case
        if ( 0 == rowMultiplicity % 2 ){
            // for each number until the rowMultiplicities[i]

            // create new copy of rowMultiplicities.
            // rowMultiplicities has to be each values corresponding to rowMultiplicities[i]
            int sign = 1;
            int countOfMinuses = 0;
            for (int multiplicity = -rowMultiplicity; multiplicity <= rowMultiplicity; multiplicity += 2){
                PicState_int newRowMultiplicities = rowMultiplicities.copy();
                newRowMultiplicities[startIndex] = multiplicity;
                // coefficient is multiplied by the binomial coefficient multiplicity over rowMultiplicity and
                // the sign determined by multiplicity modulo 4

                int newCoefficient = coefficient * sign * binomialCoeff(rowMultiplicity, countOfMinuses);

                calculatePermanentWithStartIndex(
                    newRowMultiplicities,
                    startIndex + 1,
                    newCoefficient
                );
                sign *= -1;
                countOfMinuses++;
            }
        }
        // odd case
        else{
            // create other matrix with the same rows
            // the i'th row has to be multiplied with the numbers from 1 to rowMultiplicities[i]
            // sum up the calculated values with coefficients
            int sign = 1;
            int countOfPlusOnes = rowMultiplicity;
            for (int multiplicity = rowMultiplicity; multiplicity > 0; multiplicity -= 2){
                PicState_int newRowMultiplicities = rowMultiplicities.copy();

                newRowMultiplicities[startIndex] = multiplicity;
                int newCoefficient = coefficient * sign * binomialCoeff(rowMultiplicity, countOfPlusOnes);
        
                calculatePermanentWithStartIndex(
                    newRowMultiplicities,
                    startIndex + 1,
                    newCoefficient
                );                
                sign *= -1;
                countOfPlusOnes -= 1;
            }
        }

    }
}


void GlynnPermanentCalculatorDFE::calculatePermanentFromExplicitMatrix(
    PicState_int& rowMultiplicities,
    int coefficient
){
    // Creating new matrix with the given values
    // This can be further developed by calculating the rows before and storing them
    // in a matrix with column multiplicities
    matrix finalMatrix(finalRowNumber, finalColNumber);

    //std::cout << "row multiplicities: ";
    //for (size_t i = 0; i < rowMultiplicities.size(); i++){
    //    std::cout << rowMultiplicities[i] << " ";
    //}
    //std::cout << std::endl;

    int currentRowIndex = 0;
    for (size_t rowIndex = 0; rowIndex < mtx.rows; rowIndex++){
        if (rowSummation[rowIndex] == 1){
            int currentColIndex = 0;
            // adding elements to the first row
            for (size_t colIndex = 0; colIndex < mtx.cols; colIndex++){
                for (int q = 0; q < colMultiplicities[colIndex]; q++){
                    finalMatrix[currentColIndex] += rowMultiplicities[rowIndex] * mtx[rowIndex * mtx.stride + colIndex];
                    currentColIndex++;
                }
            }
        }else{
            int currentColIndex = 0;
            for (size_t colIndex = 0; colIndex < mtx.cols; colIndex++){
                for (int q = 0; q < colMultiplicities[colIndex]; q++){
                    finalMatrix[currentRowIndex * finalMatrix.stride + currentColIndex] =
                        rowMultiplicities[rowIndex] * mtx[rowIndex * mtx.stride + colIndex];
                    currentColIndex++;
                }
            }
            currentRowIndex++;
        }
    }
    
    matrix finalMatrix2 = finalMatrix.copy();

    Complex16 partialPermanent_DFE;
    GlynnPermanentCalculator_DFEDualCard(finalMatrix, partialPermanent_DFE);

    double coefficientDouble = coefficient;
    sumOfPartialPermanents += partialPermanent_DFE * coefficientDouble;

    return;
}



} // namespace pic
