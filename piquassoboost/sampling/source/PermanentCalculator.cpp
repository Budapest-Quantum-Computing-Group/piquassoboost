#include "PermanentCalculator.h"
#include "GlynnPermanentCalculator.h"

#include "common_functionalities.h" // binomialCoeff

namespace pic {


// the maximal dimension of matrix to be ported to FPGA for permanent calculation
//#define MAX_FPGA_DIM 40
//#define MAX_SINGLE_FPGA_DIM 28

//void calcPermanentGlynn_singleDFE(const pic::Complex16* mtx_data[4], const double* renormalize_data, const uint64_t rows, const uint64_t cols, pic::Complex16* perm);

/*
pic::Complex16 calcPermanenent_DFE(pic::matrix &matrix_mtx){

    pic::Complex16* mtx_data = matrix_mtx.get_data();
    

    // calulate the maximal sum of the columns to normalize the matrix
    pic::matrix colSumMax( matrix_mtx.cols, 1);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(pic::Complex16) );
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            pic::Complex16 value1 = colSumMax[jdx] + matrix_mtx[ idx*matrix_mtx.stride + jdx];
            pic::Complex16 value2 = colSumMax[jdx] - matrix_mtx[ idx*matrix_mtx.stride + jdx];
            if ( std::abs( value1 ) < std::abs( value2 ) ) {
                colSumMax[jdx] = value2;
            }
            else {
                colSumMax[jdx] = value1;
            }

        }

    }




    // calculate the renormalization coefficients
    pic::matrix_base<double> renormalize_data(matrix_mtx.cols, 1);
    for (int jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(colSumMax[jdx]);
    }

    // renormalize the input matrix
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            matrix_mtx[ idx*matrix_mtx.stride + jdx] = matrix_mtx[ idx*matrix_mtx.stride + jdx]/renormalize_data[jdx];
        }

    }    

    // SLR and DFE split input matrices
    pic::matrix mtx_split[4];
    pic::Complex16* mtx_data_split[4];


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

        mtx_split[kdx] = pic::matrix(max_fpga_rows, max_fpga_cols);
        mtx_data_split[kdx] = mtx_split[kdx].get_data();

        pic::Complex16 padding_element(1.0,0.0);
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

        memset( mtx_data_split[kdx] + rows*mtx_split[kdx].stride, 0.0, (max_fpga_rows-rows)*max_fpga_cols*sizeof(pic::Complex16));
    }
    
    pic::Complex16 perm;
    calcPermanentGlynn_singleDFE( mtx_data_split, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, &perm);

    return perm;
}*/


PermanentCalculator::PermanentCalculator(matrix &mtx)
    : mtx(mtx)
{

}

Complex16 PermanentCalculator::calculatePermanent(
    PicState_int64 &inputState,
    PicState_int64 &outputState
){
    sumOfPartialPermanents = ComplexM<long double>();

    // row multiplicities are determined by the output state
    PicState_int rowMultiplicities =
        convert_PicState_int64_to_PicState_int(outputState);

    initialRowMultiplicities = rowMultiplicities.copy();

    // column multiplicities are determined by the input state
    colMultiplicities =
        convert_PicState_int64_to_PicState_int(inputState);

    // create vector of values which determine whether the specific row has to be added or not
    // 0 means not
    // 1 means it has to be added to the first row with multiplicity rowMultiplicity[i]
    finalRowNumber = 0;
    rowSummation = PicState_int(outputState.size());
    
    //std::cout << "rowMultiplicities: "<<std::endl;
    //for (int i = 0; i < rowMultiplicities.size(); i++){
    //    std::cout << rowMultiplicities[i] << " ";
    //}
    //std::cout << std::endl;
    
    for (int i = 0; i < rowSummation.size(); i++){
        if (rowMultiplicities[i] % 2 == 0){
            rowSummation[i] = 1;
        }else{
            rowSummation[i] = 0;
            finalRowNumber++;
        }
    }




    calculatePermanentWithStartIndex(rowMultiplicities, 0, 1);

    mtx.print_matrix();
    return 1;
}



void PermanentCalculator::calculatePermanentWithStartIndex(
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
                //std::cout << "binom: ("<<rowMultiplicity <<","<<multiplicity<<")"<<std::endl;
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
            for (int multiplicity = rowMultiplicity; multiplicity > 0; multiplicity -= 2){
                PicState_int newRowMultiplicities = rowMultiplicities.copy();
                newRowMultiplicities[startIndex] = multiplicity;
                //std::cout << "binom: ("<<rowMultiplicity <<","<<multiplicity<<")"<<std::endl;
                int newCoefficient = coefficient * sign * binomialCoeff(rowMultiplicity, multiplicity);
        
                calculatePermanentWithStartIndex(
                    newRowMultiplicities,
                    startIndex + 1,
                    newCoefficient
                );                
                sign *= -1;
            }
        }

    }
}

Complex16 PermanentCalculator::calculatePermanentFromExplicitMatrix(
    PicState_int& rowMultiplicities,
    int coefficient
){
    // Creating new matrix with the given values
    matrix finalMatrix(finalRowNumber, mtx.cols);
    memset(finalMatrix.get_data(), 0, finalMatrix.cols * sizeof(Complex16));

    int currentRowIndex = 0;
    for (int rowIndex = 0; rowIndex < mtx.rows; rowIndex++){
        if (rowSummation[rowIndex] == 1){
            // adding elements to the first row
            for (int colIndex = 0; colIndex < mtx.cols; colIndex++){
                finalMatrix[colIndex] += rowMultiplicities[rowIndex] * mtx[rowIndex * mtx.stride + colIndex];
            }
        }else{
            for (int colIndex = 0; colIndex < mtx.cols; colIndex++){
                finalMatrix[currentRowIndex * finalMatrix.stride + colIndex] =
                    rowMultiplicities[rowIndex] * mtx[rowIndex * mtx.stride + colIndex];
            }
            currentRowIndex++;
        }
    }

    
    std::cout << "permanent calculation" << std::endl;
    std::cout << "coefficient: " << coefficient << std::endl;
    std::cout << "rowSummation: ";
    for (int i = 0; i < rowSummation.size(); i++){
        std::cout << rowSummation[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "colMultiplicities: ";
    for (int i = 0; i < colMultiplicities.size(); i++){
        std::cout << colMultiplicities[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "rowMultiplicities: ";
    for (int i = 0; i < rowMultiplicities.size(); i++){
        std::cout << rowMultiplicities[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Final matrix:";
    finalMatrix.print_matrix();
    //mtx.print_matrix();
    
    matrix finalMatrix2 = finalMatrix.copy();
    Complex16 partialPermanent_DFE = 1; //calcPermanenent_DFE(finalMatrix);
    GlynnPermanentCalculator glynnCalculatorCPU;
    Complex16 partialPermanent_CPU = glynnCalculatorCPU.calculate(finalMatrix2);

    std::cout << "DFE: "<< partialPermanent_DFE<< std::endl;
    std::cout << "CPU: "<< partialPermanent_CPU<< std::endl;


    return 0;
}









} // PIC
