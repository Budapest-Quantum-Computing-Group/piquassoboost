#include "PermanentCalculator.h"
#include "common_functionalities.h" // binomialCoeff
namespace pic {


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
    rowSummation = PicState_int(outputState.size());
    for (int i = 0; i < rowSummation.size(); i++){
        rowSummation[i] = 
            rowMultiplicities[i] % 2 == 0 ?
                1 :
                0;
    }

    calculatePermanentWithStartIndex(rowMultiplicities, 0, 1);


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
    //mtx.print_matrix();
    
    return 0;
}









} // PIC
