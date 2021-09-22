// The aim of this script is to ensure every implementation of permanent calculator give concise results.


#include "GlynnPermanentCalculator.h"
#include "GlynnPermanentCalculatorRecursive.h"
#include "CChinHuhPermanentCalculator.h"
#include "matrix_helper.hpp"
#include "matrix32.h"


bool AreClose(const pic::Complex16 &a, const pic::Complex16 &b){
    double eps = 1e-6;

    if(fabs(a.real() - b.real()) > eps) return false;
    if(fabs(a.imag() - b.imag()) > eps) return false;

    return true;
}


void testOfSameValuesForRandomMatrix(int matrix_dimension, int mul){
    auto matrix = pic::getRandomComplexMatrix<pic::matrix, pic::Complex16>(matrix_dimension, pic::RANDOM);

    auto matrixMultipled = pic::matrix(mul * matrix.rows, mul * matrix.cols);
    for (unsigned int i = 0; i < matrixMultipled.rows; i++){
        for (unsigned int j = 0; j < matrixMultipled.cols; j++){
            matrixMultipled[i * matrixMultipled.stride + j] = matrix[(i / mul) * matrix.stride + (j / mul)];
        }
    }


    // First just check if I can get any results from BBFG formula.
    pic::GlynnPermanentCalculator bbfg_calculator;

    auto bbfg_permanent = bbfg_calculator.calculate(matrixMultipled);


    pic::CChinHuhPermanentCalculator cChinHuhPermanentCalculator;
    pic::PicState_int64 in_out_state(mul*matrix_dimension, 1);

    auto ch_permanent = cChinHuhPermanentCalculator.calculate(
            matrixMultipled, in_out_state, in_out_state
            );


    pic::PicState_int64 rowMultiplicities(matrix_dimension);
    pic::PicState_int64 colMultiplicities(matrix_dimension);
    for (int i = 0; i < matrix_dimension; i++){
        rowMultiplicities[i] = mul;
        colMultiplicities[i] = mul;
    }
    pic::GlynnPermanentCalculatorRecursive bbfgrec_calculator;
    auto bbfgrec_permanent = bbfgrec_calculator.calculate(matrix, rowMultiplicities, colMultiplicities);
    
    if (AreClose(bbfgrec_permanent, bbfg_permanent) && AreClose(bbfgrec_permanent, ch_permanent)){
        return;
    }
    exit(-1);
}

int main() {

    printf("\n\n********************************************************\n");
    printf("Test of permanents calculators accuracy\n");
    printf("********************************************************\n\n\n");

    constexpr int iterationNumber = 10;
    for (int dim = 2; dim < 10; dim++){
        for (int mul = 1; mul < 3; mul++){
            if (dim*mul < 10){
                for (int i = 0; i < iterationNumber; i++){
                    testOfSameValuesForRandomMatrix(dim, mul);
                }
            }
        }
    }


    int matrix_dimension = 20;
    int mul = 1;

    auto matrix = pic::getRandomComplexMatrix<pic::matrix, pic::Complex16>(matrix_dimension, pic::RANDOM);

    auto matrixMultipled = pic::matrix(mul * matrix.rows, mul * matrix.cols);
    for (unsigned int i = 0; i < matrixMultipled.rows; i++){
        for (unsigned int j = 0; j < matrixMultipled.cols; j++){
            matrixMultipled[i * matrixMultipled.stride + j] = matrix[(i / mul) * matrix.stride + (j / mul)];
        }
    }


    // First just check if I can get any results from BBFG formula.
    pic::GlynnPermanentCalculator bbfg_calculator;

    tbb::tick_count t0 = tbb::tick_count::now();
    auto bbfg_permanent = bbfg_calculator.calculate(matrixMultipled);
    tbb::tick_count t1 = tbb::tick_count::now();
    
    printf("BBFG Permanent: %4.6f + %4.6f i\n", bbfg_permanent.real(), bbfg_permanent.imag());

    pic::CChinHuhPermanentCalculator cChinHuhPermanentCalculator;
    pic::PicState_int64 in_out_state(matrix_dimension, 1);

    tbb::tick_count t2 = tbb::tick_count::now();
    auto ch_permanent = cChinHuhPermanentCalculator.calculate(
        matrixMultipled, in_out_state, in_out_state
    );
    tbb::tick_count t3 = tbb::tick_count::now();

    printf("ChHu Permanent: %4.6f + %4.6f i\n", ch_permanent.real(), ch_permanent.imag());

    pic::PicState_int64 rowMultiplicities(matrix_dimension);
    pic::PicState_int64 colMultiplicities(matrix_dimension);
    for (int i = 0; i < matrix_dimension; i++){
        rowMultiplicities[i] = mul;
        colMultiplicities[i] = mul;
    }
    pic::GlynnPermanentCalculatorRecursive bbfgrec_calculator;

    tbb::tick_count t4 = tbb::tick_count::now();
    auto bbfgrec_permanent = bbfgrec_calculator.calculate(matrix, rowMultiplicities, colMultiplicities);
    tbb::tick_count t5 = tbb::tick_count::now();
    printf("GlRe Permanent: %4.6f + %4.6f i\n", bbfgrec_permanent.real(), bbfgrec_permanent.imag());
    
    std::cout << "BBFG time: " << (t1-t0).seconds() << std::endl;
    std::cout << "ChHu time: " << (t3-t2).seconds() << std::endl;
    std::cout << "GlRe time: " << (t5-t4).seconds() << std::endl;

    std::cout << "All test cases were passed." << std::endl;
    
}