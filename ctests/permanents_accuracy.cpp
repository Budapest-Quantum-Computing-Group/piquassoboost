// The aim of this script is to ensure every implementation of permanent calculator give concise results.

#include "GlynnPermanentCalculator.h"
#include "CChinHuhPermanentCalculator.h"
#include "matrix_helper.hpp"
#include "matrix32.h"

bool AreClose(const pic::Complex16 &a, const pic::Complex16 &b){
    double eps = 1e-6;

    if(fabs(a.real() - b.real()) > eps) return false;
    if(fabs(a.imag() - b.imag()) > eps) return false;

    return true;
}

int main() {

    printf("\n\n********************************************************\n");
    printf("Test of permanents calculators accuracy\n");
    printf("********************************************************\n\n\n");

    int matrix_dimension = 10;

    auto matrix = pic::getRandomComplexMatrix<pic::matrix, pic::Complex16>(matrix_dimension, pic::RANDOM);

    // First just check if I can get any results from BBFG formula.
    pic::GlynnPermanentCalculator bbfg_calculator;

    auto bbfg_permanent = bbfg_calculator.calculate(matrix);

    printf("BBFG Permanent: %4.6f + %4.6f i\n", bbfg_permanent.real(), bbfg_permanent.imag());

    pic::CChinHuhPermanentCalculator cChinHuhPermanentCalculator;
    pic::PicState_int64 in_out_state(matrix_dimension, 1);

    auto ch_permanent = cChinHuhPermanentCalculator.calculate(
            matrix, in_out_state, in_out_state
            );

    printf("ChHu Permanent: %4.6f + %4.6f i\n", ch_permanent.real(), ch_permanent.imag());

    printf("\n\n********************************************************\n");
    printf("Test passed: %s\n", AreClose(bbfg_permanent, ch_permanent) ? "True" : "False");
    printf("********************************************************\n\n\n");
}