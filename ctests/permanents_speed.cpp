// The aim of this script is to compare runtimes of each implementation of permanent calculator.


#include "GlynnPermanentCalculator.hpp"
#include "GlynnPermanentCalculatorRepeated.h"
#include "CChinHuhPermanentCalculator.h"
#include "CGeneralizedCliffordsSimulationStrategy.h"
#include "matrix_helper.hpp"

#ifdef __MPI__
#include <mpi.h>
#endif // MPI

constexpr bool printTimes = true;

bool AreClose(const pic::Complex16 &a, const pic::Complex16 &b){
    double eps = 1e-6;

    if(fabs(a.real() - b.real()) > eps) return false;
    if(fabs(a.imag() - b.imag()) > eps) return false;

    return true;
}

void testCase(std::vector<int> input, std::vector<int> output, int iterationNumber){
    std::cout << "Test case:" << std::endl;

    std::cout << "Input  : ";
    for (auto elem : input){
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    std::cout << "Output : ";
    for (auto elem : output){
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    std::cout << "Iteration: " << iterationNumber << std::endl;

    int inputLength = 0;
    int outputLength = 0;
    int sumInput = 0;
    for (auto elem : input){
        sumInput += elem;
        inputLength++;
    }
    int sumOutput = 0;
    for (auto elem : output){
        sumOutput += elem;
        outputLength++;
    }
    
    if (sumInput != sumOutput || inputLength != outputLength){
        std::cout << "Please give me equal number of particles at input and output sides!" << std::endl;
        std::cout << "Input : ";
        for (auto elem : input){
            std::cout << elem << " ";
        }
        std::cout << std::endl;
        std::cout << "Output : ";
        for (auto elem : output){
            std::cout << elem << " ";
        }
        std::cout << std::endl;
        return;
    }

    std::vector<pic::matrix> normalMatrices;
    for (int i = 0; i < iterationNumber; i++){
        normalMatrices.push_back(
            pic::getRandomComplexMatrix<pic::matrix, pic::Complex16>(inputLength, pic::RANDOM)
        );
    }
    pic::PicState_int64 inputState(input.size());
    pic::PicState_int64 outputState(output.size());
    for (size_t i = 0; i < input.size(); i++){
        inputState[i] = input[i];
    }
    for (size_t i = 0; i < output.size(); i++){
        outputState[i] = output[i];
    }
    std::vector<pic::matrix> adaptedMatricesForNormalGlynn;
    for (int i = 0; i < iterationNumber; i++){
        adaptedMatricesForNormalGlynn.push_back(
            pic::adaptInterferometerGlynnMultiplied(normalMatrices[i], &inputState, &outputState)
        );
    }

    std::vector<pic::matrix> adaptedMatricesForRecursiveGlynn;
    for (int i = 0; i < iterationNumber; i++){
        adaptedMatricesForRecursiveGlynn.push_back(
            pic::adaptInterferometer(normalMatrices[i], inputState, outputState)
        );
    }
    std::function<bool(int64_t)> nonZero = [](int64_t elem){
        return elem > 0;
    };
    pic::PicState_int64 adaptedInputState = inputState.filter(nonZero);
    pic::PicState_int64 adaptedOutputState = outputState.filter(nonZero);
    

    tbb::tick_count timeZero = tbb::tick_count::now();
    auto sumTimeBBFG = timeZero-timeZero;
    auto sumTimeBBFGREC = timeZero-timeZero;
    std::vector<pic::Complex16> bbfg_results;
    std::vector<pic::Complex16> bbfgrec_results;

    for (int i = 0; i < iterationNumber; i++){
        pic::matrix& adaptedMatrix = adaptedMatricesForNormalGlynn[i];
        pic::GlynnPermanentCalculatorLongDouble bbfg_calculator;

        tbb::tick_count t0 = tbb::tick_count::now();
        auto bbfg_permanent = bbfg_calculator.calculate(adaptedMatrix);
        tbb::tick_count t1 = tbb::tick_count::now();

        bbfg_results.push_back(bbfg_permanent);

        sumTimeBBFG += t1-t0;
    }
    
    for (int i = 0; i < iterationNumber; i++){
        pic::matrix adaptedMatrix = adaptedMatricesForRecursiveGlynn[i];
        pic::PicState_int64 inState = inputState;
        pic::PicState_int64 outState = outputState;
        pic::GlynnPermanentCalculatorRepeated bbfgrec_calculator;

        tbb::tick_count t0 = tbb::tick_count::now();
        auto bbfgrec_permanent = bbfgrec_calculator.calculate(adaptedMatrix, adaptedInputState, adaptedOutputState);
        tbb::tick_count t1 = tbb::tick_count::now();

        bbfgrec_results.push_back(bbfgrec_permanent);

        sumTimeBBFGREC += t1-t0;
    }

    for (int i = 0; i < iterationNumber; i++){
        if (!AreClose(bbfg_results[i], bbfgrec_results[i])){
            std::cout << "Permanents are not the same!" << i << "th case" << std::endl;
            std::cout << "BBFG : " << bbfg_results[i] << std::endl;
            std::cout << "GlRe : " << bbfgrec_results[i] << std::endl;
            adaptedMatricesForRecursiveGlynn[i].print_matrix();
            adaptedMatricesForNormalGlynn[i].print_matrix();
            exit(-1);
        }
    }

    if (printTimes){
        std::cout << "Sum of BBFG runtime: " << sumTimeBBFG.seconds() << std::endl;
        std::cout << "Sum of GlRe runtime: " << sumTimeBBFGREC.seconds() << std::endl;
        std::cout << "Speed ratio: " << sumTimeBBFGREC.seconds() / sumTimeBBFG.seconds() << std::endl;

    }
    std::cout << std::endl;

}
void testCase(std::vector<int> input, std::vector<int> output){
    testCase(input, output, 1);
}
void testCase(std::vector<int> inputOutput){
    testCase(inputOutput, inputOutput, 1);
}
void testCase(std::vector<int> inputOutput, int iterationNumber){
    testCase(inputOutput, inputOutput, iterationNumber);
}
void testCaseInitializerList(std::initializer_list<int> input, std::initializer_list<int> output, int iterationNumber){
    std::vector<int> inputVector = input;
    std::vector<int> outputVector = output;
    testCase(inputVector, outputVector, iterationNumber);
}
void testCaseInitializerList(std::initializer_list<int> input){
    testCaseInitializerList(input, input, 1);
}
void testCaseInitializerList(std::initializer_list<int> inputOutput, int iterationNumber){
    testCaseInitializerList(inputOutput, inputOutput, iterationNumber);
}
void testCaseInitializerList(std::initializer_list<int> input, std::initializer_list<int> output){
    testCaseInitializerList(input, output, 1);
}


int main() {

#ifdef __MPI__
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
#endif

    printf("\n\n********************************************************\n");
    printf("Test of permanents calculators speed\n");
    printf("********************************************************\n\n\n");

    // special cases
    testCaseInitializerList({1,1,1}, 100);
    testCaseInitializerList({2,3,0}, 100);
    testCaseInitializerList({2,3,0,1,0}, 1000);
    testCaseInitializerList({0,0,0}, 1);
    testCaseInitializerList({1,1,1,1,1,1,1,2}, 100);
    testCaseInitializerList({1,1,1,1,1,1,1,1}, 100);
    testCaseInitializerList({1,1,1,1,1,1,3,3}, 100);
    testCaseInitializerList({1,1,1,1,1,1,1,1,1,1,1,1}, 100);
    testCaseInitializerList({1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 100);
    testCaseInitializerList({1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 100);


    // special cases with different input and output states
    testCaseInitializerList({2,3,0},{1,1,3}, 100);
    testCaseInitializerList({1,1,3},{2,3,0}, 100);
    testCaseInitializerList({2,3,0,1,0},{1,1,0,2,2}, 100);
    testCaseInitializerList({1,1,0,2,2},{2,3,0,1,0}, 100);
    testCaseInitializerList({0,0,1,1,0,2,2},{0,1,1,3,0,1,0}, 100);
    testCaseInitializerList({0,0,0},{0,0,0}, 100);
    testCaseInitializerList({1,1,1,1,1,1,1,2},{2,1,1,1,1,1,1,1},100);

    testCaseInitializerList({1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1},100);
    testCaseInitializerList({1,1,1,0,0,0,0,0,0},100);
    testCaseInitializerList({1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,2},{0,0,0,0,0,0,0,0,0,1,1,4,1,1,1,0,0,0,0,2},100);
    

    constexpr int iterationNumber = 100;

    
    for (int matrix_dimension = 2; matrix_dimension < 16; matrix_dimension++){
        std::cout << "Dim: "<<matrix_dimension << std::endl;

        std::vector<int> inputState;
        for (int i = 0; i < matrix_dimension; i++){
            inputState.push_back(1);
        }
        testCase(inputState, inputState, iterationNumber);
    }

    std::cout << "All test cases passed." << std::endl;

#ifdef __MPI__
    // Finalize the MPI environment.
    MPI_Finalize();
#endif


    return 0;
    
}
