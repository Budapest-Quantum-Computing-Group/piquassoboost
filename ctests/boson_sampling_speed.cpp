// The aim of this script is to compare runtimes
// of boson sampling algorithm in
// CGeneralizedCliffordsSimulationStrategy.


#include "CGeneralizedCliffordsSimulationStrategy.h"
#include "GlynnPermanentCalculatorDFE.h"
#include "matrix_helper.hpp"

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
    normalMatrices.reserve(iterationNumber);
    for (int i = 0; i < iterationNumber; i++){
        normalMatrices.push_back(
            pic::getRandomComplexMatrix<pic::matrix, pic::Complex16>(inputLength, pic::RANDOM)
        );
        //normalMatrices[i].print_matrix();
    }
    pic::PicState_int64 inputState(input.size());
    pic::PicState_int64 outputState(output.size());
    for (size_t i = 0; i < input.size(); i++){
        inputState[i] = input[i];
    }
    for (size_t i = 0; i < output.size(); i++){
        outputState[i] = output[i];
    }
    
    tbb::tick_count timeNow = tbb::tick_count::now();
    auto timeSum = (timeNow - timeNow).seconds();

    for (int i = 0; i < iterationNumber; i++){
        pic::matrix &mtx = normalMatrices[i];

        tbb::tick_count t0 = tbb::tick_count::now();
        pic::calculate_outputs_probability(mtx, inputState, outputState);
        tbb::tick_count t1 = tbb::tick_count::now();

        timeSum += (t1 - t0).seconds();
    }
    
    std::cout << "Average runtime: " << timeSum / iterationNumber << std::endl;
    std::cout << "Overall runtime: " << timeSum << std::endl;
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

    printf("\n\n********************************************************\n");
    printf("Test of boson sampling probability calculation speed\n");
    printf("********************************************************\n\n\n");

    // special cases
    //testCaseInitializerList({1,1,1}, 100);
    //testCaseInitializerList({2,3,0}, 100);
    //testCaseInitializerList({2,3,0,1,0}, 1000);
    //testCaseInitializerList({0,0,0}, 1);
    //testCaseInitializerList({1,1,1,1,1,1,1,2}, 100);
    //testCaseInitializerList({1,1,1,1,1,1,1,1}, 100);
    //testCaseInitializerList({1,1,1,1,1,1,3,3}, 100);
    //testCaseInitializerList({1,1,1,1,1,1,1,1,1,1,1,1}, 100);
    //testCaseInitializerList({1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 100);
    //testCaseInitializerList({1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 100);


    // special cases with different input and output states
    //testCaseInitializerList({2,3,0},{1,1,3}, 100);
    //testCaseInitializerList({1,1,3},{2,3,0}, 100);
    //testCaseInitializerList({2,3,0,1,0},{1,1,0,2,2}, 100);
    //testCaseInitializerList({1,1,0,2,2},{2,3,0,1,0}, 100);
    //testCaseInitializerList({0,0,1,1,0,2,2},{0,1,1,3,0,1,0}, 100);
    //testCaseInitializerList({0,0,0},{0,0,0}, 100);
    //testCaseInitializerList({1,1,1,1,1,1,1,2},{2,1,1,1,1,1,1,1},100);

    //testCaseInitializerList({1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1},100);
    //testCaseInitializerList({1,1,1,0,0,0,0,0,0},100);

    initialize_DFE();

    testCaseInitializerList(
        {1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,2,1,5,1,1,1,1,1,1,1,1,1,1,1,1},
        {0,0,0,0,0,0,0,0,0,1,1,4,1,1,1,0,0,0,0,2,1,5,1,1,1,1,1,1,1,1,1,1,1,1},
        100
    );
    

    constexpr int iterationNumber = 100;

    for (int matrix_dimension = 20; matrix_dimension < 30; matrix_dimension++){
        std::cout << "Dim: "<<matrix_dimension << std::endl;

        std::vector<int> inputState;
        for (int i = 0; i < matrix_dimension; i++){
            inputState.push_back(1);
        }
        testCase(inputState, inputState, iterationNumber);
    }
    
    releive_DFE();

    std::cout << "All test cases passed." << std::endl;
    return 0;
    
}
