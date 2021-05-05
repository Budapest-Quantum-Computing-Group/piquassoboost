 
#include <vector>

#include <random>
#include <chrono>
#include <string>

#include "TorontonianUtilities.hpp"
#include "Torontonian.h"

#include "matrix32.h"
#include "matrix.h"

#include "dot.h"

#include "constants_tests.h"


int test_cholesky_decomposition_block_based(){
    constexpr size_t dim = 130;

    pic::matrix mtx = pic::getRandomMatrix<pic::matrix, pic::Complex16>(dim, pic::POSITIVE_DEFINIT);
    pic::matrix mtx_copy = mtx.copy();

    //pic::matrix mtx2 = mtx.copy();
    //pic::matrix mtx3 = mtx.copy();

    pic::calc_cholesky_decomposition_block_based<pic::matrix, pic::Complex16>(mtx, 71);
    //pic::calc_cholesky_decomposition_block_based<pic::matrix, pic::Complex16>(mtx2, 4);

    //mtx2.print_matrix();

    /*for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = 0; jdx < idx + 1; jdx++){
            pic::Complex16 diff = mtx[idx * dim + jdx] - mtx2[idx * dim + jdx];
            if (std::abs(diff) > pic::epsilon){
                std::cout << "error: "<< idx<< ","<<jdx<< std::endl;    
            }
        }
    }*/


    //pic::calc_cholesky_decomposition<pic::matrix, pic::Complex16>(mtx3);

    //mtx.print_matrix();
    //mtx3.print_matrix();
    
    
    
    //std::cout << "Matrix decomposed by block cholesky:" << std::endl;
    //mtx.print_matrix();

    // rewrite upper triangular element to zero:
    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = idx + 1; jdx < dim; jdx++){
            mtx[idx * dim + jdx] = pic::Complex16(0.0);
        }
    }
    std::cout <<"mtx:"<<std::endl;
    mtx.print_matrix();

    pic::matrix mtx_adjoint(dim, dim);
    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = 0; jdx < dim; jdx++){
            pic::Complex16 &elem = mtx[jdx * dim + idx];
            mtx_adjoint[idx * dim + jdx] = pic::Complex16(elem.real(), -elem.imag());
        }
    }
    std::cout <<"mtx_adjoint:"<<std::endl;
    mtx_adjoint.print_matrix();

    //std::cout << "Matrix decomposed by old cholesky:" << std::endl;
    //mtx2.print_matrix();

    //std::cout << "Adjoint Matrix decomposed by block cholesky:" << std::endl;
    //mtx_adjoint.print_matrix();

    pic::matrix product = dot(mtx, mtx_adjoint);

    std::cout << "Stored original matrix:" << std::endl;
    mtx_copy.print_matrix();
    
    std::cout << "Product matrix:" << std::endl;
    product.print_matrix();
    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = 0; jdx < dim; jdx++){
            pic::Complex16 diff = product[idx * product.stride + jdx] - mtx_copy[idx * mtx_copy.stride + jdx];
            assert(std::abs(diff) < pic::epsilon);
            if (std::abs(diff) > pic::epsilon){
                std::cout << "Error " << idx << "," << jdx <<" diff: " <<diff<<std::endl;
                return 1;
            }
        }
    }

    return 0;
}


int test_cholesky_decomposition_algorithms(){
    constexpr int startDim = 20;
    constexpr int endDim = 101;
    constexpr int numberOfSamples = 100;
    
    // size of blocks
    std::vector<int> size_of_blocks;
    size_of_blocks.push_back(80);
    size_of_blocks.push_back(16);
    size_of_blocks.push_back(32);
    size_of_blocks.push_back(36);
    size_of_blocks.push_back(40);
    size_of_blocks.push_back(24);
    size_of_blocks.push_back(44);
    size_of_blocks.push_back(64);
    size_of_blocks.push_back(60);
    size_of_blocks.push_back(52);

    // vector for storing dfurations
    std::vector<long> durations;
    for (int i = 0; i < size_of_blocks.size(); i++){
        durations.push_back(0);
    }
    //std::cout << "size_of_blocks.size() = "<< size_of_blocks.size()<<std::endl;

    for (int dim = startDim; dim < endDim; dim++){
        for (int i = 0; i < size_of_blocks.size(); i++){
            durations[i] = 0;
        }
        long duration_basic = 0;
        long duration_basic2 = 0;

        //for (int kk = 0; kk < size_of_blocks.size(); kk++){
        //    std::cout << "duration: " << durations[kk] << " sizeofblock: "<<size_of_blocks[kk]<<std::endl;
        //}

        for (int i = 0; i < numberOfSamples; i++){
            //std::cout << dim << " / " << i << std::endl;

            pic::matrix mtx0 = pic::getRandomMatrix<pic::matrix, pic::Complex16>(dim, pic::POSITIVE_DEFINIT);
            pic::matrix mtx2 = mtx0.copy();
            pic::matrix mtx22 = mtx0.copy();

            //std::cout << "mtx:" << std::endl;
            //mtx2.print_matrix();


            auto start = std::chrono::high_resolution_clock::now();
            pic::calc_cholesky_decomposition<pic::matrix, pic::Complex16>(mtx2);
            auto stop = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            duration_basic += duration.count();

            for (int k = 0; k < size_of_blocks.size(); k++){

                pic::matrix mtx00 = mtx0.copy();
                
                size_t size_of_block = size_of_blocks[k];

                auto start = std::chrono::high_resolution_clock::now();
                pic::calc_cholesky_decomposition_block_based<pic::matrix, pic::Complex16>(mtx00, size_of_block);
                auto stop = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                durations[k] += duration.count();
            }


            auto start2 = std::chrono::high_resolution_clock::now();
            pic::calc_cholesky_decomposition<pic::matrix, pic::Complex16>(mtx22);
            auto stop2 = std::chrono::high_resolution_clock::now();

            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
            duration_basic2 += duration2.count();


                   /* 
            for (size_t idx = 0; idx < dim; idx++){
                for (size_t jdx = 0; jdx < idx + 1; jdx++){
                    pic::Complex16 diff = mtx0[idx * mtx0.stride + jdx] - mtx1[idx * mtx1.stride + jdx];
                    assert(std::abs(diff) < pic::epsilon);
                    if (std::abs(diff) > pic::epsilon){
                        std::cout << "Error " << dim << "/" << i << " : " << idx << "," << jdx << std::endl;
                        mtx0.print_matrix();
                        mtx1.print_matrix();
                        return 1;
                    }
                    diff = mtx1[idx * mtx1.stride + jdx] - mtx2[idx * mtx2.stride + jdx];
                    assert(std::abs(diff) < pic::epsilon);
                    if (std::abs(diff) > pic::epsilon){
                        std::cout << "Error " << dim << "/" << i << " : " << idx << "," << jdx << std::endl;
                        mtx0.print_matrix();
                        mtx1.print_matrix();
                        mtx2.print_matrix();
                        return 1;
                    }
                }
            }*/


        }
        std::cout << "Dimension: " << dim << std::endl;
        for (int i = 0; i < size_of_blocks.size(); i++){
            durations[i] /= numberOfSamples;
            std::cout << "mtx time for block number " << size_of_blocks[i] << " : " << durations[i] << std::endl;
        }

        duration_basic /= numberOfSamples;

        std::cout << "basic: " << duration_basic << std::endl;
        duration_basic2 /= numberOfSamples;

        std::cout << "basic: " << duration_basic2 << std::endl;
    }


    return 0;
}

int test_calc_torontonian(){
    constexpr size_t dim = 6;

    pic::matrix mtx = pic::getRandomMatrix<pic::matrix, pic::Complex16>(dim, pic::POSITIVE_DEFINIT);

/*
        [1j, 2, 3j, 4],
        [2, 6, 7, 8j],
        [3j, 7, 3, 4],
        [4, 8j, 4, 8j],
*/
/*
    mtx[0] = pic::Complex16(0.0, 1.0);
    mtx[1] = pic::Complex16(2.0, 0.0);
    mtx[2] = pic::Complex16(0.0, 3.0);
    mtx[3] = pic::Complex16(4.0, 0.0);

    mtx[4] = pic::Complex16(2.0, 0.0);
    mtx[5] = pic::Complex16(6.0, 0.0);
    mtx[6] = pic::Complex16(7.0, 0.0);
    mtx[7] = pic::Complex16(0.0, 8.0);

    mtx[8]  = pic::Complex16(0.0, 3.0);
    mtx[9]  = pic::Complex16(7.0, 0.0);
    mtx[10] = pic::Complex16(3.0, 0.0);
    mtx[11] = pic::Complex16(4.0, 0.0);

    mtx[12] = pic::Complex16(4.0, 0.0);
    mtx[13] = pic::Complex16(0.0, 8.0);
    mtx[14] = pic::Complex16(4.0, 0.0);
    mtx[15] = pic::Complex16(0.0, 8.0);
*/    
    pic::Torontonian torontonian_calculator(mtx);

    double result = torontonian_calculator.calculate();

    std::cout << "["<<std::endl;
    for (int i = 0; i < dim; i++){
        std::cout<<"[";
        for (int j = 0; j < dim; j++){
            std::cout<<mtx[i*dim+j].real();
            std::cout<<" + ";
            std::cout<<mtx[i*dim+j].imag();
            std::cout<< "j";
            if (j != dim - 1){
                std::cout <<", ";
            }
        }
        if (i != dim-1)
            std::cout<<"],"<< std::endl;
    }
    std::cout << "]"<< std::endl<< "]"<< std::endl;

    std::cout << result << std::endl;


    return 0;
}

int main(){
    test_cholesky_decomposition_block_based();
    test_cholesky_decomposition_algorithms();
    // test_calc_torontonian();
}
