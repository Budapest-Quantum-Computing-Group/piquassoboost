 
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


int test_cholesky_decomposition_inverse(){
    constexpr size_t dim = 6;

    pic::matrix mtx = pic::getRandomMatrix<pic::matrix, pic::Complex16>(dim, pic::LOWER_TRIANGULAR);
    pic::matrix mtx_adjoint(dim, dim);
    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = 0; jdx < dim; jdx++){
            pic::Complex16 &value = mtx[jdx * mtx.stride + idx];
            mtx_adjoint[idx * mtx_adjoint.stride + jdx] = pic::Complex16(value.real(), -value.imag());
        }
    }

    pic::matrix mtx_inverse = pic::calc_inverse_of_lower_triangular_matrix_adjoint<pic::matrix, pic::Complex16>(mtx);
    //mtx.print_matrix();
    //mtx_adjoint.print_matrix();
    //mtx_inverse.print_matrix();

    /*
    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = 0; jdx < dim; jdx++){
            std::cout<< idx<<","<<jdx<<": ";
            for (size_t k = 0; k < dim; k++){
                std::cout<< idx<<","<<k<<" * "<<k<<","<<jdx<<" + ";
            }
            std::cout<<std::endl;
            pic::Complex16 sum(0.0, 0.0);
            std::cout<< idx<<","<<jdx<<": ";
            for (size_t k = 0; k < dim; k++){
                sum += mtx_adjoint[idx*dim + k] * mtx_inverse[k*dim + jdx];
                std::cout<< mtx_adjoint[idx*dim + k] <<" * "<<mtx_inverse[k*dim + jdx]<<" + ";
            }

            std::cout<<"= "<<sum <<std::endl;
        }
    }*/


    pic::matrix product = pic::dot(mtx_adjoint, mtx_inverse);
    //product.print_matrix();

    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = 0; jdx < dim; jdx++){
            if (idx == jdx){
                pic::Complex16 diff = pic::Complex16(1.0, 0.0) - product[idx * product.stride + jdx];
                assert(diff < pic::epsilon);
            }else{
                pic::Complex16 diff = pic::Complex16(0.0, 0.0) - product[idx * product.stride + jdx];
                assert(diff < pic::epsilon);
            }
        }
    }
    return 0;
}

int test_cholesky_decomposition_block_based(){
    constexpr size_t dim = 6;

    pic::matrix mtx = pic::getRandomMatrix<pic::matrix, pic::Complex16>(dim, pic::POSITIVE_DEFINIT);
    pic::matrix mtx_copy = mtx.copy();

    pic::matrix mtx2 = mtx.copy();

    pic::calc_cholesky_decomposition_block_based<pic::matrix, pic::Complex16>(mtx, 3);
    pic::calc_cholesky_decomposition_block_based<pic::matrix, pic::Complex16>(mtx2, dim);
    
    
    
    //std::cout << "Matrix decomposed by block cholesky:" << std::endl;
    //mtx.print_matrix();

    // rewrite upper triangular element to zero:
    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = idx + 1; jdx < dim; jdx++){
            mtx[idx * dim + jdx] = 0;
        }
    }

    pic::matrix mtx_adjoint(dim, dim);
    // rewrite upper triangular element to zero:
    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = 0; jdx < dim; jdx++){
            pic::Complex16 &elem = mtx[jdx * dim + idx];
            mtx_adjoint[idx * dim + jdx] = pic::Complex16(elem.real(), -elem.imag());
        }
    }

    //std::cout << "Matrix decomposed by old cholesky:" << std::endl;
    //mtx2.print_matrix();

    //std::cout << "Adjoint Matrix decomposed by block cholesky:" << std::endl;
    //mtx_adjoint.print_matrix();

    pic::matrix product = dot(mtx, mtx_adjoint);

    //std::cout << "Stored original matrix:" << std::endl;
    //mtx_copy.print_matrix();
    
    //std::cout << "Product matrix:" << std::endl;
    //product.print_matrix();
    for (size_t idx = 0; idx < dim; idx++){
        for (size_t jdx = 0; jdx < dim; jdx++){
            pic::Complex16 diff = product[idx * product.stride + jdx] - mtx_copy[idx * mtx_copy.stride + jdx];
            assert(diff < pic::epsilon);
        }
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
    test_cholesky_decomposition_inverse();
    test_cholesky_decomposition_block_based();
    // test_calc_torontonian();
}
