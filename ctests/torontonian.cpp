 
#include <vector>

#include <random>
#include <chrono>
#include <string>

#include "Torontonian.h"

#include "matrix32.h"
#include "matrix.h"

#include "dot.h"

#include "constants_tests.h"




int main(){
    constexpr size_t dim = 16;

    
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
