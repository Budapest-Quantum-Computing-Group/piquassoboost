#include <vector>

#include <random>
#include <chrono>
#include <string>

#include "GlynnPermanentCalculator.h"


#include "matrix32.h"
#include "matrix.h"

#include "dot.h"

#include "matrix_helper.hpp"
#include "constants_tests.h"
#include "Random_Unitary.h"
#include "tbb/tbb.h"

#ifdef __cplusplus
extern "C"
{
#endif

int calcPermanentGlynnDFE(const pic::Complex16* mtxHalf_data, const pic::Complex16* mtxHalf2_data, const double* renormalize_data, const uint64_t N, pic::Complex16& perm);
int calcPermanentGlynnSingleDFE(const pic::Complex16* mtx_data, const double* renormalize_data, const uint64_t N, const pic::Complex16 perm);
void calcPermanentGlynnMultiSRLDFE(const pic::Complex16* mtx_data0, const pic::Complex16* mtx_data1, const pic::Complex16* mtx_data2, const pic::Complex16* mtx_data3, const double* renormalize_data, const uint64_t N, const pic::Complex16 permCPU);

#ifdef __cplusplus
}
#endif


int getInputCount( int colIndex, int reducedDim, int dim ) {

		int ret = 0;

		if ( colIndex==0 ) {
			for (int idx=0; idx<reducedDim; idx=idx+2) {
//System.out.print("Col index: " + colIndex + ", reduced dim: " + reducedDim + ", return:" + (dim - idx - 1) + "\n");
				ret = ret + dim - idx - 1;
			}
			return ret;
		}
		else {
			for (int idx=colIndex; idx<dim; idx=idx+2) {
//System.out.print("Col index: " + colIndex + ",pivot col index: " + idx + ", dim: " + dim + "\n");
				ret = ret + getInputCount( colIndex-2, idx, dim );
			}

			return ret;
		}	
}


int main()
{

    size_t matrix_size = 10;

    // creating class to generate general random unitary
    pic::Random_Unitary ru = pic::Random_Unitary(matrix_size);
    // create general random unitary
    pic::matrix mtx = ru.Construct_Unitary_Matrix();



    //mtx.print_matrix();


    pic::Complex16* mtx_data = mtx.get_data();
    

    // calulate the maximal sum of the columns to normalize the matrix
    pic::matrix colSumMax( mtx.cols, 1);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(pic::Complex16) );
    for (int idx=0; idx<mtx.rows; idx++) {
        for( int jdx=0; jdx<mtx.cols; jdx++) {
            pic::Complex16 value1 = colSumMax[jdx] + mtx[ idx*mtx.stride + jdx];
            pic::Complex16 value2 = colSumMax[jdx] - mtx[ idx*mtx.stride + jdx];
            if ( std::abs( value1 ) < std::abs( value2 ) ) {
                colSumMax[jdx] = value2;
            }
            else {
                colSumMax[jdx] = value1;
            }

        }

    }



tbb::tick_count t0CPU = tbb::tick_count::now();
    pic::GlynnPermanentCalculator perm_calculator;
//    pic::Complex16 perm(0.0,0.0);
    pic::Complex16 perm = perm_calculator.calculate( mtx );
tbb::tick_count t1CPU = tbb::tick_count::now();
std::cout << "CPU permamanent: " << perm << std::endl;
std::cout << "elapsed time CPU: " << (t1CPU-t0CPU).seconds() << std::endl;

    // calculate the renormalization coefficients
    pic::matrix_base<double> renormalize_data(mtx.cols, 1);
    for (int jdx=0; jdx<mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(colSumMax[jdx]);
    }

    // renormalize the input matrix
    for (int idx=0; idx<mtx.rows; idx++) {
        for( int jdx=0; jdx<mtx.cols; jdx++) {
            mtx[ idx*mtx.stride + jdx] = mtx[ idx*mtx.stride + jdx]/renormalize_data[jdx];
        }

    }    
/*
    // renormalize the result according to the normalization of th einput matrix
    pic::Complex16 perm2 = perm_calculator.calculate( mtx );
    for (int jdx=0; jdx<mtx.cols; jdx++ ) {
        perm2 = perm2 * renormalize_data[jdx];
    }
    std::cout << "CPU permamanent normalize: " << perm2 << " " << std::abs((perm-perm2))/std::abs(perm)*100 << "\% difference" <<  std::endl;
*/

    // split matrices into four parts
    size_t cols0 = matrix_size/4;
    size_t cols1 = matrix_size/4;
    size_t cols2 = matrix_size/4;
    size_t cols3 = matrix_size - cols0 - cols1 - cols2;

    pic::matrix mtx0(mtx.rows, cols0);
    for (int idx=0; idx<mtx.rows; idx++) {
        for (int jdx=0; jdx<cols0; jdx++) {
            mtx0[idx*mtx0.stride+jdx] = mtx[idx*mtx.stride+jdx];
        }
    }


    pic::matrix mtx1(mtx.rows, cols1);
    for (int idx=0; idx<mtx.rows; idx++) {
        for (int jdx=0; jdx<cols1; jdx++) {
            mtx1[idx*mtx1.stride+jdx] = mtx[idx*mtx.stride+cols0+jdx];
        }
    }


    pic::matrix mtx2(mtx.rows, cols2);
    for (int idx=0; idx<mtx.rows; idx++) {
        for (int jdx=0; jdx<cols2; jdx++) {
            mtx2[idx*mtx2.stride+jdx] = mtx[idx*mtx.stride+cols0+cols1+jdx];
        }
    }


    pic::matrix mtx3(mtx.rows, cols3);
    for (int idx=0; idx<mtx.rows; idx++) {
        for (int jdx=0; jdx<cols3; jdx++) {
            mtx3[idx*mtx3.stride+jdx] = mtx[idx*mtx.stride+cols0+cols1+cols2+jdx];
        }
    }
/*
mtx.print_matrix();
mtx0.print_matrix();
mtx1.print_matrix();
mtx2.print_matrix();
mtx3.print_matrix();
*/


    pic::matrix mtxHalf(mtx.rows, mtx.cols/2);
    for (int idx=0; idx<mtx.rows; idx++) {
        for (int jdx=0; jdx<mtx.cols/2; jdx++) {
            mtxHalf[idx*mtxHalf.stride+jdx] = mtx[idx*mtx.stride+jdx];
        }
    }


    pic::matrix mtxHalf2(mtx.rows, mtx.cols/2);
    for (int idx=0; idx<mtx.rows; idx++) {
        for (int jdx=0; jdx<mtx.cols/2; jdx++) {
            mtxHalf2[idx*mtxHalf.stride+jdx] = mtx[idx*mtx.stride+mtx.cols/2+jdx];
        }
    }

/*
mtx.print_matrix();
mtxHalf.print_matrix();
mtxHalf2.print_matrix();
*/


    pic::Complex16 permDFE;

tbb::tick_count t0 = tbb::tick_count::now();
    calcPermanentGlynnDFE( mtxHalf.get_data(), mtxHalf2.get_data(), renormalize_data.get_data(), mtx.rows, permDFE);
    //calcPermanentGlynnSingleDFE( mtx.get_data(), renormalize_data.get_data(), mtx.rows, perm);
    //calcPermanentGlynnMultiSRLDFE( mtx0.get_data(), mtx1.get_data(), mtx2.get_data(), mtx3.get_data(), renormalize_data.get_data(), mtx.rows, perm);
tbb::tick_count t1 = tbb::tick_count::now();
std::cout << "elapsed time: " << (t1-t0).seconds() << std::endl;


    std::cout <<  "perm DFE: " << permDFE << std::endl;

    double relativeError = sqrt((permDFE.real() - perm.real())*(permDFE.real() - perm.real()) + (permDFE.imag() - perm.imag())*(permDFE.imag() - perm.imag()))/sqrt(perm.real()*perm.real() + perm.imag()*perm.imag()) * 100;

    std::cout <<   "relative error: " << relativeError << std::endl;

std::cout << "Test passed" << std::endl;
    return 0;
}

