#include <vector>

#include <random>
#include <chrono>
#include <string>

#include "TorontonianUtilities.hpp"
#include "Torontonian.h"
#include "TorontonianRecursive.h"


#include "matrix32.h"
#include "matrix.h"

#include "dot.h"

#include "matrix_helper.hpp"
#include "constants_tests.h"

#ifdef __cplusplus
extern "C"
{
#endif

int calcTorontonianDFE(const double* mtx_data, const uint64_t N);

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


    constexpr size_t dim = 10;

	// create random matrix to calculate the torontonian
    pic::matrix_real mtx = pic::get_random_density_matrix_real<pic::matrix_real, double>(dim);

//mtx.print_matrix();
	pic::TorontonianRecursive recursive_torontonian_calculator(mtx);
    double result_recursive_extended = recursive_torontonian_calculator.calculate(true);




    // Calculating B := 1 - mtx
    pic::matrix_real B(dim, dim);
    for (size_t idx = 0; idx < dim; idx++) {
        for (size_t jdx = 0; jdx < dim; jdx++) {
            B[idx * dim + jdx] = -1.0 * mtx[idx * mtx.stride + jdx];
        }
        B[idx * dim + idx] += 1.0;
    }
	mtx = B;



    // convert the input matrix from a1, a2, ... a_N, a_1^*, a_2^* ... a_N^* format to
    // a_1^*,a_1^*,a_2,a_2^*, ... a_N,a_N^* format

    size_t num_of_modes = dim/2;
    pic::matrix_real mtx_reordered(dim, dim);
    for (size_t idx=0; idx<num_of_modes; idx++) {
        for (size_t jdx=0; jdx<num_of_modes; jdx++) {
            mtx_reordered[2*idx*mtx_reordered.stride + 2*jdx] = mtx[idx*mtx.stride + jdx];
            mtx_reordered[2*idx*mtx_reordered.stride + 2*jdx+1] = mtx[idx*mtx.stride + jdx + num_of_modes];
            mtx_reordered[(2*idx+1)*mtx_reordered.stride + 2*jdx] = mtx[(idx+num_of_modes)*mtx.stride + jdx];
            mtx_reordered[(2*idx+1)*mtx_reordered.stride + 2*jdx+1] = mtx[(idx+num_of_modes)*mtx.stride + jdx + num_of_modes];
        }
    }

	mtx = mtx_reordered;

// calculate the cholesky decomposition of mtx
    pic::matrix_real L = mtx.copy();
	long double det = 0.0;
    pic::calc_determinant_cholesky_decomposition<pic::matrix_real, double, long double>(L, det);


   
L.print_matrix();
    mtx.print_matrix();


    //pic::matrix C = dot(L2, L3);
    //C.print_matrix();

    calcTorontonianDFE( mtx.get_data(), mtx.rows);
    return 0;
}

