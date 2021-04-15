//#define BOOST_TEST_MODULE determinant_testing

#include <random>
#include <chrono>

//#include "PowerTraceHafnianUtilities.hpp"
#include "TorontonianUtilities.hpp"

#include "matrix32.h"
#include "matrix.h"

#include "constants_tests.h"


extern "C" {


/// Definition of the LAPACKE_zgetri function from LAPACKE to calculate the LU decomposition of a matrix
int LAPACKE_zgetrf( int matrix_layout, int n, int m, pic::Complex16* a, int lda, int* ipiv );

/// Definition of the LAPACKE_zgetri function from LAPACKE to calculate the inverse of a matirx
int LAPACKE_zgetri( int matrix_layout, int n, pic::Complex16* a, int lda, const int* ipiv );

}



 // the matrix c holds not just the polynomial coefficients, but also auxiliary
 // data. To retrieve the characteristic polynomial coeffients from the matrix c, use
 // this map for characteristic polynomial coefficient c_j:
 // if j = 0, c_0 -> 1
 // if j > 0, c_j -> c[(n - 1) * n + j - 1]
 
 // ||
 // VV
 // determinant = c_n -> c[(n-1) * n + n - 1] = c[(n-1)*(n+1)]
 
// calculating determinant by applying hessenberg transformation and labudde algorithm afterwards
pic::Complex16 determinant_by_hessenberg_labudde(pic::matrix &AZ) {

    size_t n = AZ.rows;
    double scalar = n % 2 ? -1 : 1;

    // for small matrices only the traces are casted into quad precision
    if (AZ.rows <= 10) {

        pic::transform_matrix_to_hessenberg_TU<pic::matrix, pic::Complex16>(AZ);
        //pic::Complex16 det_hess =  pic::calc_determinant_of_selfadjoint_hessenberg_matrix<pic::matrix, pic::Complex16>(AZ);
        //std::cout << "Determinant by hessenberg det: " << det_hess << std::endl;
        
        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        pic::matrix&& coeffs_labudde = pic::calc_characteristic_polynomial_coeffs<pic::matrix, pic::Complex16>(AZ, AZ.rows);

        //std::cout << "complex det: " << coeffs_labudde[(n-1)*(n+1)] << std::endl;
        return coeffs_labudde[(n-1)*(n+1)] * scalar;
    }
    // The lapack function to calculate the Hessenberg transformation is more efficient for larger matrices, but for above a given cutoff quad precision is needed
    // for these matrices of moderate size, the coefficients of the characteristic polynomials are casted into quad precision and the traces are calculated in
    // quad precision
    else if ( (AZ.rows < 30)  ) {
        // transform the matrix mtx into an upper Hessenberg format by calling lapack function
        int N = AZ.rows;
        int ILO = 1;
        int IHI = N;
        int LDA = N;
        pic::matrix tau(N-1,1);
        LAPACKE_zgehrd(LAPACK_ROW_MAJOR, N, ILO, IHI, AZ.get_data(), LDA, tau.get_data() );

        pic::matrix32 AZ32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ32[idx].real( AZ[idx].real() );
            AZ32[idx].imag( AZ[idx].imag() );
        }
        //pic::Complex32 det_hess =  pic::calc_determinant_of_selfadjoint_hessenberg_matrix<pic::matrix32, pic::Complex32>(AZ32);
        //std::cout << "Determinant by hessenberg det: " << det_hess << std::endl;
        
        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        pic::matrix32&& coeffs_labudde = pic::calc_characteristic_polynomial_coeffs<pic::matrix32, pic::Complex32>(AZ32, AZ.rows);

        //std::cout << "complex det: 2nd " << coeffs_labudde[(n-1)*(n+1)] << std::endl;
        pic::Complex16 det = pic::Complex16(coeffs_labudde[(n-1)*(n+1)].real(), coeffs_labudde[(n-1)*(n+1)].imag());
        return det * scalar;
    }
    else{
        // above a treshold matrix size all the calculations are done in quad precision
        // because of that we can not use the LAPACKE algorithm any more.

        // matrix size for which quad precision is necessary
        pic::matrix32 AZ32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ32[idx].real( AZ[idx].real() );
            AZ32[idx].imag( AZ[idx].imag() );
        }

        pic::transform_matrix_to_hessenberg_TU<pic::matrix32, pic::Complex32>(AZ32);
        //std::cout << "Determinant by hessenberg det: " << det_hess << std::endl;

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        pic::matrix32 coeffs_labudde = pic::calc_characteristic_polynomial_coeffs<pic::matrix32, pic::Complex32>(AZ32, AZ.rows);

        //std::cout << "complex det: " << coeffs_labudde[(n-1)*(n+1)] << std::endl;
        pic::Complex16 det = pic::Complex16(coeffs_labudde[(n-1)*(n+1)].real(), coeffs_labudde[(n-1)*(n+1)].imag());
        return det * scalar;     
    }

}

// calculating the determinant using the LU decomposition from PowerTraceHafnianUtilities (Lapacke)
pic::Complex16 determinant_byLU_decomposition( pic::matrix& mtx ){
    pic::matrix& Q = mtx;

    // calculate the inverse of matrix Q
    int* ipiv = (int*)scalable_aligned_malloc( Q.stride*sizeof(int), CACHELINE);
    LAPACKE_zgetrf( LAPACK_ROW_MAJOR, Q.rows, Q.cols, Q.get_data(), Q.stride, ipiv );

    //  calculate the determinant of Q
    pic::Complex16 Qdet_cmplx(1.0,0.0);
    for (size_t idx=0; idx<Q.rows; idx++) {
        if (ipiv[idx] != idx+1) {
            Qdet_cmplx = -Qdet_cmplx * Q[idx*Q.stride + idx];
        }
        else {
            Qdet_cmplx = Qdet_cmplx * Q[idx*Q.stride + idx];
        }

    }
    //std::cout << "complex det: " << Qdet_cmplx << std::endl;
    return Qdet_cmplx;
}




// embedding of the given matrix into a matrix which has higher or lower dimension
// higher dimension means embedding with identity matrix
// lower dimension means to return the corresponding submatrix
// returns a new copy of the matrix, it uses different memory.
pic::matrix embedding(pic::matrix& mtxIn, size_t toDim){
    size_t fromDim = mtxIn.rows;

    pic::matrix mtxOut(toDim, toDim);
    for ( size_t row_idx = 0; row_idx < toDim; row_idx++ ){
        for (size_t col_idx = 0; col_idx < toDim; col_idx++){
            if (row_idx < fromDim && col_idx < fromDim){
                mtxOut[row_idx * toDim + col_idx] = mtxIn[row_idx * fromDim + col_idx];
            }else{
                if (row_idx == col_idx){
                    mtxOut[row_idx * toDim + col_idx] = pic::Complex16(1, 0);
                }else{
                    mtxOut[row_idx * toDim + col_idx] = pic::Complex16(0, 0);
                }
            }
        }

    }
    return mtxOut;

}

namespace pic{

enum RandomMatrixType
{
    RANDOM,
    SYMMETRIC,
    SELFADJOINT
};

} // namespace pic

// returns a random matrix of the given type:
// RANDOM : fully random
// SYMMETRIC : random complex symmetric matrix
// SELFADJOINT : random selfadjoint (hermitian) matrix
template<class matrix_type, class complex_type>
matrix_type 
getRandomMatrix(size_t n, pic::RandomMatrixType type){
    matrix_type mtx(n, n);

    // initialize random generator as a standard normal distribution generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<long double> distribution(0.0, 1.0);

    if (type == pic::RANDOM){
        // fill up matrix with fully random elements
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = 0; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                long double randnum2 = distribution(generator);
                mtx[row_idx * n + col_idx] = complex_type(randnum1, randnum2);
            }
        }
    }else if (type == pic::SYMMETRIC){
        // fill up matrix with fully random elements symmetrically
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                long double randnum2 = distribution(generator);
                mtx[row_idx * n + col_idx] = complex_type(randnum1, randnum2);
                mtx[col_idx * n + row_idx] = complex_type(randnum1, randnum2);
            }
        }
    }else if (type == pic::SELFADJOINT){
        // hermitian case, selfadjoint matrix
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                long double randnum1 = distribution(generator);
                if (row_idx == col_idx){
                    mtx[row_idx * n + col_idx] = complex_type(randnum1, 0);
                }else{
                    long double randnum2 = distribution(generator);
                    mtx[row_idx * n + col_idx] = complex_type(randnum1, randnum2);
                    mtx[col_idx * n + row_idx] = complex_type(randnum1, -randnum2);
                }
            }
        }
    }

    return mtx;
}


// method which checks the hessenberg property with a tolerance (currently the matrix has to be selfadjoint as well)
template<class matrix_type>
bool
checkHessenbergProperty(matrix_type mtx){
    int n = mtx.rows;

    for ( int row_idx = 0; row_idx < n; row_idx++ ){
        for ( int col_idx = 0; col_idx < n; col_idx++ ){
            if ( std::abs(row_idx - col_idx) > 1){
                if ( std::abs(mtx[row_idx * n + col_idx]) > pic::epsilon ){
                    return false;
                }
            }
        }
    }
    return true;
}

int main(){
    size_t dimension = 26;

    // applying transformation on a selfadjoint matrix to hessenberg form
    /*
    pic::matrix mtx = getRandomMatrix<pic::matrix, pic::Complex16>(dimension, pic::SELFADJOINT);

    mtx.print_matrix();

    pic::matrix mtx_copy1 = embedding(mtx, dimension);
    pic::matrix mtx_copy2 = embedding(mtx, dimension);

    pic::Complex16 det_by_hessenberg_labudde_1 = determinant_by_hessenberg_labudde(mtx_copy1);
    pic::Complex16 det_by_LU_decomposition_1 = determinant_byLU_decomposition(mtx_copy2);

    pic::transform_matrix_to_hessenberg_TU<pic::matrix, pic::Complex16>(mtx);

    // check result
    mtx.print_matrix();
    pic::matrix mtx_copy3 = embedding(mtx, dimension);


    pic::Complex16 det_by_hessenberg_labudde_2 = determinant_by_hessenberg_labudde(mtx);
    pic::Complex16 det_by_LU_decomposition_2 = determinant_byLU_decomposition(mtx_copy3);

    std::cout << "det_by_hessenberg_labudde_1  " << det_by_hessenberg_labudde_1 << std::endl;
    std::cout << "det_by_LU_decomposition_1    " << det_by_LU_decomposition_1 << std::endl;
    std::cout << "det_by_hessenberg_labudde_2  " << det_by_hessenberg_labudde_2 << std::endl;
    std::cout << "det_by_LU_decomposition_2    " << det_by_LU_decomposition_2 << std::endl;


    std::cout << "Hessenberg matrix? " << checkHessenbergProperty<pic::matrix>(mtx) << std::endl;
    */

    size_t numberOfSamples = 2000;

    int startDim = 2;
    int endDim = 38; // dimensions until endDim - 1 !!!
    // iterating over dimensions
    for (int n = startDim; n < endDim; n++){
        std::cout << "dimension: " << n << std::endl;
        long durationLong0 = 0;
        long durationLong1 = 0;
        long durationLong2 = 0;

        // iterating over multiple matrices (number of matrices: numberOfSamples)
        for (int i = 0; i < numberOfSamples; i++){
            pic::matrix mtx1 = getRandomMatrix<pic::matrix, pic::Complex16>(n, pic::SELFADJOINT);
            pic::matrix mtx0 = embedding(mtx1, n);
            pic::matrix mtx2 = embedding(mtx1, n);


            auto start0 = std::chrono::high_resolution_clock::now();
            pic::Complex16 det_hessenberg_labudde_sym = pic::calc_determinant_hessenberg_labudde_symmetric<pic::matrix, pic::Complex16>(mtx0);
            auto stop0 = std::chrono::high_resolution_clock::now();

            auto start1 = std::chrono::high_resolution_clock::now();
            pic::Complex16 det_by_hessenberg_labudde = determinant_by_hessenberg_labudde(mtx1);
            auto stop1 = std::chrono::high_resolution_clock::now();
            

            auto start2 = std::chrono::high_resolution_clock::now();
            pic::Complex16 det_by_LU_decomposition = determinant_byLU_decomposition(mtx2);
            auto stop2 = std::chrono::high_resolution_clock::now();

            pic::Complex16 diff = det_by_hessenberg_labudde - det_by_LU_decomposition;
            if (std::abs(diff) > std::abs((det_by_hessenberg_labudde + det_by_LU_decomposition) / 2. ) / 1000000000){
                std::cout << "ERR: " << n << " " << i << " " << det_by_hessenberg_labudde << " " << det_by_LU_decomposition << " " << diff << std::endl;
            }

            diff = det_by_hessenberg_labudde - det_hessenberg_labudde_sym;
            if (std::abs(diff) > std::abs((det_by_hessenberg_labudde + det_hessenberg_labudde_sym) / 2. ) / 1000000000){
                std::cout << "ERR: " << n << " " << i << " " << det_by_hessenberg_labudde << " " << det_hessenberg_labudde_sym << " " << diff << std::endl;
            }

            diff = det_hessenberg_labudde_sym - det_by_LU_decomposition;
            if (std::abs(diff) > std::abs((det_hessenberg_labudde_sym + det_by_LU_decomposition) / 2. ) / 1000000000){
                std::cout << "ERR: " << n << " " << i << " " << det_hessenberg_labudde_sym << " " << det_by_LU_decomposition << " " << diff << std::endl;
            }
            //mtx0.print_matrix();
            //mtx1.print_matrix();
            //mtx2.print_matrix();

            auto duration0= std::chrono::duration_cast<std::chrono::microseconds>(stop0 - start0);
            auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
            durationLong0 += duration0.count();
            durationLong1 += duration1.count();
            durationLong2 += duration2.count();
        }
        auto averageDuration0 = 1.0 * durationLong0 / numberOfSamples;
        auto averageDuration1 = 1.0 * durationLong1 / numberOfSamples;
        auto averageDuration2 = 1.0 * durationLong2 / numberOfSamples;
        
        std::cout << n << std::endl;
        std::cout << "duration0 : " << averageDuration0 << std::endl;
        std::cout << "duration1 : " << averageDuration1 << std::endl;
        std::cout << "duration2 : " << averageDuration2 << std::endl;
    }

    return 0;


}
