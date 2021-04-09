//#define BOOST_TEST_MODULE determinant_testing

#include <random>
#include <chrono>

#include "PowerTraceHafnianUtilities.hpp"
//#include "TorontonianUtilities.hpp"

#include "matrix32.h"
#include "matrix.h"


extern "C" {


/// Definition of the LAPACKE_zgetri function from LAPACKE to calculate the LU decomposition of a matrix
int LAPACKE_zgetrf( int matrix_layout, int n, int m, pic::Complex16* a, int lda, int* ipiv );

/// Definition of the LAPACKE_zgetri function from LAPACKE to calculate the inverse of a matirx
int LAPACKE_zgetri( int matrix_layout, int n, pic::Complex16* a, int lda, const int* ipiv );

}


namespace pic{

 // the matrix c holds not just the polynomial coefficients, but also auxiliary
 // data. To retrieve the characteristic polynomial coeffients from the matrix c, use
 // this map for characteristic polynomial coefficient c_j:
 // if j = 0, c_0 -> 1
 // if j > 0, c_j -> c[(n - 1) * n + j - 1]
 
 // ||
 // VV
 // determinant = c_n -> c[(n-1) * n + n - 1] = c[(n-1)*(n+1)]
 
Complex16 determinant_by_hessenberg_labudde(matrix &AZ) {

    size_t n = AZ.rows;
    double scalar = n % 2 ? -1 : 1;

    // for small matrices only the traces are casted into quad precision
    if (AZ.rows <= 10) {

        transform_matrix_to_hessenberg<matrix, Complex16>(AZ);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix&& coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix, Complex16>(AZ, AZ.rows);

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
        matrix tau(N-1,1);
        LAPACKE_zgehrd(LAPACK_ROW_MAJOR, N, ILO, IHI, AZ.get_data(), LDA, tau.get_data() );

        matrix32 AZ32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ32[idx].real( AZ[idx].real() );
            AZ32[idx].imag( AZ[idx].imag() );
        }

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix32&& coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix32, Complex32>(AZ32, AZ.rows);

        //std::cout << "complex det: 2nd " << coeffs_labudde[(n-1)*(n+1)] << std::endl;
        Complex16 det = Complex16(coeffs_labudde[(n-1)*(n+1)].real(), coeffs_labudde[(n-1)*(n+1)].imag());
        return det * scalar;
    }
    else{
        // above a treshold matrix size all the calculations are done in quad precision
        // because of that we can not use the LAPACKE algorithm any more.

        // matrix size for which quad precision is necessary
        matrix32 AZ32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ32[idx].real( AZ[idx].real() );
            AZ32[idx].imag( AZ[idx].imag() );
        }

        transform_matrix_to_hessenberg<matrix32, Complex32>(AZ32);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix32 coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix32, Complex32>(AZ32, AZ.rows);

        //std::cout << "complex det: " << coeffs_labudde[(n-1)*(n+1)] << std::endl;
        Complex16 det = Complex16(coeffs_labudde[(n-1)*(n+1)].real(), coeffs_labudde[(n-1)*(n+1)].imag());
        return det * scalar;
    }

}


Complex16 determinant_byLU_decomposition( matrix& mtx ){
    matrix& Q = mtx;

    // calculate the inverse of matrix Q
    int* ipiv = (int*)scalable_aligned_malloc( Q.stride*sizeof(int), CACHELINE);
    LAPACKE_zgetrf( LAPACK_ROW_MAJOR, Q.rows, Q.cols, Q.get_data(), Q.stride, ipiv );

    //  calculate the determinant of Q
    Complex16 Qdet_cmplx(1.0,0.0);
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





matrix embedding(matrix& mtxIn, size_t toDim){
    size_t fromDim = mtxIn.rows;

    matrix mtxOut(toDim, toDim);
    for ( size_t row_idx = 0; row_idx < toDim; row_idx++ ){
        for (size_t col_idx = 0; col_idx < toDim; col_idx++){
            if (row_idx < fromDim && col_idx < fromDim){
                mtxOut[row_idx * toDim + col_idx] = mtxIn[row_idx * fromDim + col_idx];
            }else{
                if (row_idx == col_idx){
                    mtxOut[row_idx * toDim + col_idx] = Complex16(1, 0);
                }else{
                    mtxOut[row_idx * toDim + col_idx] = Complex16(0, 0);
                }
            }
        }

    }
    return mtxOut;

}

enum RandomMatrixType
{
    RANDOM,
    SYMMETRIC,
    SELFADJOINT
};

matrix getRandomMatrix(size_t n, enum RandomMatrixType type){
    matrix mtx(n, n);

    // initialize random generator as a standard normal distribution generator
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::normal_distribution<double> distribution(0.0, 1.0);

    if (type == RANDOM){
        // fill up matrix with fully random elements
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = 0; col_idx < n; col_idx++) {
                double randnum1 = distribution(generator);
                double randnum2 = distribution(generator);
                mtx[row_idx * n + col_idx] = pic::Complex16(randnum1, randnum2);
            }
        }
    }else if (type == SYMMETRIC){
        // fill up matrix with fully random elements symmetrically
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                double randnum1 = distribution(generator);
                double randnum2 = distribution(generator);
                mtx[row_idx * n + col_idx] = pic::Complex16(randnum1, randnum2);
                mtx[col_idx * n + row_idx] = pic::Complex16(randnum1, randnum2);
            }
        }
    }else if (type == SELFADJOINT){
        // hermitian case, selfadjoint matrix
        for (size_t row_idx = 0; row_idx < n; row_idx++) {
            for (size_t col_idx = row_idx; col_idx < n; col_idx++) {
                double randnum1 = distribution(generator);
                if (row_idx == col_idx){
                    mtx[row_idx * n + col_idx] = pic::Complex16(randnum1, 0);
                }else{
                    double randnum2 = distribution(generator);
                    mtx[row_idx * n + col_idx] = pic::Complex16(randnum1, randnum2);
                    mtx[col_idx * n + row_idx] = pic::Complex16(randnum1, -randnum2);
                }
            }
        }
    }

    return mtx;
}


} // pic namespace



int main(){
    size_t numberOfSamples = 5000;

    // iterating over dimensions
    for (int n = 2; n < 40; n++){
        long durationLong1 = 0;
        long durationLong2 = 0;

        // iterating over multiple matrices (number of matrices: numberOfSamples)
        for (int i = 0; i < numberOfSamples; i++){
            pic::matrix mtx1 = pic::getRandomMatrix(n, pic::RANDOM);
            pic::matrix mtx2 = embedding(mtx1, n);



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

            auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
            durationLong1 += duration1.count();
            durationLong2 += duration2.count();
        }
        auto averageDuration1 = 1.0 * durationLong1 / numberOfSamples;
        auto averageDuration2 = 1.0 * durationLong2 / numberOfSamples;
        
        std::cout << n << std::endl;
        std::cout << "duration1 : " << averageDuration1 << std::endl;
        std::cout << "duration2 : " << averageDuration2 << std::endl;
    }

    return 0;


}
