//#define BOOST_TEST_MODULE determinant_testing

#include <random>
#include <chrono>
#include <string>

//#include "PowerTraceHafnianUtilities.hpp"
#include "TorontonianUtilities.hpp"

#include "matrix32.h"
#include "matrix.h"

#include "dot.h"

#include "constants_tests.h"


extern "C" {


/// Definition of the LAPACKE_zgetri function from LAPACKE to calculate the LU decomposition of a matrix
int LAPACKE_zgetrf( int matrix_layout, int n, int m, pic::Complex16* a, int lda, int* ipiv );

/// Definition of the LAPACKE_zgetri function from LAPACKE to calculate the inverse of a matirx
int LAPACKE_zgetri( int matrix_layout, int n, pic::Complex16* a, int lda, const int* ipiv );

}



// calculating determinant by applying hessenberg transformation and labudde algorithm
// Code was copied from PowerTraceHafnianUtilities.hpp
pic::Complex16 test_determinant_by_hessenberg_labudde(pic::matrix &AZ) {
    size_t n = AZ.rows;
    double scalar = n % 2 ? -1 : 1;

    // for small matrices only the traces are casted into quad precision
    if (AZ.rows <= 10) {

        //pic::transform_matrix_to_hessenberg_TU<pic::matrix, pic::Complex16>(AZ);
        //pic::Complex16 det_hess =  pic::calc_determinant_of_selfadjoint_hessenberg_matrix<pic::matrix, pic::Complex16>(AZ);
        //std::cout << "Determinant by hessenberg det: " << det_hess << std::endl;
        
        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        //pic::matrix&& coeffs_labudde = pic::calc_characteristic_polynomial_coeffs<pic::matrix, pic::Complex16>(AZ, AZ.rows);

        //std::cout << "complex det: " << coeffs_labudde[(n-1)*(n+1)] << std::endl;
        //return coeffs_labudde[(n-1)*(n+1)] * scalar;
        return pic::Complex16(1.0);
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
        //pic::matrix32&& coeffs_labudde = pic::calc_characteristic_polynomial_coeffs<pic::matrix32, pic::Complex32>(AZ32, AZ.rows);

        //std::cout << "complex det: 2nd " << coeffs_labudde[(n-1)*(n+1)] << std::endl;
        //pic::Complex16 det = pic::Complex16(coeffs_labudde[(n-1)*(n+1)].real(), coeffs_labudde[(n-1)*(n+1)].imag());
        //return det * scalar;
        return pic::Complex16(1.0);
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

        //pic::transform_matrix_to_hessenberg_TU<pic::matrix32, pic::Complex32>(AZ32);
        //std::cout << "Determinant by hessenberg det: " << det_hess << std::endl;

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        //pic::matrix32 coeffs_labudde = pic::calc_characteristic_polynomial_coeffs<pic::matrix32, pic::Complex32>(AZ32, AZ.rows);

        //std::cout << "complex det: " << coeffs_labudde[(n-1)*(n+1)] << std::endl;
        //pic::Complex16 det = pic::Complex16(coeffs_labudde[(n-1)*(n+1)].real(), coeffs_labudde[(n-1)*(n+1)].imag());
        //return det * scalar; 
        return pic::Complex16(1.0);    
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



// method which checks the hessenberg property with a tolerance (currently the matrix has to be selfadjoint as well)
template<class matrix_type>
bool
test_check_hessenberg_property(matrix_type mtx){
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




int cholesky_decomposition_LAPACKE(pic::matrix& mtx){
    char UPLO = 'L';
    int N = mtx.cols;
    int LDA = N;

    LAPACKE_zpotrf(LAPACK_ROW_MAJOR, UPLO, N, mtx.get_data(), LDA);
/*
    for(int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (i < j){
                mtx[i*mtx.stride+j] = pic::Complex16(0,0);
            }
        }
    }
*/
}

pic::Complex16 test_calc_determinant_cholesky_lapacke(pic::matrix& mtx){
    cholesky_decomposition_LAPACKE(mtx);

    pic::Complex16 det(1, 0);

    for (int i = 0; i < mtx.cols; i++){
        det *= mtx[i * mtx.stride + i];
    }
    return pic::mult_a_bconj(det, det);
}

pic::Complex16 test_calc_determinant_cholesky_ownalgo(pic::matrix& mtx){
    pic::calc_cholesky_decomposition<pic::matrix, pic::Complex16>(mtx);

    pic::Complex16 det(1, 0);

    for (int i = 0; i < mtx.cols; i++){
        det *= mtx[i * mtx.stride + i];
    }
    return pic::mult_a_bconj( det, det);
}

int test_cholesky_decomposition(){
    size_t dimension = 12;

    // Cholesky decomposition by LAPACKE:
    if (0){
        std::cout<<"Matrix to be transformed: "<<std::endl;
        pic::matrix mtx = pic::getRandomMatrix<pic::matrix, pic::Complex16>(dimension, pic::POSITIVE_DEFINIT);
        pic::matrix mtx_copy = mtx.copy();

        mtx.print_matrix();

        cholesky_decomposition_LAPACKE(mtx);
        std::cout<<"Cholesky form (L): "<<std::endl;
        mtx.print_matrix();

        pic::matrix cholesky_adjoint = pic::matrix_conjugate_traspose<pic::matrix>(mtx);
        pic::matrix product = pic::dot(mtx, cholesky_adjoint);

        std::cout<<"L * L^*: "<<std::endl;
        product.print_matrix();

        for (int i = 0; i < dimension; i++){
            for (int j = 0; j < dimension; j++){
                pic::Complex16 diff = mtx_copy[i * mtx_copy.stride + j] - product[i * product.stride + j];
                if (std::abs(diff) > pic::epsilon){
                    std::cout<<"Diff: "<<i<<","<<j<<std::endl;
                }
            }
        }
    }

    // Cholesky decomposition by https://www.geeksforgeeks.org/cholesky-decomposition-matrix-decomposition/:
    if (1){
        pic::matrix mtx = pic::getRandomMatrix<pic::matrix, pic::Complex16>(dimension, pic::POSITIVE_DEFINIT);
        pic::matrix mtx_copy = mtx.copy();
        mtx.print_matrix();

        pic::calc_cholesky_decomposition<pic::matrix, pic::Complex16>(mtx);
        pic::matrix cholesky = mtx;
        
        for(int i = 0; i < dimension; i++){
            for (int j = 0; j < dimension; j++){
                if (i < j){
                    cholesky[i*mtx.stride+j] = pic::Complex16(0,0);
                }
            }
        }
        cholesky.print_matrix();
        pic::matrix c1 = cholesky.copy();
        pic::matrix c2 = pic::matrix_conjugate_traspose<pic::matrix>(cholesky);

        
        pic::matrix product = pic::dot(c1, c2);

        std::cout<<"L * L^*: "<<std::endl;
        product.print_matrix();

        for (int i = 0; i < dimension; i++){
            for (int j = 0; j < dimension; j++){
                pic::Complex16 diff = mtx_copy[i * mtx_copy.stride + j] - product[i * product.stride + j];
                if (std::abs(diff) > pic::epsilon){
                    std::cout<<"Diff: "<<i<<","<<j<<std::endl;
                }
            }
        }

    }
    return 0;
}


int test_hessenberg_labudde_selfadjoint(){
    size_t dimension = 5;

    pic::matrix mtx = pic::getRandomMatrix<pic::matrix, pic::Complex16>(dimension, pic::SELFADJOINT);

    mtx.print_matrix();

    pic::matrix mtx_copy1 = embedding(mtx, dimension);
    pic::matrix mtx_copy2 = embedding(mtx, dimension);

    //pic::Complex16 det_by_hessenberg_labudde_1 = test_determinant_by_hessenberg_labudde(mtx_copy1);
    pic::Complex16 det_by_LU_decomposition_1 = determinant_byLU_decomposition(mtx_copy2);

    // applying transformation on a selfadjoint matrix to hessenberg form
    //pic::transform_matrix_to_hessenberg_TU<pic::matrix, pic::Complex16>(mtx);

    // check result
    mtx.print_matrix();
    pic::matrix mtx_copy3 = embedding(mtx, dimension);


    //pic::Complex16 det_by_hessenberg_labudde_2 = test_determinant_by_hessenberg_labudde(mtx);
    pic::Complex16 det_by_LU_decomposition_2 = determinant_byLU_decomposition(mtx_copy3);

    //std::cout << "det_by_hessenberg_labudde_1  " << det_by_hessenberg_labudde_1 << std::endl;
    std::cout << "det_by_LU_decomposition_1    " << det_by_LU_decomposition_1 << std::endl;
    //std::cout << "det_by_hessenberg_labudde_2  " << det_by_hessenberg_labudde_2 << std::endl;
    std::cout << "det_by_LU_decomposition_2    " << det_by_LU_decomposition_2 << std::endl;


    //std::cout << "Hessenberg matrix? " << test_check_hessenberg_property<pic::matrix>(mtx) << std::endl;
    return 0;
}

int test_runtimes_determinant_calculations(){
    size_t numberOfSamples = 100;

    int startDim = 20;
    int endDim = 100; // dimensions until endDim - 1 !!!


    constexpr size_t numberOfCalcTypes = 6;

    pic::Complex16 (*calcAlgos[numberOfCalcTypes])(pic::matrix&);
    std::string algoNames[numberOfCalcTypes];

    algoNames[0] = "test_determinant_by_hessenberg_labudde";
    calcAlgos[0] = NULL;// &test_determinant_by_hessenberg_labudde;

    algoNames[1] = "calc_determinant_hessenberg_labudde_symmetric";
    calcAlgos[1] = NULL;//&pic::calc_determinant_hessenberg_labudde_symmetric<pic::matrix, pic::Complex16>;

    algoNames[2] = "determinant_byLU_decomposition";
    calcAlgos[2] = &determinant_byLU_decomposition;

    algoNames[3] = "test_calc_determinant_cholesky_ownalgo";
    calcAlgos[3] = &test_calc_determinant_cholesky_ownalgo;

    algoNames[4] = "test_calc_determinant_cholesky_lapacke";
    calcAlgos[4] = &test_calc_determinant_cholesky_lapacke;

    algoNames[5] = "calc_determinant_cholesky_decomposition";
    calcAlgos[5] = &pic::calc_determinant_cholesky_decomposition<pic::matrix, pic::Complex16>;


    // iterating over dimensions
    for (int n = startDim; n < endDim; n++){

        std::cout << "dimension: " << n << std::endl;
        long durations[numberOfCalcTypes];
        for (size_t idx = 0; idx < numberOfCalcTypes; idx++){
            durations[idx] = 0;
        }

        // iterating over multiple matrices (number of matrices: numberOfSamples)
        for (int i = 0; i < numberOfSamples; i++){
            // array of the calculated determinants to be able to check whether they are equal or not
            pic::Complex16 determinants[numberOfCalcTypes];

            pic::matrix matrices[numberOfCalcTypes];
            matrices[0] = pic::getRandomMatrix<pic::matrix, pic::Complex16>(n, pic::POSITIVE_DEFINIT);
            for (size_t typeIdx = 1; typeIdx < numberOfCalcTypes; typeIdx++){
                matrices[typeIdx] = matrices[0].copy();
            }
            for (size_t idx = 0; idx < numberOfCalcTypes; idx++){
                if (calcAlgos[idx] != NULL){
                    auto start = std::chrono::high_resolution_clock::now();
                    determinants[idx] = calcAlgos[idx](matrices[idx]);
                    auto stop = std::chrono::high_resolution_clock::now();

                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    durations[idx] += duration.count();
                }
            }

            for (size_t idx = 1; idx < numberOfCalcTypes; idx++){
                pic::Complex16 diff = determinants[idx-1]- determinants[idx];
                pic::Complex16 tolerance = (determinants[idx-1] + determinants[idx]) / pic::Complex16(2,0) * pic::epsilon;
                if (calcAlgos[idx-1] != NULL && calcAlgos[idx] != NULL && std::abs(diff) > std::abs(tolerance)){
                    std::cout << "Error:" << std::endl;
                    std::cout << idx   << ": " << determinants[idx-1] << std::endl;
                    std::cout << idx-1 << ": " << determinants[idx]<<std::endl;
                }
            }
        }

        double averageDurations[numberOfCalcTypes];
        for (size_t idx = 0; idx < numberOfCalcTypes; idx++){
            averageDurations[idx] = 1.0 * durations[idx] / numberOfSamples;
        }

        for (size_t idx = 0; idx < numberOfCalcTypes; idx++){
            std::cout <<algoNames[idx] << ":\t" << averageDurations[idx]<< std::endl;
        }
    }
    return 0;
}

int test_determinant_is_same(){
    size_t n = 8;

    pic::matrix mtx0 = pic::getRandomMatrix<pic::matrix, pic::Complex16>(n, pic::POSITIVE_DEFINIT);
    pic::matrix mtx1 = embedding(mtx0, n);
    pic::matrix mtx2 = embedding(mtx0, n);
    pic::matrix mtx3 = embedding(mtx0, n);

    // this algorithm works on symmetric or selfadjoint matrices
    //pic::Complex16 det_hessenberg_labudde_sym = pic::calc_determinant_hessenberg_labudde_symmetric<pic::matrix, pic::Complex16>(mtx0);
    // this algorithm works on all matrices
    //pic::Complex16 det_by_hessenberg_labudde = test_determinant_by_hessenberg_labudde(mtx1);
    // this algorithm works on all matrices
    pic::Complex16 det_by_LU_decomposition = determinant_byLU_decomposition(mtx2);
    // this algorithm works on positive definite symmetric or selfadjoint matrices
    pic::Complex16 det_by_cholesky = pic::calc_determinant_cholesky_decomposition<pic::matrix, pic::Complex16>(mtx3);

    //std::cout<<"det_hessenberg_labudde_sym:  " << det_hessenberg_labudde_sym << std::endl;
    //std::cout<<"det_by_hessenberg_labudde:   " << det_by_hessenberg_labudde << std::endl;
    std::cout<<"det_by_LU_decomposition:     " << det_by_LU_decomposition << std::endl;
    std::cout<<"det_by_cholesky:             " << det_by_cholesky << std::endl;

    return 0;
}

int main(){


    test_cholesky_decomposition();
    //test_hessenberg_labudde_selfadjoint();
    //test_runtimes_determinant_calculations();
    //test_determinant_is_same();

    return 0;


}
