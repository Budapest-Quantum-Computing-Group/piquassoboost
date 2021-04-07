#define BOOST_TEST_MODULE determinant_testing

#include "PowerTraceHafnianUtilities.hpp"

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
 
double determinant_by_hessenberg_labudde(matrix &AZ, size_t pow_max) {

    size_t n = AZ.rows;

    // for small matrices only the traces are casted into quad precision
    if (AZ.rows <= 10) {

        transform_matrix_to_hessenberg<matrix, Complex16>(AZ);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix&& coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix, Complex16>(AZ, AZ.rows);

        return coeffs_labudde[(n-1)*(n+1)].real();

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
/*        matrix&& traces = powtrace_from_charpoly<matrix>(coeffs_labudde, pow_max);

        matrix_type traces32(traces.rows, traces.cols);
        for (size_t idx=0; idx<traces.size(); idx++) {
            traces32[idx].real( (long double)traces[idx].real() );
            traces32[idx].imag( (long double)traces[idx].imag() );
        }

        return traces32;*/
    }
    // The lapack function to calculate the Hessenberg transformation is more efficient for larger matrices, but for above a given cutoff quad precision is needed
    // for these matrices of moderate size, the coefficients of the characteristic polynomials are casted into quad precision and the traces are calculated in
    // quad precision
    else if ( (AZ.rows < 30 && (sizeof(Complex16) > sizeof(Complex16))) || (sizeof(Complex16) == sizeof(Complex16)) ) {
        //                                 always false
        //                                                                             always true!!                            

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

        return coeffs_labudde[(n-1)*(n+1)].real();

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
//        return powtrace_from_charpoly<matrix_type>(coeffs_labudde, pow_max);



    }
    else{
    // never happens !!!!
        // above a treshold matrix size all the calculations are done in quad precision

        // matrix size for which quad precision is necessary

        matrix AZ32( AZ.rows, AZ.cols);
        for (size_t idx=0; idx<AZ.size(); idx++) {
            AZ32[idx].real( AZ[idx].real() );
            AZ32[idx].imag( AZ[idx].imag() );
        }

        transform_matrix_to_hessenberg<matrix, Complex16>(AZ32);

        // calculate the coefficients of the characteristic polynomiam by LaBudde algorithm
        matrix coeffs_labudde = calc_characteristic_polynomial_coeffs<matrix, Complex16>(AZ32, AZ.rows);

        return coeffs_labudde[(n-1)*(n+1)].real();

        // calculate the power traces of the matrix AZ using LeVerrier recursion relation
//        return powtrace_from_charpoly<matrix_type>(coeffs_labudde, pow_max);

    }

}


double determinant_byLU_decomposition( matrix& mtx ){
    double Qdet;

//    if ( state.get_representation() != complex_amplitudes ) {
//        state.ConvertToComplexAmplitudes();
//    }

    // calculate Q matrix from Eq (3) in arXiv 2010.15595v3)
    matrix& Q = mtx;
    for (size_t idx=0; idx<Q.rows; idx++) {
        Q[idx*Q.stride+idx].real( Q[idx*Q.stride+idx].real() + 0.5 );
    }



    // calculate A matrix from Eq (4) in arXiv 2010.15595v3)
//    matrix Qinv = Q; //just to reuse the memory of Q for the inverse

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
    Qdet = Qdet_cmplx.real(); // the determinant of a symmetric matrix is real
    return Qdet;
}




} // pic namespace



int main(){
    return 0;


}
