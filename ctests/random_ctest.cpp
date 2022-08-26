/**
 * Copyright 2022 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#if 1

#include "matrix.h"
#include "matrix_real.h"
#include "matrix_helper.hpp"




extern "C" {

#ifndef LAPACK_ROW_MAJOR
#define LAPACK_ROW_MAJOR 101
#endif

int LAPACKE_zgesvd(int matrix_layout, char, char, int, int, pic::Complex16 *, int, double *, pic::Complex16 *, int, pic::Complex16 *, int, pic::Complex16 *, int, double*, int* );
}



int ZGESVD(
    char JOBU
/*
[in]	JOBU	
          JOBU is CHARACTER*1
          Specifies options for computing all or part of the matrix U:
          = 'A':  all M columns of U are returned in array U:
          = 'S':  the first min(m,n) columns of U (the left singular
                  vectors) are returned in the array U;
          = 'O':  the first min(m,n) columns of U (the left singular
                  vectors) are overwritten on the array A;
          = 'N':  no columns of U (no left singular vectors) are
                  computed.
*/,
    char JOBVT
/*
[in]	JOBVT	
          JOBVT is CHARACTER*1
          Specifies options for computing all or part of the matrix
          V**H:
          = 'A':  all N rows of V**H are returned in the array VT;
          = 'S':  the first min(m,n) rows of V**H (the right singular
                  vectors) are returned in the array VT;
          = 'O':  the first min(m,n) rows of V**H (the right singular
                  vectors) are overwritten on the array A;
          = 'N':  no rows of V**H (no right singular vectors) are
                  computed.

          JOBVT and JOBU cannot both be 'O'.
*/,
int M
/*
[in]	M	
          M is INTEGER
          The number of rows of the input matrix A.  M >= 0.
*/,
int N
/*
[in]	N	
          N is INTEGER
          The number of columns of the input matrix A.  N >= 0.
*/,
pic::Complex16* A
/*
[in,out]	A	
          A is COMPLEX*16 array, dimension (LDA,N)
          On entry, the M-by-N matrix A.
          On exit,
          if JOBU = 'O',  A is overwritten with the first min(m,n)
                          columns of U (the left singular vectors,
                          stored columnwise);
          if JOBVT = 'O', A is overwritten with the first min(m,n)
                          rows of V**H (the right singular vectors,
                          stored rowwise);
          if JOBU .ne. 'O' and JOBVT .ne. 'O', the contents of A
                          are destroyed.
*/,
int LDA
/*
[in]	LDA	
          LDA is INTEGER
          The leading dimension of the array A.  LDA >= max(1,M).
*/,
double *S
/*
[out]	S	
          S is DOUBLE PRECISION array, dimension (min(M,N))
          The singular values of A, sorted so that S(i) >= S(i+1).
*/,
pic::Complex16 *U
/*
[out]	U	
          U is COMPLEX*16 array, dimension (LDU,UCOL)
          (LDU,M) if JOBU = 'A' or (LDU,min(M,N)) if JOBU = 'S'.
          If JOBU = 'A', U contains the M-by-M unitary matrix U;
          if JOBU = 'S', U contains the first min(m,n) columns of U
          (the left singular vectors, stored columnwise);
          if JOBU = 'N' or 'O', U is not referenced.
*/,
int LDU
/*
[in]	LDU	
          LDU is INTEGER
          The leading dimension of the array U.  LDU >= 1; if
          JOBU = 'S' or 'A', LDU >= M.
*/,
pic::Complex16 *VT
/*
[out]	VT	
          VT is COMPLEX*16 array, dimension (LDVT,N)
          If JOBVT = 'A', VT contains the N-by-N unitary matrix
          V**H;
          if JOBVT = 'S', VT contains the first min(m,n) rows of
          V**H (the right singular vectors, stored rowwise);
          if JOBVT = 'N' or 'O', VT is not referenced.
*/,
int LDVT
/*
[in]	LDVT	
          LDVT is INTEGER
          The leading dimension of the array VT.  LDVT >= 1; if
          JOBVT = 'A', LDVT >= N; if JOBVT = 'S', LDVT >= min(M,N).
*/,
pic::Complex16 *WORK
/*
[out]	WORK	
          WORK is COMPLEX*16 array, dimension (MAX(1,LWORK))
          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
*/,
int LWORK
/*
[in]	LWORK	
          LWORK is INTEGER
          The dimension of the array WORK.
          LWORK >=  MAX(1,2*MIN(M,N)+MAX(M,N)).
          For good performance, LWORK should generally be larger.

          If LWORK = -1, then a workspace query is assumed; the routine
          only calculates the optimal size of the WORK array, returns
          this value as the first entry of the WORK array, and no error
          message related to LWORK is issued by XERBLA.
*/,
double *RWORK
/*
[out]	RWORK	
          RWORK is DOUBLE PRECISION array, dimension (5*min(M,N))
          On exit, if INFO > 0, RWORK(1:MIN(M,N)-1) contains the
          unconverged superdiagonal elements of an upper bidiagonal
          matrix B whose diagonal is in S (not necessarily sorted).
          B satisfies A = U * B * VT, so it has the same singular
          values as A, and singular vectors related by U and VT.
*/,
int *INFO
/*
[out]	INFO	
          INFO is INTEGER
          = 0:  successful exit.
          < 0:  if INFO = -i, the i-th argument had an illegal value.
          > 0:  if ZBDSQR did not converge, INFO specifies how many
                superdiagonals of an intermediate bidiagonal form B
                did not converge to zero. See the description of RWORK
                above for details.
*/
);

/**
 * @brief Unit test to calculate something
 */
void test_calc_something(){
    constexpr size_t dimension = 6;

    pic::matrix mtx =
        pic::getRandomComplexMatrix<pic::matrix, pic::Complex16>(
            dimension,
            pic::RandomMatrixType::POSITIVE_DEFINIT
        );
/*
    mtx = pic::matrix(dimension,dimension);
    for (int i = 0; i < dimension*dimension; i++){
        mtx[i] = 0;
    }
    for (int i = 0; i < dimension; i++){
        mtx[i*mtx.stride+i] = 1;
    }
*/

        char JOBU = 'A';
        /*
        [in]	JOBU	
                JOBU is CHARACTER*1
                Specifies options for computing all or part of the matrix U:
                = 'A':  all M columns of U are returned in array U:
                = 'S':  the first min(m,n) columns of U (the left singular
                        vectors) are returned in the array U;
                = 'O':  the first min(m,n) columns of U (the left singular
                        vectors) are overwritten on the array A;
                = 'N':  no columns of U (no left singular vectors) are
                        computed.
        */
        char JOBVT = 'A';
        /*
        [in]	JOBVT	
                JOBVT is CHARACTER*1
                Specifies options for computing all or part of the matrix
                V**H:
                = 'A':  all N rows of V**H are returned in the array VT;
                = 'S':  the first min(m,n) rows of V**H (the right singular
                        vectors) are returned in the array VT;
                = 'O':  the first min(m,n) rows of V**H (the right singular
                        vectors) are overwritten on the array A;
                = 'N':  no rows of V**H (no right singular vectors) are
                        computed.

                JOBVT and JOBU cannot both be 'O'.
        */
        int M = mtx.rows;
        /*
        [in]	M	
                M is INTEGER
                The number of rows of the input matrix A.  M >= 0.
        */
        int N = mtx.cols;
        /*
        [in]	N	
                N is INTEGER
                The number of columns of the input matrix A.  N >= 0.
        */
        pic::Complex16* A = mtx.get_data();
        /*
        [in,out]	A	
                A is COMPLEX*16 array, dimension (LDA,N)
                On entry, the M-by-N matrix A.
                On exit,
                if JOBU = 'O',  A is overwritten with the first min(m,n)
                                columns of U (the left singular vectors,
                                stored columnwise);
                if JOBVT = 'O', A is overwritten with the first min(m,n)
                                rows of V**H (the right singular vectors,
                                stored rowwise);
                if JOBU .ne. 'O' and JOBVT .ne. 'O', the contents of A
                                are destroyed.
        */
        int LDA = mtx.stride;
        /*
        [in]	LDA	
                LDA is INTEGER
                The leading dimension of the array A.  LDA >= max(1,M).
        */
        pic::matrix_real s_res = pic::matrix_real(1, std::min(mtx.rows, mtx.cols));
        double *S = s_res.get_data();
        /*
        [out]	S	
                S is DOUBLE PRECISION array, dimension (min(M,N))
                The singular values of A, sorted so that S(i) >= S(i+1).
        */
        pic::matrix u_res = pic::matrix(dimension, dimension);
        pic::Complex16 *U = u_res.get_data();
        /*
        [out]	U	
                U is COMPLEX*16 array, dimension (LDU,UCOL)
                (LDU,M) if JOBU = 'A' or (LDU,min(M,N)) if JOBU = 'S'.
                If JOBU = 'A', U contains the M-by-M unitary matrix U;
                if JOBU = 'S', U contains the first min(m,n) columns of U
                (the left singular vectors, stored columnwise);
                if JOBU = 'N' or 'O', U is not referenced.
        */
        int LDU = dimension;
        /*
        [in]	LDU	
                LDU is INTEGER
                The leading dimension of the array U.  LDU >= 1; if
                JOBU = 'S' or 'A', LDU >= M.
        */
        pic::matrix vt_res = pic::matrix(dimension, dimension);
        pic::Complex16 *VT = vt_res.get_data();
        /*
        [out]	VT	
                VT is COMPLEX*16 array, dimension (LDVT,N)
                If JOBVT = 'A', VT contains the N-by-N unitary matrix
                V**H;
                if JOBVT = 'S', VT contains the first min(m,n) rows of
                V**H (the right singular vectors, stored rowwise);
                if JOBVT = 'N' or 'O', VT is not referenced.
        */
        int LDVT = dimension;
        /*
        [in]	LDVT	
                LDVT is INTEGER
                The leading dimension of the array VT.  LDVT >= 1; if
                JOBVT = 'A', LDVT >= N; if JOBVT = 'S', LDVT >= min(M,N).
        */
        pic::matrix work_res = pic::matrix(3*dimension, 1);
        pic::Complex16 *WORK = work_res.get_data();
        /*
        [out]	WORK	
                WORK is COMPLEX*16 array, dimension (MAX(1,LWORK))
                On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
        */
        int LWORK = 3*dimension;
        /*
        [in]	LWORK	
                LWORK is INTEGER
                The dimension of the array WORK.
                LWORK >=  MAX(1,2*MIN(M,N)+MAX(M,N)).
                For good performance, LWORK should generally be larger.

                If LWORK = -1, then a workspace query is assumed; the routine
                only calculates the optimal size of the WORK array, returns
                this value as the first entry of the WORK array, and no error
                message related to LWORK is issued by XERBLA.
        */
        pic::matrix_real rwork_res = pic::matrix_real(1, 5*dimension);
        double *RWORK = rwork_res.get_data();
        /*
        [out]	RWORK	
                RWORK is DOUBLE PRECISION array, dimension (5*min(M,N))
                On exit, if INFO > 0, RWORK(1:MIN(M,N)-1) contains the
                unconverged superdiagonal elements of an upper bidiagonal
                matrix B whose diagonal is in S (not necessarily sorted).
                B satisfies A = U * B * VT, so it has the same singular
                values as A, and singular vectors related by U and VT.
        */
        int info = 0;
        int *INFO = &info;
        /*
        [out]	INFO	
                INFO is INTEGER
                = 0:  successful exit.
                < 0:  if INFO = -i, the i-th argument had an illegal value.
                > 0:  if ZBDSQR did not converge, INFO specifies how many
                        superdiagonals of an intermediate bidiagonal form B
                        did not converge to zero. See the description of RWORK
                        above for details.
        */
        double superb[dimension];

    std::cout << "A matrix:";
    std::cout << "[";
    for (int i = 0; i < dimension; i++){
        std::cout << "[";
        for (int j = 0; j < dimension; j++){
            std::cout << mtx[i*mtx.stride +j].real();
            std::cout << "+" << mtx[i*mtx.stride +j].imag()<<"j, ";

        }
        std::cout<< "],";

    }
    std::cout << "]";
    
    mtx.print_matrix();

    int result_lapacke = LAPACKE_zgesvd( LAPACK_ROW_MAJOR, JOBU, JOBVT, M, N, A,
                LDA, S, U, LDU, VT, LDVT,
                WORK, LWORK, RWORK, INFO );

    //std::cout << "matrix U:\n";
    //u_res.print_matrix();
    //std::cout << "matrix VT:\n";
    //vt_res.print_matrix();
    //std::cout << "matrix WORK:\n";
    //work_res.print_matrix();
    //std::cout << "matrix RWORK:\n";
    //rwork_res.print_matrix();
    std::cout << "matrix S:\n";
    s_res.print_matrix();

    pic::matrix sigma = pic::matrix(dimension, dimension);
    for (int i = 0; i < sigma.size(); i++)
        sigma[i] = 0;

    for (int i = 0; i < dimension; i++){
        sigma[i*sigma.stride+i] = s_res[i];
    }
    


    pic::matrix prod1 = pic::dot(u_res, sigma);
    pic::matrix prod2 = pic::dot(prod1, vt_res);


    std::cout << "prod:\n";
    prod2.print_matrix();
    //std::cout << "A matrix:";
    //mtx.print_matrix();

    std::cout << "lapacke finished with "<<*INFO<<" status\n";
    std::cout << "lapacke finished with "<<result_lapacke<<" status\n";
    
}



/**
 * @brief main test function collecting all the ctest functions.
 */
int main(){


    test_calc_something();


}



#endif


#if 0 > 1
#include <stdlib.h>
#include <stdio.h>

/* DGESVD prototype */
extern void dgesvd( char* jobu, char* jobvt, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* info );
/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, double* a, int lda );

/* Parameters */
#define M 6
#define N 5
#define LDA M
#define LDU M
#define LDVT N

/* Main program */
int main() {
        /* Locals */
        int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork;
        double wkopt;
        double* work;
        /* Local arrays */
        double s[N], u[LDU*M], vt[LDVT*N];
        double a[LDA*N] = {
            8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
            9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
            9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
            5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
            3.16,  7.98,  3.01,  5.80,  4.27, -5.31
        };
        /* Executable statements */
        printf( " DGESVD Example Program Results\n" );
        /* Query and allocate the optimal workspace */
        lwork = -1;
        dgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork,
         &info );
        lwork = (int)wkopt;
        work = (double*)malloc( lwork*sizeof(double) );
        /* Compute SVD */
        dgesvd( "All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
         &info );
        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm computing SVD failed to converge.\n" );
                exit( 1 );
        }
        /* Print singular values */
        print_matrix( "Singular values", 1, n, s, 1 );
        /* Print left singular vectors */
        print_matrix( "Left singular vectors (stored columnwise)", m, n, u, ldu );
        /* Print right singular vectors */
        print_matrix( "Right singular vectors (stored rowwise)", n, n, vt, ldvt );
        /* Free workspace */
        free( (void*)work );
        exit( 0 );
} /* End of DGESVD Example */

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
                printf( "\n" );
        }
}

int LAPACKE_zgetri( int matrix_layout, int n, pic::Complex16* a, int lda, const int* ipiv );

#endif



#if 0

/*******************************************************************************
*  Copyright (C) 2009-2015 Intel Corporation. All Rights Reserved.
*  The information and material ("Material") provided below is owned by Intel
*  Corporation or its suppliers or licensors, and title to such Material remains
*  with Intel Corporation or its suppliers or licensors. The Material contains
*  proprietary information of Intel or its suppliers and licensors. The Material
*  is protected by worldwide copyright laws and treaty provisions. No part of
*  the Material may be copied, reproduced, published, uploaded, posted,
*  transmitted, or distributed in any way without Intel's prior express written
*  permission. No license under any patent, copyright or other intellectual
*  property rights in the Material is granted to or conferred upon you, either
*  expressly, by implication, inducement, estoppel or otherwise. Any license
*  under such intellectual property rights must be express and approved by Intel
*  in writing.
*
********************************************************************************
*/
/*
   LAPACKE_zgesvd Example.
   =======================

   Program computes the singular value decomposition of a general
   rectangular complex matrix A:

   (  5.91, -5.69) (  7.09,  2.72) (  7.78, -4.06) ( -0.79, -7.21)
   ( -3.15, -4.08) ( -1.89,  3.27) (  4.57, -2.07) ( -3.88, -3.30)
   ( -4.89,  4.20) (  4.10, -6.70) (  3.28, -3.84) (  3.84,  1.19)

   Description.
   ============

   The routine computes the singular value decomposition (SVD) of a complex
   m-by-n matrix A, optionally computing the left and/or right singular
   vectors. The SVD is written as

   A = U*SIGMA*VH

   where SIGMA is an m-by-n matrix which is zero except for its min(m,n)
   diagonal elements, U is an m-by-m unitary matrix and VH (V conjugate
   transposed) is an n-by-n unitary matrix. The diagonal elements of SIGMA
   are the singular values of A; they are real and non-negative, and are
   returned in descending order. The first min(m, n) columns of U and V are
   the left and right singular vectors of A.

   Note that the routine returns VH, not V.

   Example Program Results.
   ========================

 LAPACKE_zgesvd (row-major, high-level) Example Program Results

 Singular values
  17.63  11.61   6.78

 Left singular vectors (stored columnwise)
 ( -0.86,  0.00) (  0.40,  0.00) (  0.32,  0.00)
 ( -0.35,  0.13) ( -0.24, -0.21) ( -0.63,  0.60)
 (  0.15,  0.32) (  0.61,  0.61) ( -0.36,  0.10)

 Right singular vectors (stored rowwise)
 ( -0.22,  0.51) ( -0.37, -0.32) ( -0.53,  0.11) (  0.15,  0.38)
 (  0.31,  0.31) (  0.09, -0.57) (  0.18, -0.39) (  0.38, -0.39)
 (  0.53,  0.24) (  0.49,  0.28) ( -0.47, -0.25) ( -0.15,  0.19)
*/
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

#define min(a,b) ((a)>(b)?(b):(a))

/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, pic::Complex16* a, int lda );
extern void print_rmatrix( char* desc, int m, int n, double* a, int lda );

/* Parameters */
#define M 3
#define N 4
#define LDA N
#define LDU M
#define LDVT N

#define MKL_INT int
#define MKL_Complex16 pic::Complex16


extern "C" {

#ifndef LAPACK_ROW_MAJOR
#define LAPACK_ROW_MAJOR 101
#endif

int LAPACKE_zgesvd(int matrix_layout, char*,char*,int*,int*,pic::Complex16 *,int*,double*, pic::Complex16 *,int*,pic::Complex16 *,int*, pic::Complex16 *,int*,double*,int* );
}

/* Main program */
int main() {
        /* Locals */
        int m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info;
        /* Local arrays */
        double s[M];
        double superb[min(M,N)-1];
        pic::Complex16 u[LDU*M], vt[LDVT*N];
        pic::Complex16 a[LDA*M] = {
           { 5.91, -5.69}, { 7.09,  2.72}, { 7.78, -4.06}, {-0.79, -7.21},
           {-3.15, -4.08}, {-1.89,  3.27}, { 4.57, -2.07}, {-3.88, -3.30},
           {-4.89,  4.20}, { 4.10, -6.70}, { 3.28, -3.84}, { 3.84,  1.19}
        };
        /* Executable statements */
        printf( "LAPACKE_zgesvd (row-major, high-level) Example Program Results\n" );
        /* Compute SVD */
        info = LAPACKE_zgesvd( LAPACK_ROW_MAJOR, 'A', 'A', m, n, a, lda, s,
         u, ldu, vt, ldvt, superb );
        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm computing SVD failed to converge.\n" );
                exit( 1 );
        }
        /* Print singular values */
        print_rmatrix( "Singular values", 1, m, s, 1 );
        /* Print left singular vectors */
        print_matrix( "Left singular vectors (stored columnwise)", m, m, u, ldu );
        /* Print right singular vectors */
        print_matrix( "Right singular vectors (stored rowwise)", m, n, vt, ldvt );
        exit( 0 );
} /* End of LAPACKE_zgesvd Example */

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, MKL_INT m, MKL_INT n, MKL_Complex16* a, MKL_INT lda ) {
        MKL_INT i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ )
                        printf( " (%6.2f,%6.2f)", a[i*lda+j].real, a[i*lda+j].imag );
                printf( "\n" );
        }
}

/* Auxiliary routine: printing a real matrix */
void print_rmatrix( char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda ) {
        MKL_INT i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i*lda+j] );
                printf( "\n" );
        }
}

#endif