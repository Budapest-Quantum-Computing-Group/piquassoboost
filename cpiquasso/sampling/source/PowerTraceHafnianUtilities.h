#ifndef PowerTraceHafnianUtilities_H
#define PowerTraceHafnianUtilities_H

#ifndef LAPACK_ROW_MAJOR
#define LAPACK_ROW_MAJOR 101
#endif
#ifndef LAPACK_COL_MAJOR
#define LAPACK_COL_MAJOR 102
#endif


#include "matrix.h"
#include "matrix32.h"



extern "C" {

#define LAPACK_ROW_MAJOR               101

/// Definition of the LAPACKE_zgehrd function from LAPACKE to calculate the upper triangle hessenberg transformation of a matrix
int LAPACKE_zgehrd( int matrix_layout, int n, int ilo, int ihi, pic::Complex16* a, int lda, pic::Complex16* tau );


int LAPACKE_zpotrf( int matrix_layout, char UPLO, int n, pic::Complex16* mtx, int lda );
}


namespace pic {




} // PIC

#endif
