#ifndef PowerTraceHafnianUtilities_H
#define PowerTraceHafnianUtilities_H

#include "matrix.h"


extern "C" {

#define LAPACK_ROW_MAJOR               101

/// Definition of the LAPACKE_zgehrd function from LAPACKE to calculate the upper triangle hessenberg transformation of a matrix
int LAPACKE_zgehrd( int matrix_layout, int n, int ilo, int ihi, pic::Complex16* a, int lda, pic::Complex16* tau );

}


namespace pic {




} // PIC

#endif
