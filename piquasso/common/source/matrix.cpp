#include "matrix.h"
#include <cstring>
#include <iostream>
#include "tbb/tbb.h"

/// The namespace of the Picasso project
namespace pic {

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
matrix::matrix() : matrix_base<Complex16>() {

}

/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
matrix::matrix( Complex16* data_in, size_t rows_in, size_t cols_in) : matrix_base<Complex16>(data_in, rows_in, cols_in) {

}


/**
@brief Constructor of the class. Allocates data for matrix rows_in times cols_in. By default the created instance would be the owner of the stored data.
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
matrix::matrix( size_t rows_in, size_t cols_in) : matrix_base<Complex16>(rows_in, cols_in) {


}


/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input matrix. (Needed for TBB calls)
@param An instance of class matrix to be copied.
*/
matrix::matrix(const matrix &in) : matrix_base<Complex16>(in)  {

}






} //PIC
