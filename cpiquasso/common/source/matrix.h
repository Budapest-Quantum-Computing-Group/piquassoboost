#ifndef matrix_H
#define matrix_H

#include "matrix_base.hpp"




/// The namespace of the Picasso project
namespace pic {


/**
@brief Class to store data of complex arrays and its properties. Compatible with the Picasso numpy interface.
*/
class matrix : public matrix_base<Complex16> {

#if CACHELINE>=64
private:
    /// padding class object to cache line borders
    uint8_t padding[CACHELINE-sizeof(matrix_base<Complex16>)];
#endif

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
matrix();


/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
matrix( Complex16* data_in, size_t rows_in, size_t cols_in);


/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@param stride_in The column stride of the matrix array (The array elements in one row are a_0, a_1, ... a_{cols-1}, 0, 0, 0, 0. The number of zeros is stride-cols)
@return Returns with the instance of the class.
*/
matrix( Complex16* data_in, size_t rows_in, size_t cols_in, size_t stride_in);

/**
@brief Constructor of the class. Allocates data for matrix rows_in times cols_in. By default the created instance would be the owner of the stored data.
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
matrix( size_t rows_in, size_t cols_in);


/**
@brief Constructor of the class. Allocates data for matrix rows_in times cols_in. By default the created instance would be the owner of the stored data.
@param rows_in The number of rows in the stored matrix
@param cols_in The number of columns in the stored matrix
@param stride_in The column stride of the matrix array (The array elements in one row are a_0, a_1, ... a_{cols-1}, 0, 0, 0, 0. The number of zeros is stride-cols)
@return Returns with the instance of the class.
*/
matrix( size_t rows_in, size_t cols_in, size_t stride_in);

/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input matrix. (Needed for TBB calls)
@param An instance of class matrix to be copied.
*/
matrix(const matrix &in);


/**
@brief Call to create a copy of the matrix
@return Returns with the instance of the class.
*/
matrix copy();

/**
@brief Call to check the array for NaN entries.
@return Returns with true if the array has at least one NaN entry.
*/
bool isnan();


}; //matrix






}  //PIC

#endif
