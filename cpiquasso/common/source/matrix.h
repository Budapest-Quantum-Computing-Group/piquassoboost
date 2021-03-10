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


using  matrix_base::matrix_base;



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
