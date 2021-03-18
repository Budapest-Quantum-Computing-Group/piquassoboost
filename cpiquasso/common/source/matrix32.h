#ifndef matrix32_H
#define matrix32_H

#include "matrix_base.hpp"




/// The namespace of the Picasso project
namespace pic {


/**
@brief Class to store data of Complex32 arrays and its properties. Compatible with the Picasso numpy interface.
*/
class matrix32 : public matrix_base<Complex32> {

#if CACHELINE>=64
private:
    /// padding class object to cache line borders
    uint8_t padding[CACHELINE-sizeof(matrix_base<Complex32>)];
#endif

public:


using  matrix_base::matrix_base;



/**
@brief Call to create a copy of the matrix
@return Returns with the instance of the class.
*/
matrix32 copy();

/**
@brief Call to check the array for NaN entries.
@return Returns with true if the array has at least one NaN entry.
*/
bool isnan();


}; //matrix32






}  //PIC

#endif
