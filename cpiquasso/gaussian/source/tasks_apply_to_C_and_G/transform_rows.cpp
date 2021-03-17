#include <iostream>
#include <tbb/tbb.h>
#include "transform_rows.h"
#include "matrix.h"
#include "dot.h"
#include <memory.h>

namespace pic {




/**
@brief Constructor of the class.
@param mtx_in The matrix to be transformed.
@param T_in The matrix of the transformation
@return Returns with the instance of the class.
*/
Transform_Rows::Transform_Rows( matrix &T_in, matrix &mtx_in ) {

    mtx = mtx_in;
    T = T_in;

    T_data = T.get_data();
    mtx_data = mtx.get_data();


}

/**
@brief Operator to extract a row labeled by i-th element of modes.
@param msg A TBB message fireing the node
*/
const tbb::flow::continue_msg &
Transform_Rows::operator()(const tbb::flow::continue_msg &msg) {

#ifdef DEBUG
    std::cout << "Tasks apply_to_C_and_G: transforming rows" << std::endl;
#endif

    // calculating the product T*mtx
    matrix dot_res = dot( T, mtx );

    // copy the result into the input matrix
    memcpy(mtx_data, dot_res.get_data(), mtx.rows*mtx.cols*sizeof(Complex16));

    return msg;

}





} // PIC
