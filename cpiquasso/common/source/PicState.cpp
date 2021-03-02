#include <PicState.h>


namespace pic {



/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
PicState_int64::PicState_int64() : matrix_base<int64_t>() {
    rows = 1;
}

/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
PicState_int64::PicState_int64( int64_t* data_in, size_t cols_in) : matrix_base<int64_t>(data_in, 1, cols_in) {

}


/**
@brief Constructor of the class. Allocates data for the state of elements cols. By default the created instance would be the owner of the stored data.
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
PicState_int64::PicState_int64( size_t cols_in) : matrix_base<int64_t>(1, cols_in) {

}

/**
@brief Constructor of the class. Allocates data for the state of elements cols and set the values to value. By default the created instance would be the owner of the stored data.
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
PicState_int64::PicState_int64( size_t cols_in, int64_t value) : matrix_base<int64_t>(1, cols_in) {

    memset(data, value, cols*sizeof(int64_t));

}


/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input matrix. (Needed for TBB calls)
@param An instance of class matrix to be copied.
*/
PicState_int64::PicState_int64(const PicState_int64 &in) : matrix_base<int64_t>(in) {

    number_of_photons = in.number_of_photons;

}



/**
@brief Operator to compare two keys made of PicState_int64 class instances.
@param key An instance of class PicState
@return Returns with true if the two keys are equal, or false otherwise
*/
bool
PicState_int64::operator==( const PicState_int64 &state) const {

    if (this->cols != state.cols) {
        return false;
    }

    int64_t *data_this = this->data;
    int64_t *data = state.data;

    for (size_t idx=0; idx<this->cols; idx++) {
        if ( data_this[idx] != data[idx] ) {
            return false;
        }
    }

    return true;

}


/**
@brief Operator [] to access elements in array style
@param idx the index of the element
@return Returns with a reference to the idx-th element.
*/
int64_t&
PicState_int64::operator[](size_t idx) {
    return data[idx];
}


/**
@brief Overloaded assignment operator to create a copy of the state
@param state An instance of PicState_int64
@return Returns with the instance of the class.
*/
void
PicState_int64::operator= (const PicState_int64 &state ) {

    matrix_base<int64_t>::operator=( state );

    number_of_photons = state.number_of_photons;

}



/**
@brief Call to create a copy of the state. By default the created instance would be the owner of the stored array.
@return Returns with the instance of the class.
*/
PicState_int64
PicState_int64::copy() {

     PicState_int64 ret = PicState_int64(cols);

    // logical variable indicating whether the matrix needs to be conjugated in CBLAS operations
    ret.conjugated = conjugated;
    // logical variable indicating whether the matrix needs to be transposed in CBLAS operations
    ret.transposed = transposed;
    // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
    ret.owner = true;

    memcpy( ret.data, data, rows*cols*sizeof(int64_t));


    ret.number_of_photons = number_of_photons;

    return ret;

}



} // PIC

