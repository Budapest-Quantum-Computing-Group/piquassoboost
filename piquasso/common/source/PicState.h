#ifndef PICSTATE_H
#define PICSTATE_H

#include <matrix_base.hpp>
#include <vector>
#include "tbb/tbb.h"


namespace pic {


/**
@brief Class to store one-dimensional state vectors and their additional properties. Compatible with the Picasso numpy interface.
*/
class PicState_int64 : public matrix_base<int64_t> {

public:
    /// The number of phonons stored by the state. Must be set manually
    int64_t number_of_photons = -1;

public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
PicState_int64();

/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
PicState_int64( int64_t* data_in, size_t cols_in);


/**
@brief Constructor of the class. Allocates data for the state of elements cols. By default the created instance would be the owner of the stored data.
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
PicState_int64( size_t cols_in);


/**
@brief Constructor of the class. Allocates data for the state of elements cols and set the values to value. By default the created instance would be the owner of the stored data.
@param cols_in The number of columns in the stored matrix
@return Returns with the instance of the class.
*/
PicState_int64( size_t cols_in, int64_t value);


/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input matrix. (Needed for TBB calls)
@param An instance of class matrix to be copied.
*/
PicState_int64(const PicState_int64 &in);

/**
@brief Operator to compare two keys made of PicState_int64 class instances.
@param key An instance of class PicState
@return Returns with true if the two keys are equal, or false otherwise
*/
bool operator==( const PicState_int64 &key) const;


/**
@brief Operator [] to access elements in array style
@param idx the index of the element
@return Returns with a reference to the idx-th element.
*/
int64_t& operator[](size_t idx);

/**
@brief Overloaded assignment operator to create a copy of the state
@param state An instance of PicState_int64
@return Returns with the instance of the class.
*/
void operator= (const PicState_int64 &state );


/**
@brief Call to create a copy of the state. By default the created instance would be the owner of the stored array.
@return Returns with the instance of the class.
*/
PicState_int64 copy();


};


/// container of aligned states aligned to cache line border
using PicStates = std::vector< PicState_int64, tbb::cache_aligned_allocator<PicState_int64> >;

/// concurrent container of aligned states aligned to cache line border
using concurrent_PicStates = tbb::concurrent_vector<PicState_int64, tbb::cache_aligned_allocator<PicState_int64> >;



} // PIC

#endif
