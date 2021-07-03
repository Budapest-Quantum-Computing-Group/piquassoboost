#ifndef PICVECTOR_H
#define PICVECTOR_H

#include <vector>
#include <iostream>
#include <tbb/cache_aligned_allocator.h>
#include <tbb/concurrent_vector.h>


namespace pic {

/**
@brief Class representing a vector used in the Picasso project. Contains several extra attributes compared to the standard C++ vector class
*/
template<class T>
class PicVector : public std::vector<T,tbb::cache_aligned_allocator<T>> {

public:
    /// The number of nonzero elements. Must be set manually
    int64_t number_of_photons = -1;


public:

/**
@brief Nullary constructor of the class
@return Returns with the instance of the class.
*/
PicVector() : std::vector<T,tbb::cache_aligned_allocator<T>>() {

}

/**
@brief Constructor of the class
@param num number of elements to be reserved
@return Returns with the instance of the class.
*/
PicVector( size_t num ) : std::vector<T,tbb::cache_aligned_allocator<T>>(num) {

}


/**
@brief Constructor of the class
@param num number of elements to be reserved
@param value The value that is used to fill up the initial container
@return Returns with the instance of the class.
*/
PicVector( size_t num, T value ) : std::vector<T,tbb::cache_aligned_allocator<T>>(num, value) {

}

/**
@brief Overloaded assignment operator
@param vec An instance of PicVector
@return Returns with the instance of the class.
*//*
void operator= (const PicVector &vec ) {
    //nonzero_elements = vec.nonzero_elements;
    std::vector<T,tbb::cache_aligned_allocator<T>>::operator=(vec);
}
*/

};

/// container of aligned vectors aligned to cache line border
template<class T>
using vectors = std::vector< PicVector<T>, tbb::cache_aligned_allocator<PicVector<T>> >;


template <class T>
using concurrent_vectors = tbb::concurrent_vector<PicVector<T>>;

/**
@brief Call to print the elements of a container
@param vec a container
@return Returns with the sum of the elements of the container
*/
template <typename Container>
void
print_state( Container state ) {

    for (auto it = state.begin(); it!=state.end(); it++) {
        std::cout<< *it << " ,";
    }

    std::cout<<std::endl;


}





} // PIC

#endif
