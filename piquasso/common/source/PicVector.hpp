#ifndef PICVECTOR_H
#define PICVECTOR_H

#include <vector>
#include <iostream>
#include "tbb/tbb.h"


namespace pic {

/**
@brief Class representing a vector used in the Picasso project. Contains several extra attributes compared to the standard C++ vector class
*/
template<class T>
class PicVector : public std::vector<T,tbb::cache_aligned_allocator<T>> {

public:
    /// The number of nonzero elements. Must be set manually
    int64_t number_of_photons = -1;

    // inheriting all the constructors of std::vector
    using std::vector<T,tbb::cache_aligned_allocator<T>>::vector;

    // inheriting the overloaded assignment operator
    //using std::vector<T,tbb::cache_aligned_allocator<T>>::operator=;

public:
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
