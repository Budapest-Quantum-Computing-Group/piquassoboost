/**
 * Copyright 2021 Budapest Quantum Computing Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef array_base_HPP
#define array_base_HPP

#include "PicTypes.hpp"
#include <cstring>
#include <iostream>
#include <tbb/scalable_allocator.h>
#include <tbb/spin_mutex.h>


/// The namespace of the Piquasso project
namespace pic {





/**
@brief Base Class to store data of arrays and its properties.
*/
template<typename scalar>
class array_base {

public:
  /// The number of columns
  size_t cols;
  /// pointer to the stored data
  scalar* data;

protected:

  /// logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  bool owner;
  /// mutual exclusion to count the references for class instances referring to the same data.
  tbb::spin_mutex* reference_mutex;
  /// the number of the current references of the present object
  int64_t* references;



public:

/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
array_base() {

  // The number of columns
  cols = 0;
  // pointer to the stored data
  data = NULL;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  owner = false;
  // mutual exclusion to count the references for class instances referring to the same data.
  reference_mutex = new tbb::spin_mutex();
  references = new int64_t;
  (*references)=1;
}


/**
@brief Constructor of the class. By default the created class instance would not be owner of the stored data.
@param data_in The pointer pointing to the data
@param cols_in The number of columns in the stored array
@return Returns with the instance of the class.
*/
array_base( scalar* data_in, size_t cols_in) {

  // The number of columns
  cols = cols_in;
  // pointer to the stored data
  data = data_in;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  owner = false;
  // mutual exclusion to count the references for class instances referring to the same data.
  reference_mutex = new tbb::spin_mutex();
  references = new int64_t;
  (*references)=1;
}



/**
@brief Constructor of the class. Allocates data for array length of cols_in. By default the created instance would be the owner of the stored data.
@param cols_in The number of columns in the stored array
@return Returns with the instance of the class.
*/
array_base( size_t cols_in) {

  // The number of columns
  cols = cols_in;
  // pointer to the stored data
  data = (scalar*)scalable_aligned_malloc( cols*sizeof(scalar), CACHELINE);
#ifdef DEBUG
  if (cols>0) assert(data);
#endif
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  owner = true;
  // mutual exclusion to count the references for class instances referring to the same data.
  reference_mutex = new tbb::spin_mutex();
  references = new int64_t;
  (*references)=1;


}



/**
@brief Copy constructor of the class. The new instance shares the stored memory with the input array. (Needed for TBB calls)
@param An instance of class array to be copied.
*/
array_base(const array_base<scalar> &in) {

    data = in.data;
    cols = in.cols;
    owner = in.owner;

    reference_mutex = in.reference_mutex;
      references = in.references;

    {
      tbb::spin_mutex::scoped_lock my_lock{*reference_mutex};
      (*references)++;
    }

}



/**
@brief Destructor of the class
*/
~array_base() {
  release_data();
}




/**
@brief Call to get the pointer to the stored data
*/
scalar* get_data() {

  return data;

}


/**
@brief Call to replace the stored data by an another data array. If the class was the owner of the original data array, then it is released.
@param data_in The data array to be set as a new storage.
@param owner_in Set true to set the current class instance to be the owner of the data array, or false otherwise.
*/
void replace_data( scalar* data_in, bool owner_in) {

    release_data();
    data = data_in;
    owner = owner_in;

    reference_mutex = new tbb::spin_mutex();
    references = new int64_t;
    (*references)=1;

}


/**
@brief Call to release the data stored by the array. (If the class instance was not the owner of the data, then the data pointer is simply set to NULL pointer.)
*/
void release_data() {

    if (references==NULL) return;
    bool call_delete = false;

{

    tbb::spin_mutex::scoped_lock my_lock{*reference_mutex};

    if (references==NULL) return;
    call_delete = ((*references)==1);


    if (call_delete) {
      // release the data when array is the owner
      if (owner) {
        scalable_aligned_free(data);
      }
      delete references;
    }
    else {
        (*references)--;
    }

    data = NULL;
    references = NULL;

}

  if ( call_delete && reference_mutex !=NULL) {
    delete reference_mutex;
  }

}



/**
@brief Call to set the current class instance to be (or not to be) the owner of the stored data array.
@param owner_in Set true to set the current class instance to be the owner of the data array, or false otherwise.
*/
void set_owner( bool owner_in)  {

    owner=owner_in;

}

/**
@brief Assignment operator.
@param arr An instance of class array_base
@return Returns with the instance of the class.
*/
void operator= (const array_base& arr ) {

  // releasing the containing data
  release_data();

  // The number of columns
  cols = arr.cols;
  // pointer to the stored data
  data = arr.data;
  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  owner = arr.owner;

  reference_mutex = arr.reference_mutex;
  references = arr.references;

  {
      tbb::spin_mutex::scoped_lock my_lock{*reference_mutex};
      (*references)++;
  }

}


/**
@brief Operator [] to access elements in array style (checks the boundaries of the stored array in DEBUG target)
@param idx the index of the element
@return Returns with a reference to the idx-th element.
*/
scalar& operator[](size_t idx) {

#ifdef DEBUG
    if ( idx >= cols || idx < 0) {
        std::cout << "Accessing element out of bonds at array_base. Exiting" << std::endl;
        exit(-1);
    }
#endif

    return data[idx];
}






/**
@brief Call to create a copy of the array
@return Returns with the instance of the class.
*/
array_base<scalar> copy() {

  array_base<scalar> ret = array_base<scalar>(cols);

  // logical value indicating whether the class instance is the owner of the stored data or not. (If true, the data array is released in the destructor)
  ret.owner = true;

  memcpy( ret.data, data, cols*sizeof(scalar));

  return ret;

}



/**
@brief Call to get the number of the allocated elements
@return Returns with the number of the allocated elements (cols)
*/
size_t size() {

  return cols;

}


/**
@brief Call to prints the stored array on the standard output
*/
void print_array() {
    std::cout << std::endl << "The stored array:" << std::endl;
    for ( size_t col_idx=0; col_idx < cols; col_idx++ ) {
        std::cout << data[col_idx] << " ";
    }
    std::cout << std::endl << std::endl << std::endl;

}






}; //array_base






}  //PIC

#endif // array_base_HPP
