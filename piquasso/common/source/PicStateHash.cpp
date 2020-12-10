#include "PicStateHash.h"

namespace pic {


/**
@brief Constructor of the class.
@return Returns with the instance of the class.
*/
PicStateHash::PicStateHash() {

}

/**
@brief Operator to generate hash key for class instance PicState
@param key An instance of class PicState
@return Returns with the calculated hash value.
*/
size_t
PicStateHash::operator()(const PicState_int64 &key) const {

    PicState_int64 &key_loc = const_cast<PicState_int64 &>(key);
    int64_t *data = key_loc.get_data();
    size_t hash_val = 0;
    size_t pow2 = 1;

    for (size_t idx=0; idx<key.cols; idx++) {
        hash_val = hash_val + data[idx]*pow2;
        pow2 = pow2*2;
    }
/*
std::cout<< "hash: " << hash_val << std::endl;
std::cout<<"key: ";
key_loc.print_matrix();
*/

    return hash_val;


}






} // PIC

