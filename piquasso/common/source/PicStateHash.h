#ifndef PICSTATEHASH_H
#define PICSTATEHASH_H

#include "PicState.h"

namespace pic {


/**
@brief Class to hash function operator for PicState keys in unordered maps
*/
class PicStateHash {

protected:

public:

/**
@brief Constructor of the class.
@return Returns with the instance of the class.
*/
PicStateHash();

/**
@brief Operator to generate hash key for class instance PicState_int64
@param key An instance of class PicState
@return Returns with the calculated hash value.
*/
size_t operator()(const PicState_int64 &key) const;


}; //PicStateHash







} // PIC

#endif
