#ifndef RepeatedColumnPairs_H
#define RepeatedColumnPairs_H


#include "matrix.h"
#include "PicState.h"

namespace pic {
/**
@brief This class stores labels and degeneracy of modes that are not coupled into pairs
*/
class SingleMode {


public:
    /// The label of the mode
    size_t mode;
    /// The degeneracy (occupancy) of the given mode
    size_t degeneracy;

SingleMode();

SingleMode(size_t mode_, size_t degeneracy_);

};



/**
@brief This class stores labels and degeneracy of mode paris that are coupled into pairs
*/
class ModePair {

public:
    /// The label of the first mode
    size_t mode1;
    /// The label of the second mode
    size_t mode2;
    /// Describing how many times the mode pair is repeated
    size_t degeneracy;

/**
@brief Nullary constructor of the class
*/
ModePair();

/**
@brief Constructor of the class
@param mode1_in The label of the first mode in the pair
@param mode2_in The label of the second mode in the pair
@param degeneracy_in The number describing how many times the mode pair is repeated
*/
ModePair(size_t mode1_in, size_t mode2_in, size_t degeneracy_in);

};



void SortModes( PicState_int64& modes, PicState_int64& sorted_modes, PicState_int64& inverse_permutation);

/**
@brief Call to determine mode pairs and single modes for the recursive hafnian algorithm
@param single_modes The single modes are returned by this vector
@param mode_pairs The mode pairs are returned by this vector
@param modes The mode occupancy for which the mode pairs are determined.
*/
void DetermineModePairs(std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs, PicState_int64& modes );


/**
@brief Call to permute the elements of a given row. The modes corresponding to a given mode pairs are placed next to each other.
@param row An array containing the elements to be permuted.
@param single_modes The single modes determined by DetermineModePairs.
@param mode_pairs The mode pairs determined by DetermineModePairs.
@param row_permuted A preallocated array for the permuted row elements.
*/
void PermuteRow( matrix row, std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs, matrix& row_permuted, PicState_int64& inverse_permutation);

/**
@brief Call to construct the matrix containing repeated mode pairs and the array containing the repeating factors of the column pairs.
 The modes corresponding to a given mode pair are placed next to each other.
@param mtx The input matrix for which the repeated matrix should be constructed.
@param single_modes The single modes determined by DetermineModePairs.
@param mode_pairs The mode pairs determined by DetermineModePairs.
@param mtx_out A matrix (not preallocated) to reference the constructed permuted row elements.
@param repeated_column_pairs An array (not preallocated) to reference the constructed  repeating factors.
*/
void ConstructRepeatedMatrix(matrix &mtx, std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs,
                             matrix &mtx_out, PicState_int64& repeated_column_pairs, PicState_int64& inverse_permutation);



void ConstructMatrixForRecursivePowerTrace(matrix &mtx, PicState_int64& modes, matrix &mtx_out, PicState_int64& repeated_column_pairs);


} //PIC














#endif
