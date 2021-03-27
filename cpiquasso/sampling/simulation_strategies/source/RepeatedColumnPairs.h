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

/**
@brief Nullary constructor of the class
*/
SingleMode();

/**
@brief Constructor of the class
@param mode_in The label of the mode
@param degeneracy_in The number describing how many times the mode pair is repeated
*/
SingleMode(size_t mode_in, size_t degeneracy_in);

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


/**
@brief Call to sort the modes in descent order and obtain the inverse permutation indices.
@param sorted_modes The array of mode occupacies to be sorted. The sorted array is returned by this reference.
@param inverse_permutation The inverse permutation indices are returned via this reference. For initial value 1,2,3,4... should be given.
@param start_idx Starting index;
@param end_idx The last index + 1
*/
void SortModes( PicState_int64& sorted_modes, PicState_int64& inverse_permutation, size_t start_idx, size_t end_idx);
/**
@brief Call to determine mode pairs and single modes for the recursive hafnian algorithm
@param single_modes The single modes are returned by this vector
@param mode_pairs The mode pairs are returned by this vector
@param modes The mode occupancies for which the mode pairs are determined.
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
@param mtx_out A matrix (not preallocated) to reference the constructed matrix.
@param repeated_column_pairs An array (not preallocated) to reference the constructed  repeating factors.
*/
void ConstructRepeatedMatrix(matrix &mtx, std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs,
                             matrix &mtx_out, PicState_int64& repeated_column_pairs, PicState_int64& inverse_permutation);



/**
@brief Call to replace the diagonal elements of the constructed repeated matrix by the array gamma.
(The elements of gamma are permuted and repeated in the same fashion as the elements of the original matrix)
 The modes corresponding to a given mode pair are placed next to each other.
@param mtx The input matrix for which the diagonal elements should be replaced
@param gamma The diagonal elements that are about to be used in the replacement
@param single_modes The single modes determined by DetermineModePairs.
@param mode_pairs The mode pairs determined by DetermineModePairs.
@param inverse_permutation The array containing the permutation indices determined by SortModes
*/
void ReplaceDiagonalForDisplacedGBS(matrix &mtx, matrix& gamma, std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs, PicState_int64& inverse_permutation);



/**
@brief Call to construct the matrix containing repeated mode pairs and the array containing the repeating factors of the column pairs.
 The modes corresponding to a given mode pair are placed next to each other.
@param mtx The input matrix for which the repeated matrix should be constructed.
@param modes The mode occupancies for which the mode pairs are determined.
@param mtx_out A matrix (not preallocated) to reference the constructed matrix.
@param repeated_column_pairs An array containing the repeating factors of the successive column pairs is returned by this reference.
*/
void ConstructMatrixForRecursivePowerTrace(matrix &mtx, PicState_int64& modes, matrix &mtx_out, PicState_int64& repeated_column_pairs);




/**
@brief Call to construct the matrix containing repeated mode pairs and the array containing the repeating factors of the column pairs.
 The modes corresponding to a given mode pair are placed next to each other.
@param mtx The input matrix for which the repeated matrix should be constructed.
@param modes The mode occupancies for which the mode pairs are determined.
@param mtx_out A matrix (not preallocated) to reference the constructed matrix.
@param repeated_column_pairs An array containing the repeating factors of the successive column pairs is returned by this reference.
*/
void ConstructMatrixForRecursiveLoopPowerTrace(matrix &mtx, matrix& gamma, PicState_int64& modes, matrix &mtx_out, PicState_int64& repeated_column_pairs);


} //PIC














#endif
