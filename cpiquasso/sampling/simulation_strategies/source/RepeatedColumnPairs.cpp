#include "RepeatedColumnPairs.h"



namespace pic {


/**
@brief Nullary constructor of the class
*/
SingleMode::SingleMode() {

}

/**
@brief Constructor of the class
@param mode_in The label of the mode
@param degeneracy_in The number describing how many times the mode pair is repeated
*/
SingleMode::SingleMode(size_t mode_, size_t degeneracy_) {
    mode = mode_;
    degeneracy = degeneracy_;

}





/**
@brief Nullary constructor of the class
*/
ModePair::ModePair() {

}

/**
@brief Constructor of the class
@param mode1_in The label of the first mode in the pair
@param mode2_in The label of the second mode in the pair
@param degeneracy_in The number describing how many times the mode pair is repeated
*/
ModePair::ModePair(size_t mode1_in, size_t mode2_in, size_t degeneracy_in) {
    mode1 = mode1_in;
    mode2 = mode2_in;
    degeneracy = degeneracy_in;

}



/**
@brief Call to sort the modes in descent order and obtain the inverse permutation indices.
@param sorted_modes The array of mode occupacies to be sorted. The sorted array is returned by this reference.
@param inverse_permutation The inverse permutation indices are returned via this reference. For initial value 1,2,3,4... should be given.
@param start_idx Starting index;
@param end_idx The last index + 1
*/
void SortModes( PicState_int64& sorted_modes, PicState_int64& inverse_permutation, size_t start_idx, size_t end_idx) {

    if (start_idx >= end_idx ) return;

    // do shuffle
    int64_t pivot_value = sorted_modes[start_idx];
    size_t idx_tmp = start_idx;
    size_t jdx_tmp = end_idx-1;

    while ( idx_tmp < jdx_tmp ) {

        while( idx_tmp < jdx_tmp && pivot_value > sorted_modes[jdx_tmp]) jdx_tmp--;
        while( idx_tmp < jdx_tmp && sorted_modes[idx_tmp] >= pivot_value) idx_tmp++;


        // swap the elements
        int64_t val_tmp = sorted_modes[idx_tmp];
        sorted_modes[idx_tmp] = sorted_modes[jdx_tmp];
        sorted_modes[jdx_tmp] = val_tmp;

        val_tmp = inverse_permutation[idx_tmp];
        inverse_permutation[idx_tmp] = inverse_permutation[jdx_tmp];
        inverse_permutation[jdx_tmp] = val_tmp;
    }

    // swap the elements
    int64_t val_tmp = sorted_modes[idx_tmp];
    sorted_modes[idx_tmp] = sorted_modes[start_idx];
    sorted_modes[start_idx] = val_tmp;

    val_tmp = inverse_permutation[idx_tmp];
    inverse_permutation[idx_tmp] = inverse_permutation[start_idx];
    inverse_permutation[start_idx] = val_tmp;

    // recursive call
    SortModes( sorted_modes, inverse_permutation, start_idx, idx_tmp );
    SortModes( sorted_modes, inverse_permutation, idx_tmp+1, end_idx );


    return;

}


/**
@brief Call to determine mode pairs and single modes for the recursive hafnian algorithm
@param single_modes The single modes are returned by this vector
@param mode_pairs The mode pairs are returned by this vector
@param modes The mode occupancy for which the mode pairs are determined.
*/
void DetermineModePairs(std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs, PicState_int64& modes ) {

    // local, mutable version of the mode occupancies
    PicState_int64 modes_loc = modes.copy();

    ModePair modes_pair;
    size_t tmp = 0;

    for (size_t idx=0; idx<modes_loc.size(); idx++) {

        // identify a single mode
        if (modes_loc[idx] == 1) {
            single_modes.push_back( SingleMode(idx, 1));
            modes_loc[idx] = 0;
//modes_loc.print_matrix();
        }
        else if (modes_loc[idx] > 1) {
            if (tmp==0) {
                modes_pair.mode1 = idx;
                tmp++;
            }
            else if (tmp==1) {
                modes_pair.mode2 = idx;
                tmp++;
            }

            if (tmp==2) {
                modes_pair.degeneracy = modes_loc[modes_pair.mode1] <= modes_loc[modes_pair.mode2] ? modes_loc[modes_pair.mode1] : modes_loc[modes_pair.mode2];
                modes_loc[modes_pair.mode1] = modes_loc[modes_pair.mode1] - modes_pair.degeneracy;
                modes_loc[modes_pair.mode2] = modes_loc[modes_pair.mode2] - modes_pair.degeneracy;

 //modes_loc.print_matrix();

                mode_pairs.push_back( modes_pair);

                if ( modes_loc[modes_pair.mode1] == 0 && modes_loc[modes_pair.mode2] == 0) {
                    modes_pair.mode1 = 0;
                    modes_pair.mode2 = 0;
                    modes_pair.degeneracy = 0;
                    tmp = 0;
                }
                else if ( modes_loc[modes_pair.mode1] > 0 && modes_loc[modes_pair.mode2] == 0) {
                    modes_pair.mode2 = 0;
                    modes_pair.degeneracy = 0;
                    tmp = 1;

                    if (modes_loc[modes_pair.mode1] == 1) {
                        single_modes.push_back( SingleMode(modes_pair.mode1, 1));
                        modes_pair.mode1 = 0;
                        modes_loc[modes_pair.mode1] = 0;
//modes_loc.print_matrix();
                        tmp = 0;
                    }

                }
                else if ( modes_loc[modes_pair.mode1] == 0 && modes_loc[modes_pair.mode2] > 0) {
                    modes_pair.mode1 = modes_pair.mode2;
                    modes_pair.mode2 = 0;
                    modes_pair.degeneracy = 0;
                    tmp = 1;

                    if (modes_loc[modes_pair.mode1] == 1) {
                        single_modes.push_back( SingleMode(modes_pair.mode1, 1));
                        modes_loc[modes_pair.mode1] = 0;
                        modes_pair.mode1 = 0;
//modes_loc.print_matrix();
                        tmp = 0;
                    }


                }
            }



        }

    }


    // create single degenerate mode pairs from the last degenerate mode that has no further piars
    if ( tmp==1 ) {

        if ( modes_loc[modes_pair.mode1] < single_modes.size() ) {

            size_t idx_min = single_modes.size()-modes_loc[modes_pair.mode1]-1;

            for (size_t idx=single_modes.size()-1; idx>idx_min; idx--) {
                modes_pair.mode2 = single_modes[idx].mode;
                modes_pair.degeneracy = 1;
                mode_pairs.push_back( modes_pair );
                single_modes.pop_back();
            }



        }
        else {
            size_t idx_min = 0;
            size_t remaining_degeneracy = modes_loc[modes_pair.mode1] - single_modes.size();

            for (size_t idx=single_modes.size(); idx>idx_min; idx--) {

                modes_pair.mode2 = single_modes[idx-1].mode;
                modes_pair.degeneracy = 1;
                mode_pairs.push_back( modes_pair );
                single_modes.pop_back();
            }

            if (remaining_degeneracy>0) {
                single_modes.push_back( SingleMode(modes_pair.mode1, remaining_degeneracy));
            }

        }


/*
        for (size_t idx=0; idx<modes_loc[modes_pair.mode1]; idx++) {
            single_modes.push_back( SingleMode(modes_pair.mode1, 1));
        }
*/
//modes_loc[modes_pair.mode1] = 0;
//modes_loc.print_matrix();
    }

    return;
}


/**
@brief Call to permute the elements of a given row. The modes corresponding to a given mode pairs are placed next to each other.
@param row An array containing the elements to be permuted.
@param single_modes The single modes determined by DetermineModePairs.
@param mode_pairs The mode pairs determined by DetermineModePairs.
@param row_permuted A preallocated array for the permuted row elements.
*/
void PermuteRow( matrix row, const size_t& row_mode, const bool &is_row_mode_conjugated, std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs, matrix& row_permuted, PicState_int64& inverse_permutation) {

    size_t dim_over_2 = row.size()/2;

    memset( row_permuted.get_data(), 0, row_permuted.size()*sizeof(Complex16));
/*
for (size_t idx=0; idx<dim_over_2; idx++ ) {
    std::cout << row[idx];
}
std::cout << std::endl;
for (size_t idx=0; idx<dim_over_2; idx++ ) {
    std::cout << row[idx+dim_over_2];
}
std::cout << std::endl;
*/

    for (size_t idx=0; idx<mode_pairs.size(); idx++) {

        ModePair& mode_pair = mode_pairs[idx];

        row_permuted[4*idx] = row[inverse_permutation[mode_pair.mode1]];
        row_permuted[4*idx+1] = row[inverse_permutation[mode_pair.mode2]];
        row_permuted[4*idx+2] = row[inverse_permutation[mode_pair.mode1] + dim_over_2];
        row_permuted[4*idx+3] = row[inverse_permutation[mode_pair.mode2] + dim_over_2];

    }

    size_t offset = 4*mode_pairs.size();
    size_t idx = 0;
    size_t tmp = 0;
    ModePair mode_pair;
    while (idx<single_modes.size()) {

        if (tmp==0) {
            mode_pair.mode1 = single_modes[idx].mode;
            tmp++;
        }
        else if(tmp==1) {
            mode_pair.mode2 = single_modes[idx].mode;
            tmp++;
        }


        if (tmp == 2) {

            row_permuted[offset]   = row[inverse_permutation[mode_pair.mode1]];
            row_permuted[offset+1] = row[inverse_permutation[mode_pair.mode2]];
            row_permuted[offset+2] = row[inverse_permutation[mode_pair.mode1] + dim_over_2];
            row_permuted[offset+3] = row[inverse_permutation[mode_pair.mode2] + dim_over_2];
            offset = offset + 4;
            tmp = 0;
        }


        idx++;
    }


    if (tmp == 1) {

        row_permuted[offset]   = row[inverse_permutation[mode_pair.mode1]];
        row_permuted[offset+1] = row[inverse_permutation[mode_pair.mode1] + dim_over_2];
    }


    return;

}


/**
@brief Call to construct the matrix containing repeated mode pairs and the array containing the repeating factors of the column pairs.
 The modes corresponding to a given mode pair are placed next to each other.
@param mtx The input matrix for which the repeated matrix should be constructed.
@param single_modes The single modes determined by DetermineModePairs.
@param mode_pairs The mode pairs determined by DetermineModePairs.
@param mtx_out A matrix (not preallocated) to reference the constructed permuted row elements.
@param repeated_column_pairs An array (not preallocated) to reference the constructed  repeating factors.
@param
*/
void ConstructRepeatedMatrix(matrix &mtx, std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs, PicState_int64& inverse_permutation,
                             matrix &mtx_out, PicState_int64& repeated_column_pairs) {

    // determine the size of the output matrix
    size_t dim=0;
    size_t dim_over_2=0;
    for (size_t idx=0; idx<single_modes.size(); idx++) {
        dim_over_2 = dim_over_2 + 1;
    }

    for (size_t idx=0; idx<mode_pairs.size(); idx++) {
        dim_over_2 = dim_over_2 + 2;
    }
    dim = 2*dim_over_2;

    // preallocate the output arrays
    mtx_out = matrix(dim, dim);
    memset(mtx_out.get_data(), 0, mtx_out.size());
    repeated_column_pairs = PicState_int64(dim_over_2, 1);


    size_t tmp = 0;
    for (size_t idx=0; idx<mode_pairs.size(); idx++) {

        ModePair& mode_pair = mode_pairs[idx];

        repeated_column_pairs[2*idx] =  mode_pair.degeneracy;
        repeated_column_pairs[2*idx+1] =  mode_pair.degeneracy;

        size_t row_offset = inverse_permutation[mode_pair.mode1] * mtx.stride;
        matrix row( mtx.get_data() + row_offset, 1, mtx.cols );

        size_t row_offset_out = 4*idx * mtx_out.stride;
        matrix row_permuted( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, mode_pair.mode1, false, single_modes, mode_pairs, row_permuted, inverse_permutation);

        ///////////////////////////////////////////////////////////////////////////

        row_offset = inverse_permutation[mode_pair.mode2] * mtx.stride;
        row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

        row_offset_out = (4*idx+1) * mtx_out.stride;
        row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, mode_pair.mode2, false, single_modes, mode_pairs, row_permuted, inverse_permutation);


        ///////////////////////////////////////////////////////////////////////////

        row_offset = (inverse_permutation[mode_pair.mode1]+mtx.rows/2) * mtx.stride;
        row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

        row_offset_out = (4*idx+2) * mtx_out.stride;
        row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, mode_pair.mode1, true, single_modes, mode_pairs, row_permuted, inverse_permutation);


        ///////////////////////////////////////////////////////////////////////////

        row_offset = (inverse_permutation[mode_pair.mode2]+mtx.rows/2) * mtx.stride;
        row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

        row_offset_out = (4*idx+3) * mtx_out.stride;
        row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, mode_pair.mode2, true, single_modes, mode_pairs, row_permuted, inverse_permutation);

    }


    size_t offset = 4*(mode_pairs.size());
    tmp = 0;
    ModePair mode_pair;
    size_t idx = 0;
    while (idx<single_modes.size()) {

        if (tmp==0) {
            mode_pair.mode1 = single_modes[idx].mode;
            mode_pair.degeneracy = single_modes[idx].degeneracy;
            tmp++;
        }
        else if(tmp==1) {
            mode_pair.mode2 = single_modes[idx].mode;
            tmp++;
        }


        if (tmp == 2) {
            size_t row_offset = inverse_permutation[mode_pair.mode1] * mtx.stride;
            matrix row( mtx.get_data() + row_offset, 1, mtx.cols );

            size_t row_offset_out = (offset) * mtx_out.stride;
            matrix row_permuted( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, mode_pair.mode1, false, single_modes, mode_pairs, row_permuted, inverse_permutation);

            ///////////////////////////////////////////////////////////////

            row_offset = inverse_permutation[mode_pair.mode2] * mtx.stride;
            row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

            row_offset_out = (offset+1) * mtx_out.stride;
            row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, mode_pair.mode2, false, single_modes, mode_pairs, row_permuted, inverse_permutation);


            ///////////////////////////////////////////////////////////////

            row_offset = (inverse_permutation[mode_pair.mode1]+mtx.rows/2) * mtx.stride;
            row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

            row_offset_out = (offset+2) * mtx_out.stride;
            row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, mode_pair.mode1, true, single_modes, mode_pairs, row_permuted, inverse_permutation);


            ///////////////////////////////////////////////////////////////

            row_offset = (inverse_permutation[mode_pair.mode2]+mtx.rows/2) * mtx.stride;
            row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

            row_offset_out = (offset+3) * mtx_out.stride;
            row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, mode_pair.mode2, true, single_modes, mode_pairs, row_permuted, inverse_permutation);

            offset = offset + 4;


            tmp = 0;
        }


        idx++;
    }


    if (tmp == 1) {

        repeated_column_pairs[repeated_column_pairs.size()-1] = mode_pair.degeneracy;

        size_t row_offset = inverse_permutation[mode_pair.mode1] * mtx.stride;
        matrix row( mtx.get_data() + row_offset, 1, mtx.cols );

        size_t row_offset_out = offset * mtx_out.stride;
        matrix row_permuted( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, mode_pair.mode1, false, single_modes, mode_pairs, row_permuted, inverse_permutation);


        ///////////////////////////////////////////////////////////////

        row_offset = (inverse_permutation[mode_pair.mode1]+mtx.rows/2) * mtx.stride;
        row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

        row_offset_out = (offset + 1) * mtx_out.stride;
        row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, mode_pair.mode1, true, single_modes, mode_pairs, row_permuted, inverse_permutation);
    }



    return;

}


/**
@brief Call to construct the matrix containing repeated mode pairs and the array containing the repeating factors of the column pairs.
 The modes corresponding to a given mode pair are placed next to each other.
@param mtx The input matrix for which the repeated matrix should be constructed.
@param modes The mode occupancies for which the mode pairs are determined.
@param mtx_out A matrix (not preallocated) to reference the constructed matrix.
@param repeated_column_pairs An array containing the repeating factors of the successive column pairs is returned by this reference.
*/
void ConstructMatrixForRecursivePowerTrace(matrix &mtx, PicState_int64& modes, matrix &mtx_out, PicState_int64& repeated_column_pairs) {


    // preallocate and initialize computing arrays
    PicState_int64 sorted_modes = modes.copy();
    PicState_int64 inverse_permutation(modes.size());

    for (size_t idx = 0; idx < inverse_permutation.size(); idx++) {
        inverse_permutation[idx] = idx;
    }

    // sort modes in descending orders
    SortModes( sorted_modes, inverse_permutation, 0, modes.size());


    // Determine repeating mode pairs and remaining single columns
    std::vector<pic::SingleMode> single_modes;
    std::vector<pic::ModePair> mode_pairs;
    DetermineModePairs(single_modes, mode_pairs, sorted_modes);
/*
for(size_t idx=0; idx<single_modes.size(); idx++) {
    std::cout << inverse_permutation[single_modes[idx].mode] << " " << single_modes[idx].degeneracy << std::endl;
}
std::cout << std::endl;

for(size_t idx=0; idx<mode_pairs.size(); idx++) {
    std::cout << inverse_permutation[mode_pairs[idx].mode1] << " " << inverse_permutation[mode_pairs[idx].mode2] << " " << mode_pairs[idx].degeneracy << std::endl;
}
std::cout << std::endl;
*/

    // construct the final matrix with repeated column pairs and the repeating factors of the column pairs
    ConstructRepeatedMatrix(mtx, single_modes, mode_pairs, inverse_permutation, mtx_out, repeated_column_pairs );


    return;


}


/**
@brief Call to replace the diagonal elements of the constructed repeated matrix by the array gamma.
(The elements of gamma are permuted and repeated in the same fashion as the elements of the original matrix)
 The modes corresponding to a given mode pair are placed next to each other.
@param diags The prealloctaed array for the diagonal elements
@param gamma The diagonal elements that are about to be used in the replacement
@param single_modes The single modes determined by DetermineModePairs.
@param mode_pairs The mode pairs determined by DetermineModePairs.
@param inverse_permutation The array containing the permutation indices determined by SortModes
*/
void CreateDiagonalForDisplacedGBS(matrix &diags, matrix& gamma, std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs, PicState_int64& inverse_permutation) {

    // determine the size of the output matrix
    size_t dim=0;
    size_t dim_over_2=0;
    for (size_t idx=0; idx<single_modes.size(); idx++) {
        dim_over_2 = dim_over_2 + 1;
    }

    for (size_t idx=0; idx<mode_pairs.size(); idx++) {
        dim_over_2 = dim_over_2 + 2;
    }
    dim = 2*dim_over_2;
    diags = matrix(dim,1);

    dim_over_2 = gamma.size()/2;
    // add the diagonal correction to the Hamilton's matrix
    for (size_t idx=0; idx<mode_pairs.size(); idx++) {
        diags[4*idx] = gamma[inverse_permutation[mode_pairs[idx].mode1]];
        diags[4*idx+1] = gamma[inverse_permutation[mode_pairs[idx].mode2]];
        diags[4*idx+2] = gamma[inverse_permutation[mode_pairs[idx].mode1]+dim_over_2];
        diags[4*idx+3] = gamma[inverse_permutation[mode_pairs[idx].mode2]+dim_over_2];
    }

size_t offset = 4*(mode_pairs.size());
    size_t tmp = 0;
    ModePair mode_pair;
    size_t idx = 0;
    while (idx<single_modes.size()) {

        if (tmp==0) {
            mode_pair.mode1 = single_modes[idx].mode;
            tmp++;
        }
        else if(tmp==1) {
            mode_pair.mode2 = single_modes[idx].mode;
            tmp++;
        }


        if (tmp == 2) {

            diags[offset] = gamma[inverse_permutation[mode_pair.mode1]];
            diags[offset+1] = gamma[inverse_permutation[mode_pair.mode2]];
            diags[offset+2] = gamma[inverse_permutation[mode_pair.mode1]+dim_over_2];
            diags[offset+3] = gamma[inverse_permutation[mode_pair.mode2]+dim_over_2];
            offset = offset + 4;


            tmp = 0;
        }


        idx++;
    }


    if (tmp == 1) {

        diags[offset] = gamma[inverse_permutation[mode_pair.mode1]];
        diags[offset+1] = gamma[inverse_permutation[mode_pair.mode1]+dim_over_2];
    }

    return;


}



/**
@brief Call to construct the matrix containing repeated mode pairs and the array containing the repeating factors of the column pairs.
 The modes corresponding to a given mode pair are placed next to each other.
@param mtx The input matrix for which the repeated matrix should be constructed.
@param modes The mode occupancies for which the mode pairs are determined.
@param mtx_out A matrix (not preallocated) to reference the constructed matrix.
@param diags_out A matrix (not preallocated) to reference the constructed diagonal elements.
@param repeated_column_pairs An array containing the repeating factors of the successive column pairs is returned by this reference.
*/
void ConstructMatrixForRecursiveLoopPowerTrace(matrix &mtx, matrix& gamma, PicState_int64& modes, matrix &mtx_out, matrix &diags_out, PicState_int64& repeated_column_pairs) {

    // preallocate and initialize computing arrays
    PicState_int64 sorted_modes = modes.copy();
    PicState_int64 inverse_permutation(modes.size());

    for (size_t idx = 0; idx < inverse_permutation.size(); idx++) {
        inverse_permutation[idx] = idx;
    }

    // sort modes in descending orders
    SortModes( sorted_modes, inverse_permutation, 0, modes.size());


    // Determine repeating mode pairs and remaining single columns
    std::vector<pic::SingleMode> single_modes;
    std::vector<pic::ModePair> mode_pairs;
    DetermineModePairs(single_modes, mode_pairs, sorted_modes);

    // construct the final matrix with repeated column pairs and the repeating factors of the column pairs
    ConstructRepeatedMatrix(mtx, single_modes, mode_pairs, inverse_permutation, mtx_out, repeated_column_pairs );


    // replace the diagonal elements of the constructed matrix according to according to Eq (11) of arXiv 2010.15595v3
    CreateDiagonalForDisplacedGBS(diags_out, gamma, single_modes, mode_pairs, inverse_permutation);




    return;


}

} //PIC
