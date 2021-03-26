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




void SortModes( PicState_int64& modes, PicState_int64& sorted_modes, PicState_int64& inverse_permutation) {

    PicState_int64 unsorted_modes = modes.copy();
    sorted_modes = PicState_int64(modes.size());
    inverse_permutation = PicState_int64(modes.size());

    for (size_t idx=0; idx<modes.size(); idx++) {

        int64_t current_max = INT64_MIN;
        size_t max_index = 0;
        for (size_t jdx=0; jdx<unsorted_modes.size(); jdx++) {
            if (current_max < unsorted_modes[jdx]) {
                current_max = unsorted_modes[jdx];
                max_index = jdx;
            }
        }

        unsorted_modes[max_index] = INT64_MIN;
        sorted_modes[idx] = current_max;
        inverse_permutation[idx] = max_index;



    }




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

            for ( size_t idx=0; idx<remaining_degeneracy; idx++) {
                single_modes.push_back( SingleMode(modes_pair.mode1, 1));
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
void PermuteRow( matrix row, std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs, matrix& row_permuted, PicState_int64& inverse_permutation) {

    size_t dim_over_2 = row.size()/2;
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
*/
void ConstructRepeatedMatrix(matrix &mtx, std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs,
                             matrix &mtx_out, pic::PicState_int64& repeated_column_pairs, PicState_int64& inverse_permutation) {

    // determine the size of the output matrix
    size_t dim=0;
    size_t dim_over_2=0;
    for (size_t idx=0; idx<single_modes.size(); idx++) {
        dim_over_2 = dim_over_2 + single_modes[idx].degeneracy;
    }

    for (size_t idx=0; idx<mode_pairs.size(); idx++) {
        dim_over_2 = dim_over_2 + 2;
    }
    dim = 2*dim_over_2;


    // preallocate the output arrays
    mtx_out = matrix(dim, dim);
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
        PermuteRow( row, single_modes, mode_pairs, row_permuted, inverse_permutation);

        ///////////////////////////////////////////////////////////////////////////

        row_offset = inverse_permutation[mode_pair.mode2] * mtx.stride;
        row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

        row_offset_out = (4*idx+1) * mtx_out.stride;
        row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, single_modes, mode_pairs, row_permuted, inverse_permutation);


        ///////////////////////////////////////////////////////////////////////////

        row_offset = (inverse_permutation[mode_pair.mode1]+mtx.rows/2) * mtx.stride;
        row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

        row_offset_out = (4*idx+2) * mtx_out.stride;
        row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, single_modes, mode_pairs, row_permuted, inverse_permutation);


        ///////////////////////////////////////////////////////////////////////////

        row_offset = (inverse_permutation[mode_pair.mode2]+mtx.rows/2) * mtx.stride;
        row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

        row_offset_out = (4*idx+3) * mtx_out.stride;
        row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, single_modes, mode_pairs, row_permuted, inverse_permutation);

    }


    size_t offset = 4*(mode_pairs.size());
    tmp = 0;
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
            size_t row_offset = inverse_permutation[mode_pair.mode1] * mtx.stride;
            matrix row( mtx.get_data() + row_offset, 1, mtx.cols );

            size_t row_offset_out = (offset) * mtx_out.stride;
            matrix row_permuted( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted, inverse_permutation);

            ///////////////////////////////////////////////////////////////

            row_offset = inverse_permutation[mode_pair.mode2] * mtx.stride;
            row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

            row_offset_out = (offset+1) * mtx_out.stride;
            row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted, inverse_permutation);


            ///////////////////////////////////////////////////////////////

            row_offset = (inverse_permutation[mode_pair.mode1]+mtx.rows/2) * mtx.stride;
            row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

            row_offset_out = (offset+2) * mtx_out.stride;
            row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted, inverse_permutation);


            ///////////////////////////////////////////////////////////////

            row_offset = (inverse_permutation[mode_pair.mode2]+mtx.rows/2) * mtx.stride;
            row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

            row_offset_out = (offset+3) * mtx_out.stride;
            row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted, inverse_permutation);

            offset = offset + 4;


            tmp = 0;
        }


        idx++;
    }


    if (tmp == 1) {
        size_t row_offset = inverse_permutation[mode_pair.mode1] * mtx.stride;
            matrix row( mtx.get_data() + row_offset, 1, mtx.cols );

            size_t row_offset_out = offset * mtx_out.stride;
            matrix row_permuted( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted, inverse_permutation);


            ///////////////////////////////////////////////////////////////

            row_offset = (inverse_permutation[mode_pair.mode1]+mtx.rows/2) * mtx.stride;
            row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

            row_offset_out = (offset + 1) * mtx_out.stride;
            row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted, inverse_permutation);
    }



    return;

}



void ConstructMatrixForRecursivePowerTrace(matrix &mtx, PicState_int64& modes, matrix &mtx_out, PicState_int64& repeated_column_pairs) {


    PicState_int64 sorted_modes;
    PicState_int64 inverse_permutation(modes.size());
    SortModes( modes, sorted_modes, inverse_permutation);



    std::vector<pic::SingleMode> single_modes;
    std::vector<pic::ModePair> mode_pairs;
    DetermineModePairs(single_modes, mode_pairs, sorted_modes);

/*
    for (size_t idx=0; idx<single_modes.size(); idx++) {
        std::cout << inverse_permutation[single_modes[idx].mode] << " " << single_modes[idx].degeneracy << std::endl;
    }
    std::cout <<  std::endl;
    for (size_t idx=0; idx<mode_pairs.size(); idx++) {
        std::cout << inverse_permutation[mode_pairs[idx].mode1] << " " << inverse_permutation[mode_pairs[idx].mode2] << " " << mode_pairs[idx].degeneracy << std::endl;
    }
*/

    ConstructRepeatedMatrix(mtx, single_modes, mode_pairs, mtx_out, repeated_column_pairs, inverse_permutation);


    //mtx_out.print_matrix();
    //repeated_column_pairs.print_matrix();


    return;


}

} //PIC
