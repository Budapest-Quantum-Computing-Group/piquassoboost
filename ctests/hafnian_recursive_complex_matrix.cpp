#include <stdio.h>
#include <vector>
#include <random>
#include <time.h>
#include <vector>


#include "matrix.h"
#include "PowerTraceHafnian.h"
#include "PowerTraceHafnianRecursive.h"


#include "tbb/tbb.h"


/**
@brief Call to calculate sum of integers stored in a PicState
@param vec a container if integers
@return Returns with the sum of the elements of the container
*/
static inline int64_t
sum( pic::PicState_int64 &vec) {

    int64_t ret=0;

    size_t element_num = vec.cols;
    int64_t* data = vec.get_data();
    for (size_t idx=0; idx<element_num; idx++ ) {
        ret = ret + data[idx];
    }
    return ret;
}



pic::matrix
create_repeated_mtx( pic::matrix& A, pic::PicState_int64& filling_factors ) {

    size_t dim_A_S = sum(filling_factors);
    size_t dim_A = filling_factors.size();

    pic::matrix A_S(2*dim_A_S, 2*dim_A_S);
    size_t row_idx = 0;
    for (size_t idx=0; idx<filling_factors.size(); idx++) {
        for (size_t row_repeat=0; row_repeat<filling_factors[idx]; row_repeat++) {

            size_t row_offset = row_idx*A_S.stride;
            size_t row_offset_A = idx*A.stride;
            size_t col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<filling_factors.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<filling_factors[jdx]; col_repeat++) {
                    A_S[row_offset + col_idx] = A[row_offset_A + jdx];
                    col_idx++;
                }
            }

            col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<filling_factors.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<filling_factors[jdx]; col_repeat++) {
                    A_S[row_offset + col_idx + dim_A_S] = A[row_offset_A + jdx + dim_A];
                    col_idx++;
                }

            }

            row_offset = (row_idx+dim_A_S)*A_S.stride;
            row_offset_A = (idx+dim_A)*A.stride;
            col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<filling_factors.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<filling_factors[jdx]; col_repeat++) {
                    A_S[row_offset + col_idx] = A[row_offset_A + jdx];
                    col_idx++;
                }
            }

            col_idx = 0;
            // insert column elements
            for (size_t jdx=0; jdx<filling_factors.size(); jdx++) {
                for (size_t col_repeat=0; col_repeat<filling_factors[jdx]; col_repeat++) {
                    A_S[row_offset + col_idx + dim_A_S] = A[row_offset_A + jdx + dim_A];
                    col_idx++;
                }

            }


            row_idx++;
        }


    }

    return A_S;

}


/**
@brief Transforms the covariance matrix in the basis \f$ a_1, a_2, ... a_n, a_1^*, a_2^*, ...  a_n^* \f$  into the basis \f$ a_1, a_1^*,a_2, a_2^*, ... a_n, a_n^* \f$ suitable for
the PowerTraceHafnianRecursive algorithm.
@param mtx A covariance matrix in the basis \f$ a_1, a_2, ... a_n,, a_1^*, a_2^*, ...  a_n^* \f$.
@return Returns with the covariance matrix in the basis \f$ a_1, a_1^*,a_2, a_2^*, ... a_n, a_n^* \f$.
*/
pic::matrix
getPermutedMatrix( pic::matrix& mtx) {


    pic::matrix res(mtx.rows, mtx.cols);


    size_t num_of_modes = mtx.rows/2;

    for (size_t row_idx=0; row_idx<num_of_modes; row_idx++ ) {

        size_t row_offset_q_orig = row_idx*mtx.stride;
        size_t row_offset_p_orig = (row_idx+num_of_modes)*mtx.stride;

        size_t row_offset_q_permuted = 2*row_idx*res.stride;
        size_t row_offset_p_permuted = (2*row_idx+1)*res.stride;

        for (size_t col_idx=0; col_idx<num_of_modes; col_idx++ ) {

            res[row_offset_q_permuted + col_idx*2] = mtx[row_offset_q_orig + col_idx];
            res[row_offset_q_permuted + col_idx*2 + 1] = mtx[row_offset_q_orig + num_of_modes + col_idx];

            res[row_offset_p_permuted + col_idx*2] = mtx[row_offset_p_orig + col_idx];
            res[row_offset_p_permuted + col_idx*2 + 1] = mtx[row_offset_p_orig + num_of_modes + col_idx];

        }

    }

    //res.print_matrix();

    return res;

}


////////////////////////////
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

SingleMode() {

}

SingleMode(size_t mode_, size_t degeneracy_) {
    mode = mode_;
    degeneracy = degeneracy_;

}

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
    /// The degeneracy (occupancy) of the given mode pair
    size_t degeneracy;

ModePair() {

}

ModePair(size_t mode1_, size_t mode2_, size_t degeneracy_) {
    mode1 = mode1_;
    mode2 = mode2_;
    degeneracy = degeneracy_;

}

};


/**
@brief Call to determine mode pairs and single modes for the recursive hafnian algorithm
@param single_modes The single modes are returned by this vector
@param mode_pairs The mode pairs are returned by this vector
@param single_modes The mode occupancy for which the mode pairs are determined.
*/
void DetermineModePairs(std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs, PicState_int64& modes ) {

modes.print_matrix();

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


    if ( modes_pair.mode1 > 0 ) {
        for (size_t idx=0; idx<modes[modes_pair.mode1]; idx++) {
            single_modes.push_back( SingleMode(modes_pair.mode1, 1));
        }
//modes_loc[modes_pair.mode1] = 0;
//modes_loc.print_matrix();
    }


    return;
}

void PermuteRow( matrix row, std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs, matrix& row_permuted) {

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

        row_permuted[4*idx] = row[mode_pair.mode1];
        row_permuted[4*idx+1] = row[mode_pair.mode2];
        row_permuted[4*idx+2] = row[mode_pair.mode1 + dim_over_2];
        row_permuted[4*idx+3] = row[mode_pair.mode2 + dim_over_2];

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
            row_permuted[offset]   = row[mode_pair.mode1];
            row_permuted[offset+1] = row[mode_pair.mode2];
            row_permuted[offset+2] = row[mode_pair.mode1 + dim_over_2];
            row_permuted[offset+3] = row[mode_pair.mode2 + dim_over_2];
            offset = offset + 4;
            tmp = 0;
        }


        idx++;
    }


    if (tmp == 1) {
        row_permuted[offset]   = row[mode_pair.mode1];
        row_permuted[offset+1] = row[mode_pair.mode1 + dim_over_2];
    }


    return;

}


void ConstructRepeatedMatrix(matrix &mtx, std::vector<SingleMode>& single_modes, std::vector<ModePair>& mode_pairs, matrix &mtx_out, pic::PicState_int64& repeated_column_pairs) {

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

std::cout << "dim: " << dim << std::endl;

    // preallocate the output arrays
    mtx_out = matrix(dim, dim);
    repeated_column_pairs = PicState_int64(dim_over_2, 1);

for (size_t idx=0; idx<mtx_out.size(); idx++) {
    mtx_out[idx] = Complex16(0.0, 0.0);
}


    size_t tmp = 0;
    for (size_t idx=0; idx<mode_pairs.size(); idx++) {

        ModePair& mode_pair = mode_pairs[idx];

        repeated_column_pairs[2*idx] =  mode_pair.degeneracy;
        repeated_column_pairs[2*idx+1] =  mode_pair.degeneracy;

        size_t row_offset = mode_pair.mode1 * mtx.stride;
        matrix row( mtx.get_data() + row_offset, 1, mtx.cols );

        size_t row_offset_out = 4*idx * mtx_out.stride;
        matrix row_permuted( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, single_modes, mode_pairs, row_permuted);

        ///////////////////////////////////////////////////////////////////////////

        row_offset = mode_pair.mode2 * mtx.stride;
        row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

        row_offset_out = (4*idx+1) * mtx_out.stride;
        row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, single_modes, mode_pairs, row_permuted);


        ///////////////////////////////////////////////////////////////////////////

        row_offset = (mode_pair.mode1+mtx.rows/2) * mtx.stride;
        row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

        row_offset_out = (4*idx+2) * mtx_out.stride;
        row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, single_modes, mode_pairs, row_permuted);


        ///////////////////////////////////////////////////////////////////////////

        row_offset = (mode_pair.mode2+mtx.rows/2) * mtx.stride;
        row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

        row_offset_out = (4*idx+3) * mtx_out.stride;
        row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

        // permute the selected row according to the single modes and mode pairs
        PermuteRow( row, single_modes, mode_pairs, row_permuted);

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
            size_t row_offset = mode_pair.mode1 * mtx.stride;
            matrix row( mtx.get_data() + row_offset, 1, mtx.cols );

            size_t row_offset_out = (offset) * mtx_out.stride;
            matrix row_permuted( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted);

            ///////////////////////////////////////////////////////////////

            row_offset = (mode_pair.mode2) * mtx.stride;
            row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

            row_offset_out = (offset+1) * mtx_out.stride;
            row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted);


            ///////////////////////////////////////////////////////////////

            row_offset = (mode_pair.mode1+mtx.rows/2) * mtx.stride;
            row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

            row_offset_out = (offset+2) * mtx_out.stride;
            row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted);


            ///////////////////////////////////////////////////////////////

            row_offset = (mode_pair.mode2+mtx.rows/2) * mtx.stride;
            row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

            row_offset_out = (offset+3) * mtx_out.stride;
            row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted);

            offset = offset + 4;


            tmp = 0;
        }


        idx++;
    }


    if (tmp == 1) {
        size_t row_offset = mode_pair.mode1 * mtx.stride;
            matrix row( mtx.get_data() + row_offset, 1, mtx.cols );

            size_t row_offset_out = offset * mtx_out.stride;
            matrix row_permuted( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted);


            ///////////////////////////////////////////////////////////////

            row_offset = (mode_pair.mode1+mtx.rows/2) * mtx.stride;
            row = matrix( mtx.get_data() + row_offset, 1, mtx.cols );

            row_offset_out = (offset + 1) * mtx_out.stride;
            row_permuted = matrix( mtx_out.get_data() + row_offset_out, 1, mtx_out.cols );

            // permute the selected row according to the single modes and mode pairs
            PermuteRow( row, single_modes, mode_pairs, row_permuted);
    }









    return;

}


} //PIC

/////////////////////////////





/**
@brief Unit test case for the recursive hafnian of complex symmetric matrices
*/
int main() {

    printf("\n\n****************************************\n");
    printf("Test of hafnian of random complex random matrix\n");
    printf("****************************************\n\n\n");

    // seed the random generator
    srand ( time ( NULL));


    // allocate matrix array for the larger matrix
    size_t dim = 40;
    pic::matrix mtx = pic::matrix(dim, dim);

    // fill up matrix with random elements
    for (size_t row_idx = 0; row_idx < dim; row_idx++) {
        for (size_t col_idx = 0; col_idx <= row_idx; col_idx++) {
            if ( row_idx == col_idx ) continue;

            double randnum1 = ((double)rand()/RAND_MAX*2 - 1.0)*10;
            double randnum2 = ((double)rand()/RAND_MAX*2 - 1.0)*10;
            mtx[row_idx * dim + col_idx] = pic::Complex16(randnum1, randnum2);
            mtx[col_idx* dim + row_idx] = mtx[row_idx * dim + col_idx];
        }
    }


    // array of modes describing the occupancy of the individual modes
    pic::PicState_int64 filling_factors(dim/2);
    for (size_t idx=0; idx<filling_factors.size(); idx++) {
        filling_factors[idx] = 1;
    }

    filling_factors[0] = 2;
    filling_factors[1] = 1;
    filling_factors[2] = 2;
    filling_factors[3] = 0;
    filling_factors[4] = 2;
    filling_factors[5] = 0;
    filling_factors[6] = 1;
    filling_factors[7] = 2;
    filling_factors[8] = 2;

    filling_factors[9] = 0;
    filling_factors[10] = 1;
    filling_factors[11] = 0;
    filling_factors[12] = 5;
    filling_factors[12] = 0;
    filling_factors[12] = 2;
    filling_factors[14] = 1;
    filling_factors[15] = 0;
    filling_factors[17] = 1;
    filling_factors[18] = 0;
    filling_factors[19] = 4;

    // matrix containing the repeated rows and columns
    pic::matrix&& mtx_repeated = create_repeated_mtx(mtx, filling_factors);

    // reorder the input matrix
    pic::matrix &&mtx_permuted = getPermutedMatrix(mtx);


///////////////////////////////////

FILE *fp = fopen("bad_matrix", "rb");
size_t rows, cols;
fread( &rows, sizeof(size_t), 1, fp);
fread( &cols, sizeof(size_t), 1, fp);
pic::matrix A(rows, cols);
fread( A.get_data(), sizeof(pic::Complex16), A.size(), fp);

size_t num_of_modes;
fread( &num_of_modes, sizeof(size_t), 1, fp);
pic::PicState_int64 modes(num_of_modes+1);
fread( modes.get_data(), sizeof(int64_t), modes.size(), fp);
modes[modes.size()-1] = 3;
fclose(fp);

/*
std::cout << rows << " " << cols << std::endl;
A.print_matrix();
std::cout << num_of_modes << std::endl;
modes.print_matrix();

/*
pic::matrix A = mtx;
pic::PicState_int64 modes = filling_factors;
*/

mtx_repeated = create_repeated_mtx(A, modes);
mtx_permuted = getPermutedMatrix(A);


pic::PicState_int64 modes2(mtx_repeated.rows/2, 1);

std::vector<pic::SingleMode> single_modes;
std::vector<pic::ModePair> mode_pairs;
DetermineModePairs(single_modes, mode_pairs, modes);


for (size_t idx=0; idx<single_modes.size(); idx++) {
    std::cout << single_modes[idx].mode << " " << single_modes[idx].degeneracy << std::endl;
}
std::cout <<  std::endl;
for (size_t idx=0; idx<mode_pairs.size(); idx++) {
    std::cout << mode_pairs[idx].mode1 << " " << mode_pairs[idx].mode2 << " " << mode_pairs[idx].degeneracy << std::endl;
}


pic::matrix mtx_out;
pic::PicState_int64 repeated_column_pairs;
ConstructRepeatedMatrix(A, single_modes, mode_pairs, mtx_out, repeated_column_pairs);


//mtx_out.print_matrix();
repeated_column_pairs.print_matrix();

mtx_permuted = mtx_out;
modes = repeated_column_pairs;

//////



    // print the matrix on standard output
    //mtx.print_matrix();
    //mtx_repeated=getPermutedMatrix(mtx_repeated);
    //mtx_repeated.print_matrix();

    // hafnian calculated by algorithm PowerTraceHafnian
    tbb::tick_count t0 = tbb::tick_count::now();
    pic::PowerTraceHafnian hafnian_calculator = pic::PowerTraceHafnian( mtx_repeated );
    pic::Complex16 hafnian_powertrace = hafnian_calculator.calculate();
    tbb::tick_count t1 = tbb::tick_count::now();

    // calculate the hafnian by the recursive method

    // now calculated the hafnian of the whole matrix using the value calculated for the submatrix
    tbb::tick_count t2 = tbb::tick_count::now();
    pic::PowerTraceHafnianRecursive hafnian_calculator_recursive = pic::PowerTraceHafnianRecursive( mtx_permuted, modes );
    pic::Complex16 hafnian_powertrace_recursive = hafnian_calculator_recursive.calculate();
    tbb::tick_count t3 = tbb::tick_count::now();


    std::cout << "the calculated hafnian with the power trace method: " << hafnian_powertrace << std::endl;
    std::cout << "the calculated hafnian with the recursive powertrace method: " << hafnian_powertrace_recursive << std::endl;


    std::cout << (t1-t0).seconds() << " " << (t3-t2).seconds() << " " << (t1-t0).seconds()/(t3-t2).seconds() << std::endl;




  return 0;

};
