#include "GlynnPermanentCalculatorDFE.h"


#ifndef CPYTHON
#include <tbb/tbb.h>
#endif


namespace pic {

/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculator_DFEDualCard(matrix& matrix_mtx, Complex16& perm)
{


    Complex16* mtx_data = matrix_mtx.get_data();
    

    // calulate the maximal sum of the columns to normalize the matrix
    matrix colSumMax( matrix_mtx.cols, 1);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex16) );
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            Complex16 value1 = colSumMax[jdx] + matrix_mtx[ idx*matrix_mtx.stride + jdx];
            Complex16 value2 = colSumMax[jdx] - matrix_mtx[ idx*matrix_mtx.stride + jdx];
            if ( std::abs( value1 ) < std::abs( value2 ) ) {
                colSumMax[jdx] = value2;
            }
            else {
                colSumMax[jdx] = value1;
            }

        }

    }




    // calculate the renormalization coefficients
    matrix_base<double> renormalize_data(matrix_mtx.cols, 1);
    for (int jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(colSumMax[jdx]);
    }

    // renormalize the input matrix
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            matrix_mtx[ idx*matrix_mtx.stride + jdx] = matrix_mtx[ idx*matrix_mtx.stride + jdx]/renormalize_data[jdx];
        }

    }    


    // SLR and DFE split input matrices
    matrix mtx_split[8];
    Complex16* mtx_data_split[8];


    size_t max_fpga_rows =  MAX_FPGA_DIM;
    size_t max_fpga_cols =  MAX_FPGA_DIM/8;

    // SLR splitted data for the first DFE card
    size_t cols_half1_tot = matrix_mtx.cols/2;
    size_t cols_half2_tot = matrix_mtx.cols - cols_half1_tot;

    size_t rows = matrix_mtx.rows;
    size_t cols_half1[4];
    cols_half1[0] = max_fpga_cols < cols_half1_tot ? max_fpga_cols : cols_half1_tot;
    cols_half1[1] = max_fpga_cols < (cols_half1_tot -cols_half1[0]) ? max_fpga_cols : (cols_half1_tot-cols_half1[0]);
    cols_half1[2] = max_fpga_cols < (cols_half1_tot - cols_half1[0] - cols_half1[1]) ? max_fpga_cols : (cols_half1_tot - cols_half1[0] - cols_half1[1]);
    cols_half1[3] = max_fpga_cols < (cols_half1_tot - cols_half1[0] - cols_half1[1] - cols_half1[2]) ? max_fpga_cols : (cols_half1_tot - cols_half1[0] - cols_half1[1] - cols_half1[2]);


    size_t col_offset = 0;
    for (int kdx=0; kdx<4; kdx++) {

        mtx_split[kdx] = matrix(max_fpga_rows, max_fpga_cols);
        mtx_data_split[kdx] = mtx_split[kdx].get_data();

        Complex16 padding_element(1.0,0.0);
        for (size_t idx=0; idx<rows; idx++) {
            size_t offset = idx*matrix_mtx.stride+col_offset;
            size_t offset_small = idx*mtx_split[kdx].stride;
            for (size_t jdx=0; jdx<cols_half1[kdx]; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = matrix_mtx[offset+jdx];
            }

            for (size_t jdx=cols_half1[kdx]; jdx<max_fpga_cols; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = padding_element;
            }
            padding_element.real(0.0);
        }
        col_offset = col_offset + cols_half1[kdx];

        memset( mtx_data_split[kdx] + rows*mtx_split[kdx].stride, 0.0, (max_fpga_rows-rows)*max_fpga_cols*sizeof(Complex16));
    }


    // SLR splitted data for the second DFE card
    size_t cols_half2[4];
    cols_half2[0] = max_fpga_cols < cols_half2_tot ? max_fpga_cols : cols_half2_tot;
    cols_half2[1] = max_fpga_cols < (cols_half2_tot - cols_half2[0]) ? max_fpga_cols : (cols_half2_tot - cols_half2[0]);
    cols_half2[2] = max_fpga_cols < (cols_half2_tot - cols_half2[0] - cols_half2[1]) ? max_fpga_cols : (cols_half2_tot - cols_half2[0] - cols_half2[1]);
    cols_half2[3] = max_fpga_cols < (cols_half2_tot - cols_half2[0] - cols_half2[1] - cols_half2[2]) ? max_fpga_cols : (cols_half2_tot - cols_half2[0] - cols_half2[1] - cols_half2[2]);

    for (int kdx=0; kdx<4; kdx++) {

        mtx_split[kdx+4] = matrix(max_fpga_rows, max_fpga_cols);
        mtx_data_split[kdx+4] = mtx_split[kdx+4].get_data();

        Complex16 padding_element(1.0,0.0);
        for (size_t idx=0; idx<rows; idx++) {
            size_t offset = idx*matrix_mtx.stride+col_offset;
            size_t offset_small = idx*mtx_split[kdx+4].stride;
            for (size_t jdx=0; jdx<cols_half2[kdx]; jdx++) {
                mtx_data_split[kdx+4][offset_small+jdx] = matrix_mtx[offset+jdx];
            }

            for (size_t jdx=cols_half2[kdx]; jdx<max_fpga_cols; jdx++) {
                mtx_data_split[kdx+4][offset_small+jdx] = padding_element;
            }
            padding_element.real(0.0);

        }
        col_offset = col_offset + cols_half2[kdx];
        memset( mtx_data_split[kdx+4] + rows*mtx_split[kdx+4].stride, 0.0, (max_fpga_rows-rows)*max_fpga_cols*sizeof(Complex16));
    }


/*
matrix_mtx.print_matrix();
for (int idx=0; idx<8; idx++) {
   mtx_split[idx].print_matrix();
}
*/
    
    calcPermanentGlynn_DualDFE( mtx_data_split, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, &perm);


    return;
}



/**
@brief Wrapper function to call the calculate the Permanent on a DFE
*/
void
GlynnPermanentCalculator_DFESingleCard(matrix& matrix_mtx, Complex16& perm) {


    Complex16* mtx_data = matrix_mtx.get_data();
    

    // calulate the maximal sum of the columns to normalize the matrix
    matrix colSumMax( matrix_mtx.cols, 1);
    memset( colSumMax.get_data(), 0.0, colSumMax.size()*sizeof(Complex16) );
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            Complex16 value1 = colSumMax[jdx] + matrix_mtx[ idx*matrix_mtx.stride + jdx];
            Complex16 value2 = colSumMax[jdx] - matrix_mtx[ idx*matrix_mtx.stride + jdx];
            if ( std::abs( value1 ) < std::abs( value2 ) ) {
                colSumMax[jdx] = value2;
            }
            else {
                colSumMax[jdx] = value1;
            }

        }

    }




    // calculate the renormalization coefficients
    matrix_base<double> renormalize_data(matrix_mtx.cols, 1);
    for (int jdx=0; jdx<matrix_mtx.cols; jdx++ ) {
        renormalize_data[jdx] = std::abs(colSumMax[jdx]);
    }

    // renormalize the input matrix
    for (int idx=0; idx<matrix_mtx.rows; idx++) {
        for( int jdx=0; jdx<matrix_mtx.cols; jdx++) {
            matrix_mtx[ idx*matrix_mtx.stride + jdx] = matrix_mtx[ idx*matrix_mtx.stride + jdx]/renormalize_data[jdx];
        }

    }    

    // SLR and DFE split input matrices
    matrix mtx_split[4];
    Complex16* mtx_data_split[4];


    size_t max_fpga_rows =  MAX_SINGLE_FPGA_DIM;
    size_t max_fpga_cols =  MAX_SINGLE_FPGA_DIM/4;

    // SLR splitted data for the DFE card

    size_t rows = matrix_mtx.rows;
    size_t cols_split[4];
    cols_split[0] = max_fpga_cols < matrix_mtx.cols ? max_fpga_cols : matrix_mtx.cols;
    cols_split[1] = max_fpga_cols < (matrix_mtx.cols-cols_split[0]) ? max_fpga_cols : (matrix_mtx.cols-cols_split[0]);
    cols_split[2] = max_fpga_cols < (matrix_mtx.cols - cols_split[0] - cols_split[1]) ? max_fpga_cols : (matrix_mtx.cols - cols_split[0] - cols_split[1]);
    cols_split[3] = max_fpga_cols < (matrix_mtx.cols - cols_split[0] - cols_split[1] - cols_split[2]) ? max_fpga_cols : (matrix_mtx.cols - cols_split[0] - cols_split[1] - cols_split[2]);


    size_t col_offset = 0;
    for (int kdx=0; kdx<4; kdx++) {

        mtx_split[kdx] = matrix(max_fpga_rows, max_fpga_cols);
        mtx_data_split[kdx] = mtx_split[kdx].get_data();

        Complex16 padding_element(1.0,0.0);
        for (size_t idx=0; idx<rows; idx++) {
            size_t offset = idx*matrix_mtx.stride+col_offset;
            size_t offset_small = idx*mtx_split[kdx].stride;
            for (size_t jdx=0; jdx<cols_split[kdx]; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = matrix_mtx[offset+jdx];
            }

            for (size_t jdx=cols_split[kdx]; jdx<max_fpga_cols; jdx++) {
                mtx_data_split[kdx][offset_small+jdx] = padding_element;
            }
            padding_element.real(0.0);
        }
        col_offset = col_offset + cols_split[kdx];

        memset( mtx_data_split[kdx] + rows*mtx_split[kdx].stride, 0.0, (max_fpga_rows-rows)*max_fpga_cols*sizeof(Complex16));
    }
/*
matrix_mtx.print_matrix();
for (int idx=0; idx<4; idx++) {
   mtx_split[idx].print_matrix();
}
*/

    calcPermanentGlynn_SingleDFE( mtx_data_split, renormalize_data.get_data(), matrix_mtx.rows, matrix_mtx.cols, &perm);


    return;

}





}
