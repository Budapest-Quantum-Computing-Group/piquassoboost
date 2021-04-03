#include "SymmetrizationChecker.h"


namespace pic {


SymmetrizationChecker::SymmetrizationChecker( matrix &mtx_in, int* result_out ) {
    mtx = mtx_in;
    result = result_out;
}

const tbb::flow::continue_msg&
SymmetrizationChecker::operator()(const tbb::flow::continue_msg &msg) {

    // raw pointer to the stored data in matrix mtx
    Complex16* mtx_data = mtx.get_data();

    int* res = result;
    *res = 1;
    size_t dim_rows = mtx.rows;
    size_t dim_cols = mtx.cols;
    size_t dim_minus_one = dim_rows - 1;
    tbb::parallel_for((size_t)0, dim_minus_one, [mtx_data, dim_rows, dim_cols, res](size_t row_idx) {
        for (size_t col_idx = row_idx + 1; col_idx < dim_cols; col_idx++) {
            size_t elem_idx_u = row_idx * dim_rows + col_idx;
            size_t elem_idx_l = col_idx * dim_rows + row_idx;
            
            Complex16 diff = mtx_data[elem_idx_u] - mtx_data[elem_idx_l];
            if (std::abs(diff) > 0.00001) {
                *res *= 0;
                break;
            }        
        }
    }); // TBB
     
    return msg;
//    *result = *res;
}




} // PIC

