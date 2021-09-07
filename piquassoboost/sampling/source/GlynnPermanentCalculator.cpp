#include <iostream>
#include "GlynnPermanentCalculator.h"
#include <tbb/scalable_allocator.h>
#include "tbb/tbb.h"
#include "common_functionalities.h"
#include <math.h>
#include <algorithm>
#include <numeric>
#include <functional>

namespace pic {




/**
@brief Default constructor of the class.
@return Returns with the instance of the class.
*/
GlynnPermanentCalculator::GlynnPermanentCalculator() {}




/**
@brief Call to calculate the permanent of a given matrix.

@param mtx_in A square matrix.

@return Returns with the calculated permanent
*/

std::vector<int> compute_grey_code_update_positions(const int &n){

    if(n == 1){
        return {0};
    }

    std::vector<int> subproblem_positions = compute_grey_code_update_positions(n - 1);
    std::vector<int> positions(subproblem_positions) ;
    positions.push_back(n - 1);
    std::reverse(subproblem_positions.begin(), subproblem_positions.end());
    positions.insert(positions.end(), subproblem_positions.begin(), subproblem_positions.end());

    return positions;
}

Complex16
GlynnPermanentCalculator::calculate(matrix &mtx) {

    auto n = mtx.rows;

    double multiplier = n % 2 == 0 ? -1 : 1;

    std::vector<Complex16> sums = {};

    // Initialize current code
    std::vector<int> current_code = {};
    while(current_code.size() < n){ current_code.push_back(-1); }
    current_code[n - 1] = 1;

    // Initialize the sums
    for(size_t j = 0; j < n; ++j){
        Complex16 jth_sum(0.0, 0.0);
        for(size_t i = 0; i < n; ++i){
            jth_sum += current_code[i] * mtx[i * mtx.stride + j];
        }
        sums.push_back(jth_sum);
    }

    Complex16 sums_prod(1.0, 0.0);
    for(auto jth_sum : sums){ sums_prod *= jth_sum; }

    Complex16 perm = sums_prod * multiplier;

    // Now update the sums in Gray ordering.
    auto update_positions = compute_grey_code_update_positions(n - 1);

    for(auto i : update_positions){
        // Update the code and it's product
        current_code[i] = -current_code[i];
        multiplier = -multiplier;
        
        // Update the sums
        for(size_t j = 0; j < n; ++j){
            sums[j] += current_code[i] * 2 * mtx[i * mtx.stride + j];
        }

        sums_prod = Complex16(1.0, 0.0);
        for(auto jth_sum : sums){ sums_prod *= jth_sum; }

        perm += multiplier * sums_prod;
    }

    perm /= 1 << (n - 1);

    return perm;
}



} // PIC
