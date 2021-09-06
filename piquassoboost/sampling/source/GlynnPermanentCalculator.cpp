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

    vector<int> subproblem_positions = compute_grey_code_update_positions(n - 1);
    vector<int> positions(subproblem_positions) ;
    positions.push_back(n - 1);
    std::reverse(subproblem_positions.begin(), subproblem_positions.end());
    positions.insert(positions.begin(), subproblem_positions.begin(), subproblem_positions.end());

    return positions;
}

Complex16
GlynnPermanentCalculator::calculate(matrix &mtx) {

    auto n = mtx.rows;

    auto multiplier = n % 2 == 0 ? 1 : -1;

    vector<Complex16> sums = {};

    // Initialize the sums
    for(auto i = 0; i < n; ++i){
        Complex16 ith_sum(0.0, 0.0);
        for(auto j = 0; j < n; ++j){
            ith_sum -= mtx[i][j];
        }
        sums.append(ith_sum);
    }

    // Initialize the permanent
    Complex16 perm = std::accumulate(sums.begin(),
                                     sums.end(),
                                     Complex16(multiplier, 0.0),
                                     std::multiplies<Complex16>{});

    // Now update the sums in Gray ordering.
    auto update_positions = compute_grey_code_update_positions(n);
    vector<int> current_code = {};
    while(current_code.size() < n){ current_code.push_back(-1); }

    for(auto i : update_positions){
        // Update the code and it's product
        current_code[i] = -current_code[i];
        multiplier = -multiplier;
        // Update the sums
        for(auto j = 0; j < n; ++j) {
            sums[i] += 2 * current_code[i] * mtx[i][j];
        }

        perm += std::accumulate(sums.begin(),
                               sums.end(),
                               Complex16(multiplier, 0.0),
                               std::multiplies<Complex16>{});

    }

    perm /= 2 << (n - 1);

    return perm;


}



} // PIC
