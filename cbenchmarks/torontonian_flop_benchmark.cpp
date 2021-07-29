#include "Torontonian.h"
#include "TorontonianRecursive.h"
#include "matrix.h"
#include "../ctests/matrix_helper.hpp"

#include <math.h>


namespace pic {


extern long int flops;


void print(std::list<double> const &list)
{
    printf("[");
    for (auto const &i: list) {
        std::cout << i << ", ";
    }
    printf("]\n");
}


void calculate_torontonian_flops(){
    size_t dim;

    std::list<size_t> modes;

    std::list<double> log_modes;

    std::list<double> regular_flops;
    std::list<double> recursive_flops;

    for (dim = 54; dim <= 60; dim += 2)
    {
        flops = 0;

        modes.push_back(dim);
        log_modes.push_back(log((double) dim));

        pic::matrix complex_matrix = pic::get_random_density_matrix_complex<pic::matrix, pic::Complex16>(dim);

        pic::Torontonian torontonian_calculator(complex_matrix);
        torontonian_calculator.calculate();

        regular_flops.push_back(log(flops) - dim / 2 * log(2));

        flops = 0;

        pic::matrix_real real_matrix = pic::get_random_density_matrix_real<pic::matrix_real, double>(dim);
        pic::TorontonianRecursive recursive_torontonian_calculator(real_matrix);
        recursive_torontonian_calculator.calculate(/* use_extended = */ false);

        recursive_flops.push_back(log(flops) - dim / 2 * log(2));

        std::cout << "." << std::flush;
    }

    printf("\n");

    print(log_modes);

    print(regular_flops);
    print(recursive_flops);
}


} // PIC


int main() {
    pic::calculate_torontonian_flops();
}