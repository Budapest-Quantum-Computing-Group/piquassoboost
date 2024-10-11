#
# Copyright 2021-2022 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from piquasso._simulators.connectors import NumpyConnector

from piquasso._math.hafnian.utils import match_occupation_numbers, ix_

from piquassoboost.sampling.Boson_Sampling_Utilities import (
    PowerTraceHafnianRecursive,
    PowerTraceLoopHafnianRecursive,
    GlynnPermanent
)

from theboss.boson_sampling_utilities.boson_sampling_utilities import (
    EffectiveScatteringMatrixCalculator
)


def cpp_hafnian(matrix_orig, diagonal_orig, occupation_numbers):
    all_edges, edge_indices = match_occupation_numbers(occupation_numbers)

    matrix = ix_(matrix_orig, edge_indices, edge_indices)
    diagonal = diagonal_orig[edge_indices]

    np.fill_diagonal(matrix, diagonal)

    return PowerTraceHafnianRecursive(matrix, all_edges).calculate()



def cpp_loop_hafnian(matrix_orig, diagonal_orig, occupation_numbers):
    n = sum(occupation_numbers)

    diagonal_orig = diagonal_orig.astype(matrix_orig.dtype)

    if n == 0:
        return 1.0
    elif n % 2 == 1:
        # Handling the odd case by extending the matrix with a 1 to be even.
        #
        # TODO: This is not the best handling of the odd case, and we can definitely
        # squeeze out a bit more performance if needed.
        dim = len(matrix_orig)
        new_matrix = np.empty((dim + 1, dim + 1), dtype=matrix_orig.dtype)
        new_matrix[1:, 1:] = matrix_orig
        new_matrix[0, 0] = 1.0
        new_matrix[1:, 0] = 0.0
        new_matrix[0, 1:] = 0.0

        matrix_orig = new_matrix

        d = len(occupation_numbers)

        new_occupation_numbers = np.empty(d + 1, dtype=occupation_numbers.dtype)
        new_diagonal = np.empty(d + 1, dtype=diagonal_orig.dtype)

        new_occupation_numbers[0] = 1
        new_diagonal[0] = 1.0

        for i in range(d):
            new_occupation_numbers[i + 1] = occupation_numbers[i]
            new_diagonal[i + 1] = diagonal_orig[i]

        occupation_numbers = new_occupation_numbers
        diagonal_orig = new_diagonal

    all_edges, edge_indices = match_occupation_numbers(occupation_numbers)

    matrix = ix_(matrix_orig, edge_indices, edge_indices)
    diagonal = diagonal_orig[edge_indices]

    np.fill_diagonal(matrix, diagonal)

    return PowerTraceLoopHafnianRecursive(matrix, all_edges).calculate()


def cpp_permanent_function(matrix, input, output):
    scattering_matrix = EffectiveScatteringMatrixCalculator(
        matrix, input, output
    ).calculate()

    calculator = GlynnPermanent(scattering_matrix)

    result = calculator.calculate()

    if isinstance(result, list):
        # TODO: Here an empty list is passed instead of 1.0.
        result = 1.0

    return result


class BoostConnector(NumpyConnector):
    def __init__(self) -> None:
        super().__init__()

        self.permanent = cpp_permanent_function
        self.loop_hafnian = cpp_loop_hafnian
        self.number_of_approximated_modes = None