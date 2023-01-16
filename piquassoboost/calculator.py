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

from piquasso._backends.calculator import NumpyCalculator

from piquassoboost.sampling.Boson_Sampling_Utilities import (
    PowerTraceLoopHafnian,
    GlynnPermanent
)

from theboss.boson_sampling_utilities.boson_sampling_utilities import (
    EffectiveScatteringMatrixCalculator
)

from piquasso._math.hafnian import _reduce_matrix_with_diagonal


def cpp_loop_hafnian(matrix, diagonal, reduce_on):
    reduced_matrix = _reduce_matrix_with_diagonal(matrix, diagonal, reduce_on)

    return PowerTraceLoopHafnian(reduced_matrix).calculate()


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


class BoostCalculator(NumpyCalculator):
    def __init__(self) -> None:
        super().__init__()

        self.permanent = cpp_permanent_function
        self.loop_hafnian = cpp_loop_hafnian
        self.number_of_approximated_modes = None