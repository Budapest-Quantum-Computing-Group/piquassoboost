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

from piquasso.api.config import Config

from piquassoboost.sampling.Boson_Sampling_Utilities import (
    PowerTraceLoopHafnian,
    GlynnPermanent
)

from theboss.boson_sampling_utilities.boson_sampling_utilities import (
    EffectiveScatteringMatrixCalculator
)


def cpp_loop_hafnian(matrix):
    return PowerTraceLoopHafnian(matrix).calculate()


def cpp_permanent_function(matrix, input, output):
    scattering_matrix = EffectiveScatteringMatrixCalculator(
        matrix, input, output
    ).calculate()

    calculator = GlynnPermanent(scattering_matrix)

    return calculator.calculate()


class BoostConfig(Config):
    def __init__(
        self,
        loop_hafnian_function=cpp_loop_hafnian,
        permanent_function=cpp_permanent_function,
        **kwargs,
    ) -> None:
        super().__init__(
            loop_hafnian_function=loop_hafnian_function,
            permanent_function=permanent_function,
            **kwargs,
        )