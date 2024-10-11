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

from typing import Tuple

import numpy as np

import piquasso as pq

from .calculation_extension import apply_passive_linear_to_C_and_G

from piquasso.api.result import Result
from piquasso._simulators.gaussian import calculations as pq_calculations

from piquassoboost.sampling.simulation_strategies import ThresholdBosonSampling


def passive_linear(
    state, instruction, shots
):
    modes = instruction.modes
    T: np.ndarray = instruction._get_passive_block(state._calculator, state._config)

    # NOTE: We need to explicitely cast complex64 to complex128 for piquassoboost to
    # handle it correctly, and then cast it back.
    dtype_is_complex64 = state._C.dtype == np.complex64

    if dtype_is_complex64:
        state._C = state._C.astype(np.complex128)
        state._G = state._G.astype(np.complex128)
        T = T.astype(np.complex128)

    state._m[(modes,)] = (
        T
        @ state._m[
            modes,
        ]
    )

    apply_passive_linear_to_C_and_G(
        C=state._C, G=state._G,
        T=T, modes=modes,
    )

    if dtype_is_complex64:
        state._C = state._C.astype(np.complex64)
        state._G = state._G.astype(np.complex64)
        T = T.astype(np.complex64)

    return Result(state=state)


def threshold_measurement(state, instruction, shots):
    """
    NOTE: This function calculates only by using torontonian.
    """

    if not np.allclose(state.xpxp_mean_vector, np.zeros_like(state.xpxp_mean_vector)):
        raise NotImplementedError(
            "Threshold measurement for displaced states are not supported: "
            f"xpxp_mean_vector={state.xpxp_mean_vector}"
        )

    reduced_state = state.reduced(instruction.modes)

    th = ThresholdBosonSampling.ThresholdBosonSampling(
        covariance_matrix=(
            reduced_state.xxpp_covariance_matrix
            / (2 * state._config.hbar)
        )
    )
    samples = th.simulate(shots)

    return Result(state=state, samples=samples)


def particle_number_measurement(state, instruction, shots) -> Result:
    samples = pq_calculations._get_particle_number_measurement_samples(
        state, instruction, shots
    )

    return Result(state=state, samples=samples)
