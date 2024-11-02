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

from piquasso.api.result import Result
from piquasso._simulators.gaussian import calculations as pq_calculations

from piquassoboost.sampling.simulation_strategies import ThresholdBosonSampling


def threshold_measurement(state, instruction, shots):
    """
    NOTE: This function calculates only by using torontonian.
    """

    if not np.allclose(state.xpxp_mean_vector, np.zeros_like(state.xpxp_mean_vector)):
        return pq_calculations.threshold_measurement(state, instruction, shots)

    reduced_state = state.reduced(instruction.modes)

    th = ThresholdBosonSampling.ThresholdBosonSampling(
        covariance_matrix=(
            reduced_state.xxpp_covariance_matrix / (2 * state._config.hbar)
        )
    )
    samples = th.simulate(shots)

    return Result(state=state, samples=samples)
