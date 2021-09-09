#
#  Copyright 2021 Budapest Quantum Computing Group
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np

import piquasso as pq

from .state_wrapper import GaussianState_Wrapper

from piquasso._math.linalg import block_reduce
from piquasso.api.result import Result

from piquassoboost.sampling.simulation_strategies.GaussianSimulationStrategy import (
    GaussianSimulationStrategyFast
)

from piquassoboost.sampling.simulation_strategies import ThresholdBosonSampling


class GaussianState(GaussianState_Wrapper, pq.GaussianState):
    def __init__(self, *, d):
        self._d = d

        self.result = None
        self.shots = None

        vector_shape = (self.d, )
        matrix_shape = vector_shape * 2

        super().__init__(
            m=np.zeros(vector_shape, dtype=complex),
            G=np.zeros(matrix_shape, dtype=complex),
            C=np.zeros(matrix_shape, dtype=complex),
        )

    def apply_passive(self, T, modes):
        self._m[modes, ] = T @ self._m[modes, ]
        super().apply_to_C_and_G(T, modes)

    def __deepcopy__(self, memo):
        """
        NOTE: This method is needed to deepcopy the GaussianState by `copy.deepcopy`
        at state registration, see :class:`pq.Program`.
        """
        obj = GaussianState(d=self.d)

        obj._m = np.copy(self._m)
        obj._G = np.copy(self._G)
        obj._C = np.copy(self._C)

        return obj

    def _apply_threshold_measurement(self, *, instruction):
        """
        NOTE: The same logic is used here, as for the particle number measurement.
        However, at threshold measurement there is no sense of cutoff, therefore it is
        set to 2 to make the logic sensible in this case as well.

        Also note, that one could speed up this calculation by not calculating the
        probability of clicks (i.e. 1 as sample), and argue that the probability of a
        click is equal to one minus the probability of no click.
        """
        if not np.allclose(self.xpxp_mean_vector, np.zeros_like(self.xpxp_mean_vector)):
            raise NotImplementedError(
                "Threshold measurement for displaced states are not supported: "
                f"xpxp_mean_vector={self.xpxp_mean_vector}"
            )

        reduced_state = self.reduced(instruction.modes)

        th = ThresholdBosonSampling.ThresholdBosonSampling(
            covariance_matrix=(
                reduced_state.xxpp_covariance_matrix
                / (2 * pq.api.constants.HBAR)
            )
        )
        samples = th.simulate(self.shots)

        self.result = Result(instruction=instruction, samples=samples)


    def _apply_particle_number_measurement(self, *, instruction):

        cutoff: int = instruction._all_params["cutoff"]

        reduced_state = self.reduced(instruction.modes)

        samples = GaussianSimulationStrategyFast(
            covariance_matrix=(
                reduced_state.xpxp_covariance_matrix / (2 * pq.api.constants.HBAR)
            ),
            m=reduced_state.xpxp_mean_vector / np.sqrt(pq.api.constants.HBAR),
            fock_cutoff=cutoff,
        ).simulate(self.shots)

        self.result = Result(instruction=instruction, samples=samples)

def calculate_threshold_detection_probability(
    state,
    subspace_modes,
    occupation_numbers,
):
    d = len(subspace_modes)

    Q = (state.complex_covariance + np.identity(2 * d)) / 2

    OS = (np.identity(2 * d, dtype=complex) - np.linalg.inv(Q)).conj()

    OS_reduced = block_reduce(OS, reduce_on=occupation_numbers)
